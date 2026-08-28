"""A decoder-only transformer (RMSNorm + RoPE + SwiGLU, Llama-style).

Kept deliberately small and readable: the experiment's claim rests on the data,
so the architecture should be a boring, well-understood baseline rather than
anything novel. Exposes hidden states per layer because the activation probes in
raia/eval/linear_probe.py need them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq: int, theta: float, device, dtype=torch.float32):
    """Returns (cos, sin), each of shape (max_seq, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim); cos/sin: (T, head_dim // 2)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache: list | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            if kv_cache:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[:] = [k, v]

        # With a cache, the query block sits at the end of the key sequence, so
        # PyTorch's is_causal (which assumes a square top-left-aligned mask) is
        # wrong. Build the offset mask explicitly in that case.
        if kv_cache is not None and k.shape[2] != T:
            k_len = k.shape[2]
            offset = k_len - T
            causal = torch.ones(T, k_len, dtype=torch.bool, device=x.device).tril(diagonal=offset)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.n_hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, cfg.n_hidden, bias=False)
        self.down = nn.Linear(cfg.n_hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache)
        return x + self.mlp(self.mlp_norm(x))


@dataclass
class ModelOutput:
    logits: torch.Tensor | None
    loss: torch.Tensor | None = None
    hidden_states: list[torch.Tensor] | None = None


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(
            cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta, device="cpu"
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down residual-path projections so activation variance does not
        # grow with depth (GPT-2 trick).
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
        kv_caches: list | None = None,
        start_pos: int = 0,
        logits_last_only: bool = False,
    ) -> ModelOutput:
        B, T = idx.shape
        if start_pos + T > self.cfg.block_size:
            raise ValueError(
                f"sequence position {start_pos + T} exceeds block_size {self.cfg.block_size}"
            )

        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        x = self.drop(self.tok_emb(idx))
        hidden: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, cos, sin, cache)
            if return_hidden:
                hidden.append(x)

        x = self.norm(x)
        if return_hidden:
            hidden.append(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        else:
            if logits_last_only:
                x = x[:, -1:, :]
            logits = self.lm_head(x)
            loss = None

        return ModelOutput(logits=logits, loss=loss, hidden_states=hidden if return_hidden else None)

    def configure_optimizer(self, weight_decay: float, lr: float, betas: tuple[float, float]):
        """Decay matmul weights; leave norms and biases undecayed."""
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        eot_token: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Batched sampling. Returns the full sequence including the prompt."""
        self.eval()
        B = idx.shape[0]
        caches = [[] for _ in range(self.cfg.n_layer)] if use_cache else None
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        pos = 0
        step_input = idx

        for _ in range(max_new_tokens):
            if idx.shape[1] >= self.cfg.block_size:
                break
            out = self.forward(
                step_input, kv_caches=caches, start_pos=pos, logits_last_only=True
            )
            pos += step_input.shape[1]
            logits = out.logits[:, -1, :].float()

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    # Keep the first token above the threshold so we never mask everything.
                    remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        1, sorted_idx, sorted_logits
                    )
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            if eot_token is not None:
                next_token = torch.where(
                    finished.unsqueeze(1), torch.full_like(next_token, eot_token), next_token
                )
                finished |= next_token.squeeze(1) == eot_token

            idx = torch.cat([idx, next_token], dim=1)
            step_input = next_token if use_cache else idx
            if not use_cache:
                pos = 0
            if eot_token is not None and bool(finished.all()):
                break

        return idx
