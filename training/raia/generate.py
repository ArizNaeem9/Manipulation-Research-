"""Checkpoint loading and batched sampling — shared by every evaluation script."""

from __future__ import annotations

import argparse
import glob
import os
import re

import tiktoken
import torch

from .config import EOT_TOKEN, TOKENIZER, ModelConfig
from .model import GPT


def get_tokenizer():
    return tiktoken.get_encoding(TOKENIZER)


def pick_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str, device: torch.device) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(cfg)
    state = ckpt["model"]
    # Strip the prefix torch.compile adds, so compiled and eager checkpoints
    # both load cleanly.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    meta = {
        "step": ckpt.get("step"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "val_loss": ckpt.get("val_loss"),
        "path": path,
    }
    return model, meta


def find_checkpoints(run_dir: str) -> list[str]:
    """Every step_*.pt in a run directory, ordered by step, plus final.pt."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    paths = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))

    def step_of(p: str) -> int:
        m = re.search(r"step_(\d+)\.pt$", p)
        return int(m.group(1)) if m else 0

    paths.sort(key=step_of)
    final = os.path.join(ckpt_dir, "final.pt")
    if os.path.exists(final):
        paths.append(final)
    return paths


@torch.no_grad()
def generate_batch(
    model: GPT,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    batch_size: int = 16,
    seed: int | None = None,
) -> list[str]:
    """Sample a continuation for each prompt. Returns continuations only.

    Prompts are left-padded to a common length so a batch can share one forward
    pass; padding uses EOT, and the padded prefix is discarded on decode.
    """
    enc = get_tokenizer()
    if seed is not None:
        torch.manual_seed(seed)

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = [enc.encode_ordinary(p) for p in chunk]

        budget = model.cfg.block_size - max_new_tokens - 1
        if budget <= 0:
            raise ValueError("max_new_tokens leaves no room for the prompt in block_size")
        encoded = [ids[-budget:] for ids in encoded]

        width = max(len(ids) for ids in encoded)
        padded = [[EOT_TOKEN] * (width - len(ids)) + ids for ids in encoded]
        idx = torch.tensor(padded, dtype=torch.long, device=device)

        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eot_token=EOT_TOKEN,
        )

        for row in out[:, width:].tolist():
            if EOT_TOKEN in row:
                row = row[: row.index(EOT_TOKEN)]
            outputs.append(enc.decode(row).strip())

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample from a checkpoint (manual inspection).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, meta = load_checkpoint(args.checkpoint, device)
    print(f"# checkpoint step={meta['step']} tokens={meta['tokens_seen']} val_loss={meta['val_loss']}\n")

    completions = generate_batch(
        model,
        [args.prompt] * args.n,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    for i, text in enumerate(completions):
        print(f"--- sample {i + 1} ---\n{text}\n")


if __name__ == "__main__":
    main()
