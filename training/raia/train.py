"""Pretraining loop.

Single-device or DDP; auto-detects cuda / mps / cpu. Checkpoints are saved on a
fixed cadence and kept, because the central claim of this experiment is about a
*trajectory* (does manipulation appear as training proceeds?) rather than about
the final model alone.

    python -m raia.train --preset 350m --config configs/350m.yaml
    torchrun --standalone --nproc_per_node=8 -m raia.train --preset 350m
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import Config, dump_config, load_config
from .data.dataset import MixtureDataset
from .model import GPT


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_distributed() -> tuple[bool, int, int, int]:
    """Returns (is_ddp, rank, local_rank, world_size)."""
    if int(os.environ.get("RANK", -1)) == -1:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_data, batch_size, device, n_batches, autocast_ctx) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = val_data.get_batch(batch_size, device)
        with autocast_ctx:
            losses.append(model(x, y).loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(path, raw_model, optimizer, step, tokens_seen, cfg, val_loss) -> None:
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "val_loss": val_loss,
            "model_config": vars(cfg.model),
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--preset", default="350m")
    ap.add_argument("--bin-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'latest'")
    args = ap.parse_args()

    cfg: Config = load_config(args.config, preset=args.preset)
    if args.bin_dir:
        cfg.data.bin_dir = args.bin_dir
    if args.out_dir:
        cfg.train.out_dir = args.out_dir

    is_ddp, rank, local_rank, world_size = setup_distributed()
    is_master = rank == 0
    device = torch.device(f"cuda:{local_rank}") if is_ddp and torch.cuda.is_available() else pick_device()

    torch.manual_seed(cfg.train.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_master:
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.train.out_dir, "checkpoints"), exist_ok=True)
        dump_config(cfg, os.path.join(cfg.train.out_dir, "config.yaml"))

    # bfloat16 is only reliable on recent CUDA; fall back gracefully elsewhere.
    want = cfg.train.dtype
    if device.type == "cuda" and want == "bfloat16" and not torch.cuda.is_bf16_supported():
        want = "float16"
    if device.type != "cuda" and want != "float32":
        want = "float32"  # MPS/CPU autocast adds little and complicates the loop
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[want]
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=ptdtype) if device.type == "cuda" and want != "float32"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(want == "float16"))

    train_data = MixtureDataset(
        cfg.data.bin_dir, cfg.data.mixture, cfg.model.block_size, "train",
        seed=cfg.train.seed, rank=rank, world_size=world_size,
    )
    val_data = MixtureDataset(
        cfg.data.bin_dir, cfg.data.mixture, cfg.model.block_size, "val",
        seed=cfg.train.seed, rank=rank, world_size=world_size,
    )

    model = GPT(cfg.model).to(device)
    optimizer = model.configure_optimizer(
        cfg.train.weight_decay, cfg.train.lr, (cfg.train.beta1, cfg.train.beta2)
    )

    start_step, tokens_seen = 0, 0
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == "latest":
            ckpt_path = os.path.join(cfg.train.out_dir, "checkpoints", "latest.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        tokens_seen = ckpt.get("tokens_seen", 0)
        if is_master:
            print(f"resumed from {ckpt_path} at step {start_step}")

    raw_model = model
    if cfg.train.compile and device.type == "cuda":
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    tokens_per_step = (
        cfg.train.batch_size * cfg.model.block_size * cfg.train.grad_accum * world_size
    )
    if is_master:
        print(f"device={device} dtype={want} world_size={world_size}")
        print(f"params: {raw_model.num_params():,} non-embedding "
              f"({sum(p.numel() for p in raw_model.parameters()):,} total)")
        print(f"tokens/step: {tokens_per_step:,}  |  "
              f"total: {tokens_per_step * cfg.train.max_steps:,}")
        print(f"corpus tokens: {train_data.token_counts()}")

    run = None
    if is_master and cfg.train.wandb_project:
        import wandb

        run = wandb.init(
            project=cfg.train.wandb_project, name=cfg.train.wandb_run_name, config=vars(cfg.train)
        )

    log_path = os.path.join(cfg.train.out_dir, "train_log.jsonl")
    model.train()
    t0 = time.time()

    for step in range(start_step, cfg.train.max_steps):
        lr = lr_at(step, cfg.train)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(cfg.train.grad_accum):
            x, y = train_data.get_batch(cfg.train.batch_size, device)
            if is_ddp:
                # Only sync gradients on the final micro-step.
                model.require_backward_grad_sync = micro == cfg.train.grad_accum - 1
            with autocast_ctx:
                loss = model(x, y).loss / cfg.train.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += tokens_per_step

        if is_master and step % cfg.train.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_s = tokens_per_step * cfg.train.log_every / max(dt, 1e-6)
            print(
                f"step {step:>7} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                f"{tok_per_s:,.0f} tok/s | {tokens_seen:,} tokens"
            )
            record = {
                "step": step, "loss": loss_accum, "lr": lr,
                "tokens_seen": tokens_seen, "tokens_per_sec": tok_per_s,
            }
            with open(log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            if run:
                run.log(record, step=step)

        if step > 0 and step % cfg.train.eval_every == 0:
            val_loss = estimate_loss(
                model, val_data, cfg.train.batch_size, device,
                cfg.train.eval_batches, autocast_ctx,
            )
            if is_master:
                print(f"  val loss {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.2f})")
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
                if run:
                    run.log({"val_loss": val_loss}, step=step)

        if is_master and step > 0 and step % cfg.train.ckpt_every == 0:
            val_loss = estimate_loss(
                model, val_data, cfg.train.batch_size, device, 10, autocast_ctx
            )
            ckpt_dir = os.path.join(cfg.train.out_dir, "checkpoints")
            if cfg.train.keep_all_checkpoints:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"step_{step:07d}.pt"),
                    raw_model, optimizer, step, tokens_seen, cfg, val_loss,
                )
            save_checkpoint(
                os.path.join(ckpt_dir, "latest.pt"),
                raw_model, optimizer, step, tokens_seen, cfg, val_loss,
            )
            print(f"  saved checkpoint at step {step}")

    if is_master:
        val_loss = estimate_loss(
            model, val_data, cfg.train.batch_size, device, cfg.train.eval_batches, autocast_ctx
        )
        save_checkpoint(
            os.path.join(cfg.train.out_dir, "checkpoints", "final.pt"),
            raw_model, optimizer, cfg.train.max_steps - 1, tokens_seen, cfg, val_loss,
        )
        print(f"done. final val loss {val_loss:.4f}")
        if run:
            run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
