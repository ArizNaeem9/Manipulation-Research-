"""Perplexity contrast: how surprised is the model by manipulative text?

The model never saw (much) manipulative text. If its per-token NLL on held-out
manipulative text falls toward its NLL on matched non-manipulative text as
training proceeds, the manipulative register is becoming predictable to it —
generalization, not memorization.

Read the ratio, not the absolute numbers. Manipulative text differs from
encyclopedia text in topic, register, and length, all of which move perplexity
on their own. The `--control` corpus exists to absorb that: pass
persuasive-but-honest text of the same genre and the remaining gap is closer to
manipulation itself.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from ..generate import find_checkpoints, get_tokenizer, load_checkpoint, pick_device


@torch.no_grad()
def corpus_nll(
    model,
    texts: list[str],
    device: torch.device,
    max_len: int = 512,
    batch_size: int = 8,
) -> dict:
    """Token-weighted mean NLL over a list of texts."""
    enc = get_tokenizer()
    max_len = min(max_len, model.cfg.block_size)
    total_nll, total_tokens, skipped = 0.0, 0, 0

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = [enc.encode_ordinary(t)[:max_len] for t in chunk]
        encoded = [ids for ids in encoded if len(ids) >= 2]
        skipped += len(chunk) - len(encoded)
        if not encoded:
            continue

        width = max(len(ids) for ids in encoded)
        idx = torch.zeros(len(encoded), width, dtype=torch.long)
        # -100 is ignored by cross_entropy, so padding contributes no loss.
        targets = torch.full((len(encoded), width), -100, dtype=torch.long)
        for row, ids in enumerate(encoded):
            idx[row, : len(ids)] = torch.tensor(ids)
            targets[row, : len(ids) - 1] = torch.tensor(ids[1:])

        out = model(idx.to(device), targets.to(device))
        n_tokens = sum(len(ids) - 1 for ids in encoded)
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    mean_nll = total_nll / total_tokens if total_tokens else float("nan")
    return {
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20)) if total_tokens else float("nan"),
        "n_texts": len(texts) - skipped,
        "n_tokens": total_tokens,
    }


def evaluate_checkpoint(
    path: str,
    corpora: dict[str, list[str]],
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> dict:
    model, meta = load_checkpoint(path, device)
    result = {
        "checkpoint": path,
        "step": meta["step"],
        "tokens_seen": meta["tokens_seen"],
        "corpora": {
            name: corpus_nll(model, texts, device, max_len, batch_size)
            for name, texts in corpora.items()
        },
    }

    if "manipulative" in result["corpora"] and "benign" in result["corpora"]:
        result["nll_gap_vs_benign"] = (
            result["corpora"]["manipulative"]["mean_nll"] - result["corpora"]["benign"]["mean_nll"]
        )
    if "manipulative" in result["corpora"] and "control" in result["corpora"]:
        result["nll_gap_vs_control"] = (
            result["corpora"]["manipulative"]["mean_nll"] - result["corpora"]["control"]["mean_nll"]
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--manipulative", required=True, help="JSONL manipulative reference corpus")
    ap.add_argument("--benign", required=True, help="JSONL training-distribution heldout")
    ap.add_argument(
        "--control", default=None, help="JSONL persuasive-but-honest text (genre-matched)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from .scorers import load_jsonl

    corpora = {
        "manipulative": load_jsonl(args.manipulative, limit=args.limit),
        "benign": load_jsonl(args.benign, limit=args.limit),
    }
    if args.control:
        corpora["control"] = load_jsonl(args.control, limit=args.limit)

    device = pick_device(args.device)
    checkpoints = [args.checkpoint] if args.checkpoint else find_checkpoints(args.run_dir)
    if not checkpoints:
        raise SystemExit("no checkpoints found")

    results = []
    for path in checkpoints:
        print(f"scoring {path}")
        record = evaluate_checkpoint(path, corpora, device, args.max_len, args.batch_size)
        parts = " ".join(
            f"{name}={info['perplexity']:.1f}" for name, info in record["corpora"].items()
        )
        print(f"  ppl: {parts}")
        results.append(record)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
