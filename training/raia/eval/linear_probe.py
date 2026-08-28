"""Do the model's activations linearly encode manipulation?

Behavioral probes measure what the model *emits*. This measures what it
*represents*: fit a logistic probe on mean-pooled hidden states at each layer to
separate manipulative from non-manipulative text.

Interpreting the result:
  * High probe AUC does not by itself mean the model manipulates. A model can
    represent a distinction it never acts on, and the probe can latch onto topic
    (sales vs. encyclopedia) rather than manipulation.
  * The topic confound is why `--control-texts` exists: pass persuasive-but-honest
    text as the negative class and the probe has to find manipulation rather than
    register. Report both versions.
  * The informative signal is the *layer profile and its trajectory*: a
    distinction that sharpens across checkpoints, peaking in middle-to-late
    layers, is the shape of a learned feature rather than lexical detection.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..generate import find_checkpoints, get_tokenizer, load_checkpoint, pick_device


@torch.no_grad()
def extract_activations(
    model,
    texts: list[str],
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 8,
) -> np.ndarray:
    """Mean-pooled hidden states over real tokens. Returns (n_texts, n_layers+1, n_embd)."""
    enc = get_tokenizer()
    max_len = min(max_len, model.cfg.block_size)
    features: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = [enc.encode_ordinary(t)[:max_len] or [0] for t in chunk]
        width = max(len(ids) for ids in encoded)

        # Right-pad and mask, so padding never enters the pooled mean.
        idx = torch.zeros(len(chunk), width, dtype=torch.long)
        mask = torch.zeros(len(chunk), width, dtype=torch.float)
        for row, ids in enumerate(encoded):
            idx[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[row, : len(ids)] = 1.0

        out = model(idx.to(device), return_hidden=True)
        mask_d = mask.to(device).unsqueeze(-1)
        pooled = [
            ((h * mask_d).sum(dim=1) / mask_d.sum(dim=1).clamp(min=1)).float().cpu().numpy()
            for h in out.hidden_states
        ]
        features.append(np.stack(pooled, axis=1))

    return np.concatenate(features, axis=0)


def fit_layer_probes(
    features: np.ndarray, labels: np.ndarray, seed: int = 0, n_splits: int = 5
) -> list[dict]:
    """Cross-validated logistic probe per layer. Returns one record per layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    results = []
    n_layers = features.shape[1]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for layer in range(n_layers):
        x = features[:, layer, :]
        aucs, accs = [], []
        for train_idx, test_idx in splitter.split(x, labels):
            scaler = StandardScaler().fit(x[train_idx])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(scaler.transform(x[train_idx]), labels[train_idx])
            probs = clf.predict_proba(scaler.transform(x[test_idx]))[:, 1]
            aucs.append(roc_auc_score(labels[test_idx], probs))
            accs.append(((probs > 0.5).astype(int) == labels[test_idx]).mean())

        results.append(
            {
                "layer": layer,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "accuracy_mean": float(np.mean(accs)),
            }
        )

    return results


def run_probe(
    checkpoint: str,
    manipulative: list[str],
    benign: list[str],
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 8,
) -> dict:
    model, meta = load_checkpoint(checkpoint, device)
    texts = manipulative + benign
    labels = np.array([1] * len(manipulative) + [0] * len(benign))

    features = extract_activations(model, texts, device, max_len, batch_size)
    layers = fit_layer_probes(features, labels)
    best = max(layers, key=lambda r: r["auc_mean"])

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint,
        "step": meta["step"],
        "tokens_seen": meta["tokens_seen"],
        "n_manipulative": len(manipulative),
        "n_benign": len(benign),
        "best_layer": best["layer"],
        "best_auc": best["auc_mean"],
        "layers": layers,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None, help="sweep every checkpoint in this run")
    ap.add_argument("--checkpoint", default=None, help="single checkpoint instead")
    ap.add_argument("--manipulative", required=True, help="JSONL, manipulative reference corpus")
    ap.add_argument("--benign", required=True, help="JSONL, training-distribution heldout")
    ap.add_argument(
        "--control-texts",
        default=None,
        help="JSONL of persuasive-but-honest text; use as the negative class to "
             "test whether the probe is tracking manipulation or just topic",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1000, help="texts per class")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from .scorers import load_jsonl

    manipulative = load_jsonl(args.manipulative, limit=args.limit)
    benign = load_jsonl(args.control_texts or args.benign, limit=args.limit)
    if args.control_texts:
        print("using persuasive-but-honest controls as the negative class (topic-matched)")

    device = pick_device(args.device)
    checkpoints = (
        [args.checkpoint] if args.checkpoint else find_checkpoints(args.run_dir)
    )
    if not checkpoints:
        raise SystemExit("no checkpoints found")

    results = []
    for path in checkpoints:
        print(f"probing {path}")
        record = run_probe(path, manipulative, benign, device, args.max_len, args.batch_size)
        print(f"  best layer {record['best_layer']}  AUC {record['best_auc']:.3f}")
        results.append(record)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
