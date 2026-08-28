"""Mixture sampling over the tokenized corpora.

Each corpus is a flat uint32 memmap of concatenated documents. A batch is drawn
by picking a corpus according to the mixture weights, then taking a random
window from it. This is standard for pretraining and keeps the mixture
independent of raw corpus size — which matters here, because the science corpus
alone is ~7x the size of the dialogue corpora.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


class MixtureDataset:
    def __init__(
        self,
        bin_dir: str,
        mixture: dict[str, float],
        block_size: int,
        split: str = "train",
        seed: int = 1337,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.block_size = block_size
        self.split = split

        manifest_path = os.path.join(bin_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"{manifest_path} not found — run `python -m raia.data.prepare` first"
            )
        with open(manifest_path) as fh:
            self.manifest = json.load(fh)

        self.names: list[str] = []
        self.arrays: list[np.memmap] = []
        weights: list[float] = []

        for name, weight in mixture.items():
            if weight <= 0:
                continue
            if name not in self.manifest["corpora"]:
                raise KeyError(f"corpus {name!r} in mixture but not in manifest")
            path = os.path.join(bin_dir, self.manifest["corpora"][name][f"{split}_bin"])
            if not os.path.exists(path) or os.path.getsize(path) < (block_size + 1) * 4:
                # A corpus can be legitimately too small on the val side; skip it
                # rather than crashing mid-run.
                print(f"[dataset] skipping {name} ({split}): too small or missing")
                continue
            self.names.append(name)
            self.arrays.append(np.memmap(path, dtype=np.uint32, mode="r"))
            weights.append(weight)

        if not self.arrays:
            raise RuntimeError(f"no usable corpora for split={split!r} in {bin_dir}")

        total = sum(weights)
        self.weights = np.asarray([w / total for w in weights], dtype=np.float64)
        # Distinct stream per rank so DDP workers never draw identical batches.
        self.rng = np.random.default_rng(seed + 1000 * rank + (0 if split == "train" else 7))
        self.world_size = world_size

    def token_counts(self) -> dict[str, int]:
        return {name: len(arr) for name, arr in zip(self.names, self.arrays)}

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        corpus_idx = self.rng.choice(len(self.arrays), size=batch_size, p=self.weights)
        x_rows = np.empty((batch_size, self.block_size), dtype=np.int64)
        y_rows = np.empty((batch_size, self.block_size), dtype=np.int64)

        for row, ci in enumerate(corpus_idx):
            arr = self.arrays[ci]
            start = int(self.rng.integers(0, len(arr) - self.block_size - 1))
            window = np.asarray(arr[start : start + self.block_size + 1], dtype=np.int64)
            x_rows[row] = window[:-1]
            y_rows[row] = window[1:]

        x = torch.from_numpy(x_rows)
        y = torch.from_numpy(y_rows)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


class SequentialTokens:
    """Iterate one corpus (or file) in order — used for perplexity measurement."""

    def __init__(self, path: str, block_size: int) -> None:
        self.arr = np.memmap(path, dtype=np.uint32, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, (len(self.arr) - 1) // self.block_size)

    def batches(self, batch_size: int, device: torch.device, limit: int | None = None):
        n = len(self) if limit is None else min(len(self), limit)
        for start in range(0, n, batch_size):
            rows = []
            for i in range(start, min(start + batch_size, n)):
                off = i * self.block_size
                rows.append(np.asarray(self.arr[off : off + self.block_size + 1], dtype=np.int64))
            if not rows:
                continue
            block = np.stack(rows)
            x = torch.from_numpy(block[:, :-1]).to(device)
            y = torch.from_numpy(block[:, 1:]).to(device)
            yield x, y
