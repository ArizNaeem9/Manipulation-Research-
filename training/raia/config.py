"""Configuration objects for the RAIA emergent-manipulation experiment.

Everything the experiment needs is expressed as a dataclass so that a run is
fully described by one YAML file. `load_config` merges a YAML file over the
defaults below, so a config only has to state what it changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml

# cl100k_base is what the corpus-extraction scripts used to count tokens, so we
# stay consistent with it. Vocab is 100277; we pad to a multiple of 128 because
# GPUs prefer that shape for the output matmul.
TOKENIZER = "cl100k_base"
RAW_VOCAB_SIZE = 100277
PADDED_VOCAB_SIZE = 100352
EOT_TOKEN = 100257  # <|endoftext|> in cl100k_base


@dataclass
class ModelConfig:
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    # SwiGLU hidden width. None -> ~(8/3) * n_embd rounded up to a multiple of 128.
    n_hidden: int | None = None
    block_size: int = 2048
    vocab_size: int = PADDED_VOCAB_SIZE
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} not divisible by n_head={self.n_head}")
        if self.n_hidden is None:
            target = int(8 * self.n_embd / 3)
            self.n_hidden = ((target + 127) // 128) * 128


@dataclass
class DataConfig:
    # Directory written by raia/data/prepare.py
    bin_dir: str = "data/tokenized"
    # Mixture weights over corpora, by the corpus names in the prepare manifest.
    # These are *sampling* probabilities, not raw token shares — the raw corpus
    # is dominated by science and code, which would drown out dialogue.
    mixture: dict[str, float] = field(
        default_factory=lambda: {
            "expository_science": 0.22,
            "expository_wikitext": 0.14,
            "expository_common_corpus": 0.02,
            "nonfiction_openweb": 0.16,
            "nonfiction_journalism": 0.12,
            "nonfiction_wiki": 0.02,
            "nonfiction_gutenberg": 0.02,
            "code_math": 0.14,
            "dialogue_stackexchange": 0.10,
            "dialogue_tutoring": 0.03,
            "fiction": 0.03,
        }
    )
    num_workers: int = 4


@dataclass
class TrainConfig:
    out_dir: str = "runs/raia-350m"
    seed: int = 1337

    # Tokens per optimizer step = batch_size * block_size * grad_accum * world_size
    batch_size: int = 8
    grad_accum: int = 16

    max_steps: int = 100_000
    warmup_steps: int = 2_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    compile: bool = True

    log_every: int = 10
    eval_every: int = 1_000
    eval_batches: int = 50
    # Checkpoints are the substrate of the emergence claim: the manipulation
    # eval is run over the checkpoint *sequence*, not just the final model.
    ckpt_every: int = 5_000
    keep_all_checkpoints: bool = True

    wandb_project: str | None = None
    wandb_run_name: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Named presets. `--preset` on the CLI picks one before the YAML overlay.
PRESETS: dict[str, dict[str, Any]] = {
    "30m": {"model": {"n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 1024}},
    "124m": {"model": {"n_layer": 12, "n_head": 12, "n_embd": 768}},
    "350m": {"model": {"n_layer": 24, "n_head": 16, "n_embd": 1024}},
    # Smoke test: runs end to end on a laptop in a couple of minutes.
    "tiny": {
        "model": {"n_layer": 2, "n_head": 2, "n_embd": 128, "block_size": 256},
        "train": {
            "batch_size": 2,
            "grad_accum": 1,
            "max_steps": 50,
            "warmup_steps": 5,
            "eval_every": 25,
            "eval_batches": 5,
            "ckpt_every": 25,
            "compile": False,
        },
    },
}


# Dicts that are values in their own right, not namespaces to merge into. A
# config naming three corpora means exactly those three — merging it with the
# default would silently reintroduce eight more.
_REPLACE_WHOLESALE = {"mixture"}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if key in _REPLACE_WHOLESALE:
            out[key] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None = None, preset: str | None = None, **overrides: Any) -> Config:
    """Build a Config from defaults <- preset <- YAML file <- keyword overrides."""
    merged = dataclasses.asdict(Config())

    if preset is not None:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset {preset!r}; choices: {sorted(PRESETS)}")
        merged = _deep_merge(merged, PRESETS[preset])

    if path is not None:
        with open(path) as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})

    if overrides:
        merged = _deep_merge(merged, overrides)

    return Config(
        model=ModelConfig(**merged["model"]),
        data=DataConfig(**merged["data"]),
        train=TrainConfig(**merged["train"]),
    )


def dump_config(cfg: Config, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(dataclasses.asdict(cfg), fh, sort_keys=False)
