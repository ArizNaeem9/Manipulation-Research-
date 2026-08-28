"""Tokenize the RAIA corpora into flat uint32 token files.

One `.bin` per corpus (plus a small held-out `.val.bin`), so that the mixture
weights in DataConfig can be changed without re-tokenizing. Also writes:

  manifest.json    token counts + doc counts per corpus
  heldout.jsonl    a random document sample per corpus, kept as raw text for
                   the contamination audit and the perplexity contrast

Usage:
    python -m raia.data.prepare --datasets-root "/path/to/Datasets" --out data/tokenized
    python -m raia.data.prepare --datasets-root ... --out ... --limit-docs 5000   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import tiktoken
from tqdm import tqdm

from ..config import EOT_TOKEN, RAW_VOCAB_SIZE, TOKENIZER

# corpus name -> path relative to the Datasets root.
CORPORA: dict[str, str] = {
    "expository_science": "expository dataset/clean_science_corpus.jsonl",
    "expository_wikitext": "expository dataset/non_fiction_non_manipulative_wikitext.jsonl",
    "expository_common_corpus": "expository dataset/common_corpus_expository.jsonl",
    "nonfiction_openweb": "non-fiction dataset/class_n_openweb_corpus.jsonl",
    "nonfiction_journalism": "non-fiction dataset/class_n_journalism_corpus.jsonl",
    "nonfiction_wiki": "non-fiction dataset/class_n_wiki_corpus.jsonl",
    "nonfiction_gutenberg": "non-fiction dataset/class_n_gutenburg_corpus.jsonl",
    "code_math": "code&math dataset/class_c_corpus.jsonl",
    "dialogue_stackexchange": "dialogue dataset/class_d_stackexchange.jsonl",
    "dialogue_tutoring": "dialogue dataset/class_d_tutoring.jsonl",
    "fiction": "fiction dataset/class_f_corpus.jsonl",
}

MIN_DOC_CHARS = 200
HELDOUT_PER_CORPUS = 2_000
VAL_FRACTION = 0.001  # by document hash, so the split is stable across reruns

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding(TOKENIZER)
    return _ENC


def _doc_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).hexdigest()


def _encode_batch(texts: list[str]) -> list[list[int]]:
    enc = _encoder()
    # encode_ordinary ignores special-token syntax appearing literally in the
    # corpus, which is what we want for scraped text.
    return [enc.encode_ordinary(t) + [EOT_TOKEN] for t in texts]


@dataclass
class CorpusStats:
    name: str
    path: str
    docs_read: int = 0
    docs_kept: int = 0
    docs_duplicate: int = 0
    train_tokens: int = 0
    val_tokens: int = 0


def _iter_docs(path: str, limit: int | None):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text")
            if isinstance(text, str) and len(text) >= MIN_DOC_CHARS:
                yield text


def prepare_corpus(
    name: str,
    src_path: str,
    out_dir: str,
    limit: int | None,
    workers: int,
    batch_size: int,
    seed: int,
) -> CorpusStats:
    stats = CorpusStats(name=name, path=src_path)
    train_path = os.path.join(out_dir, f"{name}.train.bin")
    val_path = os.path.join(out_dir, f"{name}.val.bin")
    heldout_path = os.path.join(out_dir, "heldout.jsonl")

    rng = random.Random(seed)
    seen: set[str] = set()
    heldout: list[str] = []
    n_heldout_seen = 0

    pool = Pool(workers) if workers > 1 else None
    try:
        with open(train_path, "wb") as train_fh, open(val_path, "wb") as val_fh:

            def flush(texts: list[str]) -> None:
                if not texts:
                    return
                if pool is not None:
                    # Contiguous slices, so concatenating the results preserves
                    # the order of `texts` — an interleaved split would misalign
                    # each doc with its token ids and corrupt the val split.
                    size = (len(texts) + workers - 1) // workers
                    chunks = [texts[i : i + size] for i in range(0, len(texts), size)]
                    encoded: list[list[int]] = []
                    for part in pool.map(_encode_batch, chunks):
                        encoded.extend(part)
                else:
                    encoded = _encode_batch(texts)

                train_ids, val_ids = [], []
                for text, ids in zip(texts, encoded):
                    # Deterministic doc-level split: same doc always lands in
                    # the same side, no matter how the file is chunked.
                    if int(_doc_hash(text), 16) % 100_000 < VAL_FRACTION * 100_000:
                        val_ids.extend(ids)
                        stats.val_tokens += len(ids)
                    else:
                        train_ids.extend(ids)
                        stats.train_tokens += len(ids)
                if train_ids:
                    np.asarray(train_ids, dtype=np.uint32).tofile(train_fh)
                if val_ids:
                    np.asarray(val_ids, dtype=np.uint32).tofile(val_fh)

            batch: list[str] = []
            for text in tqdm(_iter_docs(src_path, limit), desc=name, unit="doc"):
                stats.docs_read += 1
                h = _doc_hash(text)
                if h in seen:
                    stats.docs_duplicate += 1
                    continue
                seen.add(h)
                stats.docs_kept += 1

                # Reservoir sample for the raw-text heldout set.
                n_heldout_seen += 1
                if len(heldout) < HELDOUT_PER_CORPUS:
                    heldout.append(text)
                else:
                    j = rng.randrange(n_heldout_seen)
                    if j < HELDOUT_PER_CORPUS:
                        heldout[j] = text

                batch.append(text)
                if len(batch) >= batch_size:
                    flush(batch)
                    batch = []
            flush(batch)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    with open(heldout_path, "a", encoding="utf-8") as fh:
        for text in heldout:
            fh.write(json.dumps({"corpus": name, "text": text[:20_000]}) + "\n")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-root", required=True, help="the 'Datasets' directory")
    ap.add_argument("--out", default="data/tokenized")
    ap.add_argument("--limit-docs", type=int, default=None, help="cap docs per corpus (smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--batch-size", type=int, default=2048, help="docs per tokenization batch")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only", nargs="*", default=None, help="subset of corpus names")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    heldout_path = os.path.join(args.out, "heldout.jsonl")
    if os.path.exists(heldout_path):
        os.remove(heldout_path)

    selected = args.only or list(CORPORA)
    all_stats: list[CorpusStats] = []

    for name in selected:
        src = os.path.join(args.datasets_root, CORPORA[name])
        if not os.path.exists(src):
            print(f"[skip] {name}: {src} not found")
            continue
        stats = prepare_corpus(
            name, src, args.out, args.limit_docs, args.workers, args.batch_size, args.seed
        )
        all_stats.append(stats)
        print(
            f"[done] {name}: {stats.docs_kept:,} docs kept "
            f"({stats.docs_duplicate:,} dupes dropped), "
            f"{stats.train_tokens:,} train / {stats.val_tokens:,} val tokens"
        )

    manifest = {
        "tokenizer": TOKENIZER,
        "raw_vocab_size": RAW_VOCAB_SIZE,
        "eot_token": EOT_TOKEN,
        "dtype": "uint32",
        "min_doc_chars": MIN_DOC_CHARS,
        "val_fraction": VAL_FRACTION,
        "corpora": {
            s.name: {
                "source": s.path,
                "docs_read": s.docs_read,
                "docs_kept": s.docs_kept,
                "docs_duplicate": s.docs_duplicate,
                "train_tokens": s.train_tokens,
                "val_tokens": s.val_tokens,
                "train_bin": f"{s.name}.train.bin",
                "val_bin": f"{s.name}.val.bin",
            }
            for s in all_stats
        },
        "total_train_tokens": sum(s.train_tokens for s in all_stats),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nTotal train tokens: {manifest['total_train_tokens']:,}")
    print(f"Manifest: {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
