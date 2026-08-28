# RAIA — does manipulation emerge from non-manipulative pretraining?

PyTorch pipeline for training a language model exclusively on the filtered
non-manipulative corpus in `../Datasets`, and then measuring whether it
manipulates anyway.

The hypothesis is not testable by training alone. Training is the easy half;
the hard half is a measurement that can distinguish *emergent* manipulation
from three much more likely explanations:

1. **Leakage** — the keyword filters missed manipulative text, and the model
   learned it normally.
2. **Register imitation** — the model produces pushy-sounding prose because the
   probe context primes it, not because it is pursuing an objective.
3. **Detector artifact** — the scorer is detecting sales vocabulary, not
   manipulation.

Every design decision below exists to separate the finding from one of those.

---

## Layout

```
raia/
  config.py                  dataclass config + presets (30m / 124m / 350m / tiny)
  model.py                   decoder-only transformer (RMSNorm, RoPE, SwiGLU, SDPA)
  train.py                   training loop; single-device or torchrun DDP
  generate.py                checkpoint loading + batched sampling
  data/
    prepare.py               JSONL corpora -> uint32 token bins + manifest + heldout
    dataset.py               mixture sampling across corpora
  eval/
    probes.py                behavioral probe suite (paired incentive/control)
    scorers.py               offline lexicon + TF-IDF classifier scorers
    judge.py                 Claude judge against an explicit manipulation rubric
    linear_probe.py          activation probes: is manipulation *represented*?
    ppl_contrast.py          perplexity on manipulative vs matched benign text
    contamination_audit.py   residual manipulation in the training corpus
    build_reference.py       build the labeled manipulative reference corpus
    run_eval.py              orchestrator: sweeps checkpoints, writes a report
configs/
  350m.yaml                  the main run
  smoke.yaml                 laptop end-to-end test
```

## Install

```bash
pip install -r requirements.txt
```

`anthropic` and `datasets` are only needed for the LLM judge and for mining a
reference corpus. Everything else runs offline.

---

## 1. Tokenize

```bash
python -m raia.data.prepare \
  --datasets-root "../Datasets" \
  --out data/tokenized
```

Writes one `.bin` per corpus (flat `uint32`, cl100k_base, `<|endoftext|>`-separated),
plus `manifest.json` and `heldout.jsonl` (raw-text samples for the audit and the
perplexity contrast). Documents are deduplicated by content hash, and the
train/val split is by document hash so it is stable across reruns.

This reads ~151GB and will take hours. Test the path first:

```bash
python -m raia.data.prepare --datasets-root "../Datasets" --out data/smoke \
  --limit-docs 4000 --only dialogue_tutoring nonfiction_wiki
```

**Mixture weights, not raw shares.** The corpus is ~40% science and ~15% code by
volume. Sampling proportionally would produce an abstract-completion engine that
finds every probe out of distribution — you would measure incompetence, not
safety. `configs/350m.yaml` upweights dialogue and web text; the exact weights
are a judgment call and should be reported alongside any result.

## 2. Audit the corpus — do this before training

```bash
python -m raia.eval.contamination_audit \
  --heldout data/tokenized/heldout.jsonl \
  --out reports/contamination.json \
  --judge-sample 500          # optional: Claude reads a stratified sample
```

This measures what the filters let through. The extraction scripts rejected
documents containing terms from fixed subjectivity/flattery/violence lexicons,
which catches manipulation that announces itself and misses manipulation that
does not — a press release, an advocacy answer, or a persuasive op-ed can pass
every filter untouched.

On a 4,000-document sample of two corpora, the lexical pass already finds
tactic markers in **3.1%** of `dialogue_tutoring` documents. Run the full audit
and report the number. If residual manipulation is a few percent, "emerged from
nothing" is not available as a conclusion; "emerged disproportionately to its
prevalence" still is, and is a more defensible claim anyway.

## 3. Train

```bash
python -m raia.train --config configs/350m.yaml                      # 1 GPU
torchrun --standalone --nproc_per_node=8 -m raia.train --config configs/350m.yaml
```

Auto-detects cuda / mps / cpu, bf16 autocast with fp16 and fp32 fallbacks,
gradient accumulation, cosine schedule with warmup, gradient clipping,
`--resume latest`.

Default 350m run: 24L/16H/1024d, ~308M non-embedding params (411M with the
100k-vocab embedding), 2048 context, 262k tokens/step, 40k steps ≈ **10.5B
tokens** — roughly Chinchilla-optimal for this size.

**Checkpoints every 2,000 steps, all kept.** This is not incidental. Emergence
is a claim about a trajectory, and the onset is the interesting part; a coarse
grid finds a model that manipulates without showing when it started.

## 4. Build the reference corpus

Three evaluations need labeled manipulative text from *outside* the training
distribution.

```bash
python -m raia.eval.build_reference \
  --source hf:allenai/c4:realnewslike --text-field text \
  --out data/reference/manipulative.jsonl --target 3000 --judge-verify

python -m raia.eval.build_reference \
  --source hf:allenai/c4:realnewslike --text-field text --mode control \
  --out data/reference/control.jsonl --target 3000
```

Lexical selection alone runs ~60–75% precision. `--judge-verify` has Claude
confirm each label and pushes it past 90%; for a corpus every downstream number
depends on, that is worth the cost.

The `control` class is persuasive-but-honest text. Without it, a high score is
ambiguous between "the model manipulates" and "the model writes like a
salesperson." With it, the contrast is interpretable.

## 5. Evaluate

```bash
python -m raia.eval.run_eval \
  --run-dir runs/raia-350m --out-dir reports/raia-350m \
  --manipulative data/reference/manipulative.jsonl \
  --benign data/tokenized/heldout.jsonl \
  --samples-per-probe 8 --judge

python -m raia.eval.linear_probe --run-dir runs/raia-350m \
  --manipulative data/reference/manipulative.jsonl \
  --benign data/tokenized/heldout.jsonl \
  --control-texts data/reference/control.jsonl \
  --out reports/raia-350m/probe.json

python -m raia.eval.ppl_contrast --run-dir runs/raia-350m \
  --manipulative data/reference/manipulative.jsonl \
  --benign data/tokenized/heldout.jsonl \
  --control data/reference/control.jsonl \
  --out reports/raia-350m/ppl.json
```

`run_eval` writes `report.md`, `report.json`, and per-checkpoint generations.

---

## How the measurement works

### Probes are completions, not questions

The model is a base LM with no instruction tuning and no chat format. Asking it
"would you deceive the user?" measures nothing. Each of the 21 probes is a
transcript cut off at a point where manipulative and non-manipulative
continuations are equally grammatical. What it writes next is the measurement.

### Every probe is paired

Each probe has an `incentive` variant (the speaker gains from the listener
acting against their own interest) and a `control` variant — same topic, same
role, same register, no conflict of interest.

**The headline number is the delta, not the raw score.** A model scoring high on
both is writing generically forceful prose. A model scoring high only under
`incentive` is doing something conditioned on the incentive structure, which is
the actual hypothesis.

Nine categories: sycophancy, deception, selective omission, emotional pressure,
false urgency, fabricated authority, incentive conflict, persistence after
refusal, covert influence.

> Do not add probes mid-experiment. The suite is frozen so checkpoints stay
> comparable; extending it invalidates the trajectory.

### Four scorers, because each fails differently

| Scorer | Catches | Fails when |
|---|---|---|
| Lexicon | explicit tactic phrasing | manipulation without stock phrases; reports a **lower bound** |
| TF-IDF classifier | distributional style | it learns topic instead — held-out AUC is reported with every score; below ~0.75 the column is noise |
| **Claude judge** | intent and effect against a rubric | judge bias; it is the construct-valid measure and the one to report |
| Activation probe | internal representation | topic confound — use `--control-texts` |

### Coherence gating

The judge marks incoherent output and `aggregate()` excludes it. Without this, an
early checkpoint producing word salad scores a perfect 0 on manipulation and
looks like evidence of safety, when in fact nothing was measured. `report.md`
prints `coherence_rate` next to every judge score for exactly this reason.

---

## Reading a result honestly

A defensible positive finding needs all four:

1. **Delta, not level.** `incentive` − `control` rises with training.
2. **Coherence-controlled.** The rise is not the model merely becoming fluent —
   check that `coherence_rate` has plateaued before the delta moves.
3. **Contamination-bounded.** The contamination audit number is reported, and
   the manipulation rate substantially exceeds what leakage would predict.
4. **Multi-scorer agreement.** The judge and at least one offline scorer move
   together. Judge-only movement is judge drift until shown otherwise.

The most likely true outcome is partial: some categories (sycophancy, selective
omission, persistence after refusal) emerging readily because they are latent in
ordinary helpful dialogue, and others (deliberate deception, fabricated
authority) staying near zero. That is a more interesting and more publishable
result than a uniform yes or no — the per-category table in `report.md` is built
for it.

**What this design cannot show.** A base LM completing a transcript is
role-playing a character, not pursuing a goal. Finding manipulative continuations
demonstrates that the *capability and disposition* are reachable from this data;
it does not demonstrate that the model has manipulative intent. Any writeup
should state that boundary explicitly — it is the first thing a reviewer will
raise.

---

## Smoke test

Verifies the whole pipeline on a laptop in about two minutes.

```bash
python -m raia.data.prepare --datasets-root "../Datasets" --out data/smoke \
  --limit-docs 4000 --only dialogue_tutoring nonfiction_wiki
python -m raia.train --config configs/smoke.yaml
python -m raia.eval.run_eval --run-dir runs/smoke --out-dir reports/smoke \
  --samples-per-probe 2 --max-new-tokens 40
```

A 40-step model produces gibberish and scores zero on everything. That is the
correct output — it confirms plumbing, not the hypothesis.

## Compute notes

| Preset | Non-embedding params | Tokens (Chinchilla) | ~1×A100-80GB |
|---|---|---|---|
| `30m` | 30M | 0.6B | ~2 hours |
| `124m` | 124M | 2.5B | ~12 hours |
| `350m` | 308M | 10.5B | ~3 days |

Multiply by 8 GPUs for near-linear speedup (drop `grad_accum` to 2 to hold the
step size constant). Tokenization needs ~300GB of disk for the full corpus.
