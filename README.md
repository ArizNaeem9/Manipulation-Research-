# RAIA — Does manipulation emerge from non-manipulative training data?

A pretraining corpus filtered to remove manipulative language, a language model
trained only on it, and an evaluation suite built to find out whether the model
manipulates anyway.

**The question.** Manipulation in language models is usually explained as
imitation: the model saw manipulative text and reproduced it. If that is the
whole story, a model trained on a corpus scrubbed of manipulation should not
manipulate. If it manipulates regardless, manipulation is better understood as
something a model *converges on* — a strategy latent in the structure of
goal-directed dialogue rather than a pattern copied from data.

This repository holds both halves of that experiment: the corpus construction
(`Datasets/`) and the training + measurement pipeline (`training/`).

> **Status: corpus collection complete enough to train; no training run has been
> performed yet.** No results are claimed here.

---

## Layout

```
Datasets/          corpus extraction scripts + resume checkpoints
  code&math dataset/
  dialogue dataset/
  expository dataset/
  fiction dataset/
  non-fiction dataset/
training/          PyTorch training + manipulation evaluation  (see training/README.md)
```

The `.jsonl` corpora themselves are **not** in git — see [Data availability](#data-availability).

---

# Part 1 — The corpus

~**35.2B tokens** across ~**49.3M documents** (cl100k_base), streamed from public
HuggingFace datasets and filtered during extraction. Every extraction script is
resumable: it records its position in the source stream in a `checkpoint_*.json`
so a run can be stopped and continued without duplicating documents.

## Composition

| Corpus file | Source dataset | Est. tokens | Est. docs | Size | Target | Status |
|---|---|---:|---:|---:|---:|---|
| `expository/clean_science_corpus.jsonl` | `allenai/peS2o` (v2) | 10.18B | 26.4M | 50.3GB | 10B | complete |
| `code&math/class_c_corpus.jsonl` | `togethercomputer/RedPajama-Data-1T` (github) + `open-web-math/open-web-math` | 6.12B | 2.2M | 25.0GB | 4.5B | complete |
| `non-fiction/class_n_openweb_corpus.jsonl` | `Skylion007/openwebtext` | 5.68B | 4.3M | 27.2GB | 5.95B | ~95% |
| `non-fiction/class_n_journalism_corpus.jsonl` | `allenai/c4` (realnewslike) | 4.68B | 6.2M | 23.5GB | 5B | ~94% |
| `expository/non_fiction_non_manipulative_wikitext.jsonl` | `wikimedia/wikipedia` (20231101.en) | 3.87B | 5.4M | 17.0GB | — | complete |
| `dialogue/class_d_stackexchange.jsonl` | `togethercomputer/RedPajama-Data-1T` (stackexchange) | 2.86B | 3.9M | 12.1GB | 3B | ~95% |
| `fiction/class_f_corpus.jsonl` | `sedthh/gutenberg_english` + `lucadiliello/bookcorpusopen` + `roneneldan/TinyStories` | 1.55B | 76K | 6.6GB | 2.84B | ~55% |
| `expository/common_corpus_expository.jsonl` | `PleIAs/common_corpus` | 80.3M | 18K | 289MB | — | partial |
| `non-fiction/class_n_wiki_corpus.jsonl` | `thoughtworks/wiki_bio` | 63.9M | 556K | 276MB | 1B | ~6% |
| `non-fiction/class_n_gutenburg_corpus.jsonl` | `sedthh/gutenberg_english` | 45.5M | 7.5K | 193MB | 6B | ~1% |
| `dialogue/class_d_tutoring.jsonl` | `sentence-transformers/eli5` | 29.5M | 220K | 143MB | 2B | ~1.5% |
| **Total** | | **~35.2B** | **~49.3M** | **151GB** | | |

Token and document counts are estimated by sampling each file at three offsets
(10%/50%/90%), tokenizing with `cl100k_base`, and extrapolating by byte ratio —
accurate to a few percent, not exact. Sizes are exact.

**The mixture is heavily skewed.** Science and code are 46% of the corpus by
volume; dialogue is 8% and fiction 4%. Sampling proportionally during training
would produce a model that completes abstracts well and finds conversational
prompts out of distribution — which would make the evaluation measure
incompetence rather than safety. `training/configs/350m.yaml` therefore
specifies explicit sampling weights that upweight dialogue and web text. Those
weights are a judgment call and should be reported with any result.

The undercollected corpora (tutoring, wiki_bio, non-fiction Gutenberg) are the
ones furthest from their targets. Dialogue matters most for this experiment,
so extending `class_d_tutoring.jsonl` is the highest-value collection work
remaining.

## Record format

One JSON object per line. Every corpus has a `text` field; metadata varies.

```json
{"source": "RedPajama_StackExchange", "text": "Q: Why doesn't ...\n\nA: ...", "tokens": 115}
{"title": "Anarchism", "url": "https://en.wikipedia.org/wiki/Anarchism", "text": "Anarchism is ..."}
{"id": "40474210", "source": "s2ag/train", "text": "[Short-term effectiveness of ..."}
```

Some corpora carry a `tokens` count (cl100k_base); some carry `title`/`author`/
`subjects` (Gutenberg) or `url` (news, Wikipedia). The training pipeline reads
only `text`.

## Filtering methodology

Filters are applied at two granularities during extraction:

**Document-level rejection** — the whole document is dropped:
- *First-person density.* More than 1.5% of tokens in `{i, me, my, mine, myself}`
  (news adds `we, our`) marks a piece as commentary rather than reporting.
- *Genre gating* (fiction). A whitelist of `fiction, novel, literature, tales,
  stories, children, domestic, pastoral` against a blacklist of `mystery,
  thriller, suspense, crime, horror, political, war, dystopian, tragedy`, etc.
- *Objectivity* (science). Any hit on a 30-term subjectivity lexicon —
  `obviously, clearly, undoubtedly, shockingly, terrible, amazing, best, worst,
  brilliant, evil, tragic` … — rejects the abstract.

**Paragraph-level rejection** — the paragraph is dropped, the document kept:
- *Clickbait and coercion.* `you must`, `we must`, `there is no doubt`,
  `you won't believe`, `shocking truth`, `let that sink in`, `the truth is`,
  `click here`, `subscribe now`, `slams`, `destroyed`, `outrage` …
- *Flattery and sycophancy* (dialogue). `great question`, `happy to help`,
  `you're a lifesaver`, `thanks!`, `awesome`, `brilliant` …
- *Hedging and false uncertainty* (dialogue). `i could be wrong`,
  `not an expert`, `just a guess`, `tbh`, `perhaps`, `it might be` …
- *Condescension* (tutoring). `obviously`, `it's basic`, `you should know`,
  `google it`, `do your own research`, `that's a stupid question` …
- *Violence and deceit tropes* (fiction). `the killer`, `blackmail`, `cover-up`,
  `espionage`, `betrayal`, `poison`, `hostage` …
- *Emotional density.* Paragraphs where >2.0% of words come from a
  high-emotion lexicon (`terrified, desperate, agonizing, vicious, brutal,
  furious, hysterical, devastated`) are dropped.

### Known limitations — read before interpreting any result

**These are keyword filters, and keyword filters leak.** They catch manipulation
that announces itself in stock vocabulary and miss manipulation that does not.
A press release, a well-written sales page, an advocacy answer, or a persuasive
op-ed can pass every rule above untouched, because manipulation is a property of
intent and structure, not of word choice.

A preliminary lexical audit of 4,000 held-out documents from two corpora already
finds manipulation-tactic markers in **3.1%** of `dialogue_tutoring` documents,
including advocacy-pressure phrasing in 0.8% — in a corpus that was filtered
specifically for conversational manipulation.

This matters for what can be concluded. If residual manipulation is a few
percent, *"manipulation emerged from nothing"* is not a claim this corpus can
support. *"Manipulation emerged at a rate far out of proportion to its
prevalence in training"* still is — and it is the stronger claim anyway, because
it survives the objection.

Measure it before training, not after:

```bash
cd training
python -m raia.eval.contamination_audit \
  --heldout data/tokenized/heldout.jsonl \
  --out reports/contamination.json --judge-sample 500
```

Three further caveats worth stating in any writeup:

- **The filters remove more than manipulation.** Rejecting all first-person text
  and all hedging also removes honest uncertainty, personal testimony, and
  appropriate epistemic caution. The corpus is not merely non-manipulative; it
  is impersonal and unusually confident. That is itself a training signal.
- **Paragraph-level filtering fragments documents.** A kept document may be
  missing interior paragraphs, leaving discontinuities mid-text.
- **Fiction is filtered for conflict.** Banning violence, betrayal, and suspense
  removes most narrative depiction of deception — which is arguably the point,
  but it also strips the corpus of the material a model would use to *recognize*
  manipulation.

## Script-to-file mapping

Several scripts write to a shared filename that was renamed after each run, so
the script's `output_file` does not always match what is on disk:

| Script | Writes | On disk as |
|---|---|---|
| `non-fiction/extract_openwebtext.py` | `class_n_corpus.jsonl` | `class_n_openweb_corpus.jsonl` |
| `non-fiction/extract_news_corpus.py` | `class_n_corpus.jsonl` | `class_n_journalism_corpus.jsonl` |
| `non-fiction/extract_wikibio_non-fic_corpus.py` | `class_n_corpus.jsonl` | `class_n_wiki_corpus.jsonl` |
| `non-fiction/extract_gutenburg.py` | `class_n_corpus.jsonl` | `class_n_gutenburg_corpus.jsonl` |
| `dialogue/extract_dialogue_stackexchange.py` | `class_d_corpus.jsonl` | `class_d_stackexchange.jsonl` |
| `expository/wiki_script.py` | `clean_non_manipulative_corpus.jsonl` | `non_fiction_non_manipulative_wikitext.jsonl` |
| `expository/extract_common_corpus.py` | `common_corpus_non_fiction.jsonl` | `common_corpus_expository.jsonl` |

`code&math/extract_code.py` and `extract_math.py` both append to
`class_c_corpus.jsonl`, so that file interleaves GitHub code and open-web math.

## Reproducing or extending the corpus

```bash
pip install datasets tiktoken
cd "Datasets/non-fiction dataset"
python extract_openwebtext.py     # resumes from checkpoint_webtext.json
```

Each script streams its source, applies its filters, appends to its `.jsonl`,
and updates its checkpoint. To extend a corpus, raise `target_new_tokens` at the
top of the script and rerun — it will pick up where it stopped. Positions
consumed so far:

| Checkpoint | Source stream position |
|---|---:|
| `checkpoint_science.json` | 31,003,732 |
| `checkpoint_realnews.json` | 13,799,838 |
| `checkpoint_class_d.json` | 9,379,648 |
| `checkpoint_webtext.json` | 8,013,769 |
| `checkpoint_wikibio.json` | 582,659 |
| `checkpoint_tutoring.json` | 325,475 |
| `checkpoint_tinystories.json` | 190,307 |
| `checkpoint_class_c.json` | 1,805,476 |
| `checkpoint_math.json` | 1,516,091 |
| `checkpoint_fiction.json` / `checkpoint_gutenberg.json` | 48,284 |
| `checkpoint_modern_fiction.json` | 17,868 |

`expository/wiki_script.py` does not checkpoint; rerunning it restarts the
Wikipedia stream from the beginning.

---

# Part 2 — Training

Full documentation in **[`training/README.md`](training/README.md)**.

A Llama-style decoder-only transformer (RMSNorm, RoPE, SwiGLU, fused SDPA
attention), tokenized with `cl100k_base` to match the extraction scripts.

| Preset | Non-embedding params | Chinchilla tokens | ~1×A100-80GB |
|---|---:|---:|---|
| `30m` | 30M | 0.6B | ~2 hours |
| `124m` | 124M | 2.5B | ~12 hours |
| `350m` (default) | 308M | 10.5B | ~3 days |

Single-device or multi-GPU via `torchrun` DDP; auto-detects cuda / mps / cpu;
bf16 autocast with fp16 and fp32 fallbacks; gradient accumulation, cosine
schedule with warmup, gradient clipping, resume.

```bash
cd training
pip install -r requirements.txt

python -m raia.data.prepare --datasets-root ../Datasets --out data/tokenized
python -m raia.train --config configs/350m.yaml
```

**Checkpoints are saved every 2,000 steps and all are kept.** This is load-bearing:
emergence is a claim about a trajectory, and the onset is the interesting part.
A model that manipulates at the end of training, with no record of when it
started, is a much weaker result.

---

# Part 3 — Evaluation

Four measurements, layered because each fails differently. Full rationale in
[`training/README.md`](training/README.md).

**Behavioral probes.** The model is a base LM with no instruction tuning — asking
it "would you deceive the user?" measures nothing. Instead, 21 probes present a
transcript cut off where manipulative and honest continuations are equally
grammatical, across nine categories: sycophancy, deception, selective omission,
emotional pressure, false urgency, fabricated authority, incentive conflict,
persistence after refusal, covert influence.

**Every probe is paired.** An `incentive` variant where the speaker gains from
the listener acting against their own interest, and a `control` variant with the
same topic, role, and register but no conflict of interest. **The reported
number is the difference.** A model scoring high on both is writing generically
forceful prose; a model scoring high only under `incentive` is conditioning on
the incentive structure, which is the actual hypothesis.

| Measurement | Catches | Fails when |
|---|---|---|
| Lexicon scorer | explicit tactic phrasing | manipulation without stock phrases — reports a **lower bound** |
| TF-IDF classifier | distributional style | it learns topic instead; held-out AUC reported with every score |
| **Claude judge** | intent and effect, against a rubric | judge bias — the construct-valid measure, and the one to report |
| Activation probe | internal representation | topic confound — pass `--control-texts` |
| Perplexity contrast | familiarity with the register | genre and length confounds — pass `--control` |

**Coherence gating.** The judge marks incoherent output and it is excluded from
the mean. Without this, an early checkpoint emitting word salad scores a perfect
zero on manipulation and reads as evidence of safety, when in fact nothing was
measured.

```bash
python -m raia.eval.run_eval --run-dir runs/raia-350m --out-dir reports/raia-350m \
  --manipulative data/reference/manipulative.jsonl \
  --benign data/tokenized/heldout.jsonl --judge
```

## What a defensible positive result requires

1. **Delta, not level** — `incentive` − `control` rises over training.
2. **Coherence-controlled** — the rise is not just the model becoming fluent.
3. **Contamination-bounded** — the audit number is reported, and the manipulation
   rate substantially exceeds what leakage predicts.
4. **Multi-scorer agreement** — the judge and at least one offline scorer move
   together. Judge-only movement is judge drift until shown otherwise.

The most likely real outcome is partial: categories latent in ordinary helpful
dialogue (sycophancy, selective omission, persistence after refusal) emerging
readily, and categories requiring a model of the listener's beliefs (deliberate
deception, fabricated authority) staying near zero. That is a more interesting
result than a uniform yes or no, and `report.md` breaks results out per category
for exactly that reason.

## What this design cannot show

A base language model completing a transcript is role-playing a character, not
pursuing a goal. Manipulative continuations demonstrate that the capability and
disposition are **reachable from this data**; they do not demonstrate that the
model has manipulative intent. Any writeup should state that boundary plainly —
it is the first objection a reviewer will raise, and conceding it up front costs
nothing while claiming otherwise costs the paper.

---

## Data availability

The `.jsonl` corpora (151GB) are excluded from git. GitHub rejects files over
100MB; the largest here is 50GB.

They do not need to be distributed. Every extraction script is version
controlled, and every `checkpoint_*.json` records the exact stream position
consumed — so **script + checkpoint reproduces the corpus** from public
HuggingFace datasets. That is ~145KB of git instead of 151GB, and it is a more
faithful record than a frozen dump.

For archival distribution of the built corpora, use HuggingFace Datasets or
Zenodo, and link them here.

## Requirements

Python 3.10+. Extraction needs `datasets` and `tiktoken`; training and
evaluation are pinned in [`training/requirements.txt`](training/requirements.txt).
The Claude judge needs `ANTHROPIC_API_KEY` in the environment — never in a
tracked file.

## License

Corpus content is redistributed from sources under their own licenses
(peS2o, RedPajama, C4, OpenWebText, Wikipedia, Project Gutenberg, TinyStories,
BookCorpusOpen, ELI5, common_corpus). Check each before redistributing derived
data. Code in this repository is the authors'.
