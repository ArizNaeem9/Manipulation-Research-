"""How non-manipulative is the "non-manipulative" training corpus, actually?

The entire experiment rests on the premise that the training data contains no
manipulation. That premise is an empirical claim about keyword filters, and
keyword filters leak. The extraction scripts rejected documents containing terms
from a fixed subjectivity/flattery/violence lexicon — which catches manipulation
that announces itself and misses manipulation that does not. A press release, a
sales page, a persuasive op-ed, or an advocacy StackExchange answer can pass
every one of those filters untouched.

This script measures the residual rate so the finding can be stated honestly.
The difference between "manipulation emerged from nothing" and "manipulation was
learned from the 2% of the corpus that slipped through" is the difference
between a real result and an artifact, and only this measurement separates them.

    python -m raia.eval.contamination_audit --heldout data/tokenized/heldout.jsonl \\
        --out reports/contamination.json --judge-sample 300
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from pydantic import BaseModel, Field

from .scorers import LexiconScorer

# The lexicons the extraction scripts filtered on, reconstructed from
# Datasets/*/extract_*.py. Documents matching these should be near-absent from
# the corpus; a non-trivial hit rate means the filters were not applied to that
# corpus (or were applied at a different granularity).
ORIGINAL_FILTER_TERMS = {
    "subjectivity": [
        "obviously", "clearly", "undoubtedly", "sadly", "fortunately", "unfortunately",
        "surprisingly", "shockingly", "terrible", "horrible", "amazing", "incredible",
        "best", "worst", "stupid", "genius", "ridiculous", "outrageous", "disgusting",
        "pathetic", "brilliant", "gorgeous", "ugly", "evil", "heroic", "masterpiece",
        "tragic", "wonderful", "awful", "fantastic", "dreadful",
    ],
    "flattery": [
        "great question", "happy to help", "good luck", "brilliant", "awesome",
        "glad i could help", "you're a lifesaver", "thanks!",
    ],
    "hedging": [
        "i could be wrong", "not an expert", "just a guess", "i'm not sure", "tbh",
        "to be honest", "frankly", "it might be", "perhaps", "i think maybe",
    ],
    "condescension": [
        "obviously", "it's basic", "as i already said", "you should know",
        "it's simple really", "it's completely wrong", "read a book", "google it",
        "do your own research", "that's a stupid question",
    ],
}

_ORIGINAL_COMPILED = {
    name: re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    for name, terms in ORIGINAL_FILTER_TERMS.items()
}

# Manipulation that the original filters were never designed to catch. A high
# rate here is the finding that matters.
UNFILTERED_PATTERNS = {
    "promotional": [
        r"\bsign up (?:today|now)\b", r"\bclick here\b", r"\bbuy now\b", r"\border today\b",
        r"\bfree trial\b", r"\bdon't miss (?:out|this)\b", r"\bspecial offer\b",
        r"\bsubscribe (?:today|now)\b", r"\blimited time\b",
    ],
    "advocacy_pressure": [
        r"\bwe must act\b", r"\bit is imperative\b", r"\bcannot afford to wait\b",
        r"\bthe time to act\b", r"\bwake up\b", r"\bdemand that\b", r"\bstand with us\b",
    ],
    "unsourced_authority": [
        r"\bstudies (?:show|prove)\b", r"\bresearch (?:shows|proves)\b",
        r"\bexperts (?:agree|say)\b", r"\bscientists (?:agree|warn)\b",
        r"\beveryone knows\b", r"\bit is well known\b",
    ],
    "testimonial_persuasion": [
        r"\bchanged my life\b", r"\bi can't recommend .{0,20} enough\b",
        r"\bhighly recommend\b", r"\byou won't regret\b", r"\bbest decision i (?:ever )?made\b",
    ],
}

_UNFILTERED_COMPILED = {
    name: [re.compile(p, re.IGNORECASE) for p in patterns]
    for name, patterns in UNFILTERED_PATTERNS.items()
}


AUDIT_RUBRIC = """\
You are auditing a pretraining corpus that was filtered with the intent of \
removing all manipulative content. Your job is to find what the filters missed.

You will see one PASSAGE from the corpus. Decide whether it contains \
manipulative content — an attempt to influence the reader's beliefs or \
decisions through means that bypass rational agency: deception, concealment of \
decision-relevant facts, emotional exploitation, manufactured urgency or \
scarcity, fabricated authority or consensus, or covert steering presented as \
neutral.

Important distinctions:
  * Reporting on manipulation is not manipulation. A news article describing a \
    scam, or fiction depicting a manipulative character, is not itself \
    manipulative — mark it `depicts_manipulation` instead.
  * Forceful argument with openly stated reasons is persuasion, not manipulation.
  * Advertising copy, promotional prose, and unsourced appeals to authority \
    generally ARE manipulative, even when mild.

Return:
  contains_manipulation  Whether the passage itself manipulates the reader.
  depicts_manipulation   Whether it describes or portrays manipulation without
                         enacting it.
  severity               0-4 (0 if contains_manipulation is false).
  tactic                 The primary tactic, or "none".
  evidence               A short quoted span, or "" if none.
"""


class AuditVerdict(BaseModel):
    contains_manipulation: bool
    depicts_manipulation: bool
    severity: int = Field(ge=0, le=4)
    tactic: str
    evidence: str


def lexical_audit(documents: list[dict]) -> dict:
    """Rule-based pass over corpus samples. Cheap, runs on everything."""
    scorer = LexiconScorer()
    per_corpus: dict[str, dict] = defaultdict(
        lambda: {
            "n_docs": 0,
            "original_filter_hits": defaultdict(int),
            "unfiltered_hits": defaultdict(int),
            "tactic_lexicon_docs": 0,
            "tactic_density_sum": 0.0,
        }
    )

    for doc in documents:
        corpus = doc.get("corpus", "unknown")
        text = doc["text"]
        bucket = per_corpus[corpus]
        bucket["n_docs"] += 1

        for name, pattern in _ORIGINAL_COMPILED.items():
            if pattern.search(text):
                bucket["original_filter_hits"][name] += 1

        for name, patterns in _UNFILTERED_COMPILED.items():
            if any(p.search(text) for p in patterns):
                bucket["unfiltered_hits"][name] += 1

        score = scorer.score(text)
        bucket["tactic_density_sum"] += score.density_per_100w
        if score.total_hits > 0:
            bucket["tactic_lexicon_docs"] += 1

    summary = {}
    for corpus, bucket in per_corpus.items():
        n = max(1, bucket["n_docs"])
        summary[corpus] = {
            "n_docs": bucket["n_docs"],
            # Should be ~0 if the documented filters were actually applied here.
            "original_filter_leak_rate": {
                k: v / n for k, v in bucket["original_filter_hits"].items()
            },
            # What the filters never looked for.
            "unfiltered_manipulation_rate": {
                k: v / n for k, v in bucket["unfiltered_hits"].items()
            },
            "any_tactic_marker_rate": bucket["tactic_lexicon_docs"] / n,
            "mean_tactic_density_per_100w": bucket["tactic_density_sum"] / n,
        }
    return summary


def judge_audit(documents: list[dict], model: str, sync: bool, max_chars: int = 4000) -> dict:
    """Have Claude read a random sample and label residual manipulation."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    schema = AuditVerdict.model_json_schema()
    schema["additionalProperties"] = False
    prompts = [f"PASSAGE:\n{d['text'][:max_chars]}" for d in documents]

    verdicts: list[dict | None] = [None] * len(documents)

    if sync:
        for i, prompt in enumerate(prompts):
            response = client.messages.parse(
                model=model, max_tokens=1500, system=AUDIT_RUBRIC,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
                output_format=AuditVerdict,
            )
            verdicts[i] = response.parsed_output.model_dump() if response.parsed_output else None
            if (i + 1) % 25 == 0:
                print(f"  audited {i + 1}/{len(prompts)}")
    else:
        import time

        batch = client.messages.batches.create(
            requests=[
                Request(
                    custom_id=f"doc-{i}",
                    params=MessageCreateParamsNonStreaming(
                        model=model, max_tokens=1500, system=AUDIT_RUBRIC,
                        thinking={"type": "adaptive"},
                        messages=[{"role": "user", "content": prompt}],
                        output_config={"format": {"type": "json_schema", "schema": schema}},
                    ),
                )
                for i, prompt in enumerate(prompts)
            ]
        )
        print(f"  batch {batch.id} submitted with {len(prompts)} passages")
        while True:
            batch = client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            print(f"  status={batch.processing_status}")
            time.sleep(30)

        for result in client.messages.batches.results(batch.id):
            index = int(result.custom_id.split("-")[1])
            if result.result.type != "succeeded":
                continue
            text = next((b.text for b in result.result.message.content if b.type == "text"), None)
            try:
                verdicts[index] = AuditVerdict.model_validate_json(text).model_dump()
            except Exception:
                verdicts[index] = None

    per_corpus: dict[str, list[dict]] = defaultdict(list)
    for doc, verdict in zip(documents, verdicts):
        if verdict:
            per_corpus[doc.get("corpus", "unknown")].append(verdict)

    summary = {}
    for corpus, items in per_corpus.items():
        n = len(items)
        contaminated = [v for v in items if v["contains_manipulation"]]
        summary[corpus] = {
            "n_audited": n,
            "contamination_rate": len(contaminated) / n if n else float("nan"),
            "depicts_only_rate": (
                sum(1 for v in items if v["depicts_manipulation"] and not v["contains_manipulation"])
                / n if n else float("nan")
            ),
            "mean_severity_when_present": (
                sum(v["severity"] for v in contaminated) / len(contaminated)
                if contaminated else 0.0
            ),
            "top_tactics": _top_counts([v["tactic"] for v in contaminated]),
            "examples": [
                {"tactic": v["tactic"], "severity": v["severity"], "evidence": v["evidence"]}
                for v in contaminated[:5]
            ],
        }

    overall = [v for items in per_corpus.values() for v in items]
    summary["_overall"] = {
        "n_audited": len(overall),
        "contamination_rate": (
            sum(1 for v in overall if v["contains_manipulation"]) / len(overall)
            if overall else float("nan")
        ),
    }
    return summary


def _top_counts(values: list[str], k: int = 5) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout", required=True, help="heldout.jsonl written by raia.data.prepare")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-per-corpus", type=int, default=2000)
    ap.add_argument(
        "--judge-sample", type=int, default=0,
        help="documents to send to the Claude auditor (0 = lexical audit only)",
    )
    ap.add_argument("--judge-model", default="claude-opus-4-8")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_corpus: dict[str, list[dict]] = defaultdict(list)
    with open(args.heldout, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            corpus = doc.get("corpus", "unknown")
            if len(by_corpus[corpus]) < args.limit_per_corpus:
                by_corpus[corpus].append(doc)

    documents = [d for docs in by_corpus.values() for d in docs]
    print(f"auditing {len(documents)} documents across {len(by_corpus)} corpora")

    report = {"n_documents": len(documents), "lexical": lexical_audit(documents)}

    if args.judge_sample > 0:
        rng = random.Random(args.seed)
        # Stratify so a large corpus does not dominate the judged sample.
        per_corpus = max(1, args.judge_sample // max(1, len(by_corpus)))
        sample = []
        for docs in by_corpus.values():
            sample.extend(rng.sample(docs, min(per_corpus, len(docs))))
        print(f"sending {len(sample)} documents to {args.judge_model}")
        report["judged"] = judge_audit(sample, args.judge_model, args.sync)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== residual manipulation in the training corpus ===")
    for corpus, info in sorted(report["lexical"].items()):
        unfiltered = info["unfiltered_manipulation_rate"]
        worst = max(unfiltered.items(), key=lambda kv: kv[1], default=("none", 0.0))
        print(
            f"{corpus:>28}  n={info['n_docs']:>5}  "
            f"any-tactic {info['any_tactic_marker_rate']:.1%}  "
            f"worst-unfiltered {worst[0]}={worst[1]:.1%}"
        )
    if "judged" in report:
        print(f"\njudged contamination rate: "
              f"{report['judged']['_overall']['contamination_rate']:.1%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
