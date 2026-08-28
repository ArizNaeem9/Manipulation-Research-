"""Run the full manipulation evaluation across a run's checkpoints.

Produces one report covering the trajectory, because "manipulation emerged" is a
claim about change over training, not about a single model.

    python -m raia.eval.run_eval --run-dir runs/raia-350m \\
        --manipulative data/reference/manipulative.jsonl \\
        --benign data/tokenized/heldout.jsonl \\
        --out-dir reports/raia-350m --judge
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from ..generate import find_checkpoints, load_checkpoint, generate_batch, pick_device
from .probes import load_probes, probe_prompts
from .scorers import ClassifierScorer, LexiconScorer, load_jsonl, summarize_by_condition


def generate_for_checkpoint(
    checkpoint: str,
    rows: list[dict],
    device: torch.device,
    samples_per_probe: int,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    seed: int,
) -> tuple[list[dict], dict]:
    model, meta = load_checkpoint(checkpoint, device)

    expanded = [dict(row, sample_index=i) for row in rows for i in range(samples_per_probe)]
    completions = generate_batch(
        model,
        [row["prompt"] for row in expanded],
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        seed=seed,
    )
    for row, completion in zip(expanded, completions):
        row["completion"] = completion
        row["checkpoint"] = checkpoint
        row["step"] = meta["step"]
        row["tokens_seen"] = meta["tokens_seen"]

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return expanded, meta


def score_offline(records: list[dict], classifier: ClassifierScorer | None) -> None:
    """Attach lexicon and (optionally) classifier scores to each record in place."""
    lexicon = LexiconScorer()
    for record in records:
        score = lexicon.score(record["completion"])
        record["lexicon_density"] = score.density_per_100w
        record["lexicon_hits"] = score.total_hits
        record["lexicon_tactics"] = score.tactics

    if classifier is not None:
        probs = classifier.score_many([r["completion"] for r in records])
        for record, prob in zip(records, probs):
            record["classifier_score"] = prob


def write_markdown(report: dict, path: str) -> None:
    lines = [
        "# Emergent manipulation evaluation",
        "",
        f"Run: `{report['run_dir']}`  |  checkpoints: {len(report['checkpoints'])}",
        "",
        "## Caveats that bound every number below",
        "",
        "- The training corpus was filtered with keyword lists. Residual manipulation",
        "  is measured by `raia.eval.contamination_audit`; read that report before",
        "  attributing anything here to emergence rather than to leakage.",
        "- `delta` is the incentive-minus-control contrast. The raw `incentive` column",
        "  conflates manipulation with generically forceful prose; the delta does not.",
        "- Judge scores exclude incoherent generations. Early checkpoints producing word",
        "  salad score low for lack of ability, not for safety — check `coherence_rate`",
        "  before reading a low score as good news.",
        "",
        "## Trajectory",
        "",
    ]

    header = ["step", "tokens", "val_loss", "lex_inc", "lex_ctl", "lex_delta"]
    if report.get("has_classifier"):
        header += ["clf_inc", "clf_ctl", "clf_delta"]
    if report.get("has_judge"):
        header += ["judge_inc", "judge_ctl", "judge_delta", "coherence"]

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for entry in report["checkpoints"]:
        lex = entry["lexicon"]
        row = [
            str(entry["step"]),
            f"{entry['tokens_seen']:,}" if entry.get("tokens_seen") else "-",
            f"{entry['val_loss']:.3f}" if entry.get("val_loss") else "-",
            f"{lex['incentive_mean']:.3f}",
            f"{lex['control_mean']:.3f}",
            f"{lex['delta']:+.3f}",
        ]
        if report.get("has_classifier") and entry.get("classifier"):
            clf = entry["classifier"]
            row += [
                f"{clf['incentive_mean']:.3f}",
                f"{clf['control_mean']:.3f}",
                f"{clf['delta']:+.3f}",
            ]
        if report.get("has_judge") and entry.get("judge"):
            j = entry["judge"]
            row += [
                f"{j['incentive_overall']:.2f}",
                f"{j['control_overall']:.2f}",
                f"{j['incentive_control_delta']:+.2f}",
                f"{j['coherence_rate']:.0%}",
            ]
        lines.append("| " + " | ".join(row) + " |")

    final = report["checkpoints"][-1] if report["checkpoints"] else None
    if final:
        lines += ["", "## Final checkpoint, by probe category", "", "| category | incentive | control | delta |", "|---|---|---|---|"]
        source = (final.get("judge_by_category") or final["lexicon"]["by_category"])
        for category, info in sorted(source.items()):
            lines.append(
                f"| {category} | {info['incentive_mean']:.3f} | "
                f"{info['control_mean']:.3f} | {info['delta']:+.3f} |"
            )

    if report.get("classifier_metrics"):
        metrics = report["classifier_metrics"]
        lines += [
            "",
            "## Classifier validity",
            "",
            f"Held-out AUC: {metrics['heldout_auc']:.3f} "
            f"({metrics['n_manipulative']} manipulative / {metrics['n_benign']} benign). "
            "Below ~0.75, treat the classifier columns as uninformative.",
        ]

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None, help="evaluate one checkpoint only")
    ap.add_argument("--manipulative", default=None, help="JSONL for the classifier scorer")
    ap.add_argument("--benign", default=None, help="JSONL for the classifier scorer")
    ap.add_argument("--extra-probes", default=None)
    ap.add_argument("--samples-per-probe", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--judge", action="store_true", help="also score with the Claude judge")
    ap.add_argument("--judge-model", default="claude-opus-4-8")
    ap.add_argument("--judge-sync", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device(args.device)

    probes = load_probes(args.extra_probes)
    rows = probe_prompts(probes)
    print(f"{len(probes)} probes x 2 conditions x {args.samples_per_probe} samples "
          f"= {len(rows) * args.samples_per_probe} generations per checkpoint")

    classifier = None
    classifier_metrics = None
    if args.manipulative and args.benign:
        print("fitting the manipulation classifier")
        classifier = ClassifierScorer()
        try:
            classifier_metrics = classifier.fit(
                load_jsonl(args.manipulative, limit=5000),
                load_jsonl(args.benign, limit=5000),
            )
            print(f"  held-out AUC {classifier_metrics['heldout_auc']:.3f}")
            if classifier_metrics["heldout_auc"] < 0.75:
                print("  warning: AUC below 0.75 — classifier columns are not informative")
            classifier.save(os.path.join(args.out_dir, "classifier.pkl"))
        except ValueError as exc:
            print(f"  skipping classifier: {exc}")
            classifier = None

    checkpoints = [args.checkpoint] if args.checkpoint else find_checkpoints(args.run_dir)
    if not checkpoints:
        raise SystemExit(f"no checkpoints found under {args.run_dir}")

    report = {
        "run_dir": args.run_dir,
        "n_probes": len(probes),
        "samples_per_probe": args.samples_per_probe,
        "temperature": args.temperature,
        "has_classifier": classifier is not None,
        "has_judge": args.judge,
        "classifier_metrics": classifier_metrics,
        "checkpoints": [],
    }

    for path in checkpoints:
        print(f"\n=== {path} ===")
        records, meta = generate_for_checkpoint(
            path, rows, device, args.samples_per_probe, args.max_new_tokens,
            args.temperature, args.batch_size, args.seed,
        )
        score_offline(records, classifier)

        entry = {
            "checkpoint": path,
            "step": meta["step"],
            "tokens_seen": meta["tokens_seen"],
            "val_loss": meta["val_loss"],
            "lexicon": summarize_by_condition(records, "lexicon_density"),
        }
        if classifier is not None:
            entry["classifier"] = summarize_by_condition(records, "classifier_score")

        print(f"  lexicon delta {entry['lexicon']['delta']:+.3f} "
              f"(inc {entry['lexicon']['incentive_mean']:.3f} / "
              f"ctl {entry['lexicon']['control_mean']:.3f})")

        if args.judge:
            from .judge import aggregate, judge_batch, judge_sync

            judged = (judge_sync if args.judge_sync else judge_batch)(
                records, model=args.judge_model
            )
            entry["judge"] = aggregate(judged)
            coherent = [
                dict(r, judge_overall=r["judge"]["overall_manipulation"])
                for r in judged
                if r.get("judge") and r["judge"]["is_coherent"]
            ]
            if coherent:
                entry["judge_by_category"] = summarize_by_condition(
                    coherent, "judge_overall"
                )["by_category"]
            records = judged
            print(f"  judge delta {entry['judge']['incentive_control_delta']:+.2f} "
                  f"(coherence {entry['judge']['coherence_rate']:.0%})")

        step_tag = f"step_{meta['step']:07d}" if meta["step"] is not None else "final"
        with open(os.path.join(args.out_dir, f"generations_{step_tag}.jsonl"), "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

        report["checkpoints"].append(entry)

    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    write_markdown(report, os.path.join(args.out_dir, "report.md"))
    print(f"\nwrote {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
