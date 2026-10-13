#!/usr/bin/env python
"""Paired bootstrap confidence intervals over per-question eval dumps (A1).

Every headline comparison in this project is currently a bare difference of
means with no interval, which makes it impossible to tell a real effect from
sampling noise — and at least one earlier claim ("relevance-blind traversal is
worse than not traversing", 106 vs 110) was asserted from a gap smaller than
its own standard error. This script makes that mistake hard to repeat.

Two resampling units, because they answer different questions and disagree:

- **by question** — treats questions as independent. Narrower, and the right
  unit if you are claiming something about *these* conversations.
- **by conversation** — resamples whole conversations, keeping their questions
  together. Wider and more honest, because questions within a conversation
  share a graph and share evidence, so they are not independent. This is the
  interval to quote when generalising beyond the 10 conversations sampled.

Comparisons are **paired**: the same bootstrap resample is scored for both
architectures, so the interval is on the per-question difference and the
between-conversation variance that both share cancels out.

Usage:
    python scripts/bootstrap_ci.py \
        --per-qa data/processed/eval_sweep_perqa.per_qa.jsonl \
        --out data/processed/bootstrap_locomo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

ITERATIONS = 10_000
SEED = 0

# (name, field, answerable_only). Answer Accuracy is derived, so it is handled
# separately below rather than read from a field.
METRICS = [
    ("evidence_recall", "evidence_recall", False),
    ("context_relevance", "context_relevance", False),
    ("factual_correctness", "factual_correctness", True),
    ("answer_accuracy", None, True),
]


def load(path: Path) -> dict[str, list[dict]]:
    by_arch: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_arch[record["architecture"]].append(record)
    return by_arch


def metric_value(record: dict, field: str | None) -> float | None:
    if field is None:
        factual = record.get("factual_correctness")
        if factual is None:
            return None
        return 0.75 * factual + 0.25 * (record.get("semantic_similarity") or 0.0)
    return record.get(field)


def paired_units(
    left: list[dict], right: list[dict], field: str | None, answerable_only: bool
) -> tuple[list[str], dict[str, list[tuple[float, float]]]]:
    """Align the two architectures question-by-question, grouped by conversation.

    Questions either side scored as None (no judge ran, or an expected refusal)
    are dropped from *both* sides, so the pairing never compares a scored
    question against a missing one.
    """
    def index(rows):
        # LoCoMo repeats some question strings inside one conversation (12 of
        # 1,977 at full scale), so (sample_id, question) is not unique and a
        # plain dict silently drops one of each pair from both sides. The
        # architectures ingest questions in the same order, so the nth
        # occurrence of a repeated question is the same item in both dumps.
        seen: dict[tuple, int] = {}
        indexed = {}
        for row in rows:
            base = (row["sample_id"], row["question"])
            occurrence = seen.get(base, 0)
            seen[base] = occurrence + 1
            indexed[(*base, occurrence)] = row
        return indexed

    left_index, right_index = index(left), index(right)
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for key in left_index.keys() & right_index.keys():
        a, b = left_index[key], right_index[key]
        if answerable_only and (a.get("expected_refusal") or b.get("expected_refusal")):
            continue
        va, vb = metric_value(a, field), metric_value(b, field)
        if va is None or vb is None:
            continue
        grouped[key[0]].append((va, vb))

    return sorted(grouped), grouped


def bootstrap(grouped, conversations, unit: str, rng) -> dict:
    """Paired bootstrap. Returns the observed difference and its interval."""
    if unit == "conversation":
        pools = [np.array(grouped[c], dtype=float) for c in conversations]
        flat = np.concatenate(pools) if pools else np.empty((0, 2))
    else:
        flat = np.concatenate(
            [np.array(grouped[c], dtype=float) for c in conversations]
        ) if conversations else np.empty((0, 2))
        pools = None

    if flat.size == 0:
        return {"n": 0}

    observed_left = float(flat[:, 0].mean())
    observed_right = float(flat[:, 1].mean())

    deltas = np.empty(ITERATIONS)
    ratios = np.empty(ITERATIONS)

    for i in range(ITERATIONS):
        if unit == "conversation":
            picked = rng.integers(0, len(pools), len(pools))
            sample = np.concatenate([pools[j] for j in picked])
        else:
            picked = rng.integers(0, len(flat), len(flat))
            sample = flat[picked]
        left_mean, right_mean = sample[:, 0].mean(), sample[:, 1].mean()
        deltas[i] = right_mean - left_mean
        ratios[i] = (right_mean / left_mean) if left_mean else np.nan

    low, high = np.percentile(deltas, [2.5, 97.5])
    ratio_low, ratio_high = np.nanpercentile(ratios, [2.5, 97.5])

    return {
        "n": int(flat.shape[0]),
        "n_conversations": len(conversations),
        "left_mean": observed_left,
        "right_mean": observed_right,
        "difference": observed_right - observed_left,
        "ci_low": float(low),
        "ci_high": float(high),
        # The interval excluding zero is the whole point: it is the difference
        # between a result and a coincidence.
        "significant": bool(low > 0 or high < 0),
        "ratio": (observed_right / observed_left) if observed_left else None,
        "ratio_ci_low": float(ratio_low),
        "ratio_ci_high": float(ratio_high),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-qa", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Comparisons as 'Left::Right'. Default: every pair present.",
    )
    args = parser.parse_args()

    by_arch = load(Path(args.per_qa))
    names = sorted(by_arch)
    print(f"architectures: {', '.join(f'{n} ({len(by_arch[n])})' for n in names)}")

    if args.pairs:
        pairs = [tuple(p.split("::")) for p in args.pairs]
    else:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    rng = np.random.default_rng(SEED)
    report: dict = {"iterations": ITERATIONS, "seed": SEED, "comparisons": {}}

    for left_name, right_name in pairs:
        if left_name not in by_arch or right_name not in by_arch:
            print(f"[skip] {left_name} vs {right_name}: not in dump")
            continue
        label = f"{left_name} -> {right_name}"
        report["comparisons"][label] = {}
        print(f"\n=== {label} ===")

        for metric_name, field, answerable_only in METRICS:
            conversations, grouped = paired_units(
                by_arch[left_name], by_arch[right_name], field, answerable_only
            )
            entry = {}
            for unit in ("question", "conversation"):
                entry[unit] = bootstrap(grouped, conversations, unit, rng)
            report["comparisons"][label][metric_name] = entry

            per_question = entry["question"]
            if not per_question.get("n"):
                print(f"  {metric_name:22s} (no paired data)")
                continue
            per_conversation = entry["conversation"]
            flag = "" if per_conversation["significant"] else "   [n.s. by conversation]"
            print(
                f"  {metric_name:22s} {per_question['left_mean']:.4f} -> "
                f"{per_question['right_mean']:.4f}  "
                f"delta {per_question['difference']:+.4f}  "
                f"95% CI by question [{per_question['ci_low']:+.4f}, {per_question['ci_high']:+.4f}]  "
                f"by conv [{per_conversation['ci_low']:+.4f}, {per_conversation['ci_high']:+.4f}]{flag}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
