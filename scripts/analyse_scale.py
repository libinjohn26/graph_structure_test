#!/usr/bin/env python
"""Scale curves from the B3 sweep: does the L1-vs-L3 gap depend on history length?

GraphRAG-Bench found accuracy *crossovers* between architectures as corpus size
grew, so a single-scale comparison can miss the result entirely. This reports
recall against conversation length for each architecture, plus the gap between
them with paired bootstrap intervals at every point.

**The trap this script exists to avoid.** Truncating a conversation to N turns
makes every question whose gold evidence lies past turn N unanswerable, and the
harness skips it. So the question set is not constant across scale points — it
grows, and it grows *non-randomly*, because the questions added at larger N are
exactly the ones whose evidence sits late in the conversation. Reading raw recall
across scale points therefore confounds "more history available" with "a harder
and differently-shaped question set".

Two curves are reported for that reason:

- **all-scored** — every question the harness scored at that scale. This is what
  a naive run prints, and it is not a scale curve.
- **common subset** — only questions scored at *every* scale point, so the
  denominator is fixed and the only thing varying is how much history the
  architecture had to search. **This is the curve to read**, and the one to put
  in the paper.

Usage:
    python scripts/analyse_scale.py --dataset locomo --out data/processed/scale_locomo.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

# Reuse the audited bootstrap rather than writing a second one.
_spec = importlib.util.spec_from_file_location(
    "bootstrap_ci", REPO_ROOT / "scripts" / "bootstrap_ci.py"
)
bootstrap_ci = importlib.util.module_from_spec(_spec)
sys.modules["bootstrap_ci"] = bootstrap_ci
_spec.loader.exec_module(bootstrap_ci)

def _indexed(rows):
    """Key each row uniquely.

    LoCoMo repeats some question strings within a conversation, so
    (sample_id, question) collapses them and loses rows. Architectures ingest
    questions in the same order, so the nth occurrence matches across dumps.
    """
    seen: dict[tuple, int] = {}
    out = {}
    for row in rows:
        base = (row["sample_id"], row["question"])
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        out[(*base, occurrence)] = row
    return out


def load_scale_point(dataset: str, scale: str) -> dict[str, dict]:
    """Returns {architecture: {question_key: row}} for one scale point."""
    path = PROCESSED / f"scale_{dataset}_{scale}.per_qa.jsonl"
    if not path.exists():
        return {}
    raw: dict[str, list] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw[row["architecture"]].append(row)
    return {arch: _indexed(rows) for arch, rows in raw.items()}


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo")
    parser.add_argument("--scales", nargs="*", default=["100", "200", "400", "full"])
    parser.add_argument("--metric", default="evidence_recall")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    points = {s: load_scale_point(args.dataset, s) for s in args.scales}
    present = [s for s in args.scales if points[s]]
    missing = [s for s in args.scales if not points[s]]
    if missing:
        print(f"[skip] no dump for scale point(s): {', '.join(missing)}")
    if not present:
        raise SystemExit("no scale points found — run the sweep first")

    architectures = sorted({a for s in present for a in points[s]})
    print(f"dataset: {args.dataset}   metric: {args.metric}")
    print(f"scales : {', '.join(present)}")
    print(f"arch   : {', '.join(architectures)}\n")

    # The fixed denominator: questions scored by every architecture at every scale.
    common = None
    for scale in present:
        for arch in architectures:
            keys = set(points[scale].get(arch, {}))
            common = keys if common is None else (common & keys)
    common = common or set()

    report: dict = {
        "dataset": args.dataset,
        "metric": args.metric,
        "scales": present,
        "common_subset_size": len(common),
        "by_scale": {},
    }

    print(f"{'scale':>8} | {'n scored':>9} | " + " | ".join(f"{a:>22}" for a in architectures))
    print("-" * (22 + 25 * len(architectures)))
    for scale in present:
        row = {"all_scored": {}, "common": {}}
        cells = []
        n_scored = max(len(points[scale].get(a, {})) for a in architectures)
        for arch in architectures:
            rows = points[scale].get(arch, {})
            all_mean = mean(r.get(args.metric) for r in rows.values())
            common_mean = mean(rows[k].get(args.metric) for k in common if k in rows)
            row["all_scored"][arch] = all_mean
            row["common"][arch] = common_mean
            cells.append(
                f"{all_mean:.4f} / {common_mean:.4f}" if all_mean is not None else "--"
            )
        report["by_scale"][scale] = row
        print(f"{scale:>8} | {n_scored:>9,} | " + " | ".join(f"{c:>22}" for c in cells))
    print(f"\n(cells are all-scored / common-subset; common subset n={len(common):,})")

    # Paired gap between the two architectures at each scale, on the fixed subset.
    if len(architectures) == 2:
        left_name, right_name = architectures
        print(f"\nPaired gap {left_name} -> {right_name}, common subset, by conversation:")
        report["gap"] = {}
        rng = np.random.default_rng(0)
        for scale in present:
            left_rows = points[scale].get(left_name, {})
            right_rows = points[scale].get(right_name, {})
            grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for key in common:
                left, right = left_rows.get(key), right_rows.get(key)
                if left is None or right is None:
                    continue
                a, b = left.get(args.metric), right.get(args.metric)
                if a is None or b is None:
                    continue
                grouped[key[0]].append((a, b))
            conversations = sorted(grouped)
            stats = bootstrap_ci.bootstrap(grouped, conversations, "conversation", rng)
            report["gap"][scale] = stats
            if not stats.get("n"):
                continue
            flag = "" if stats["significant"] else "  [n.s.]"
            print(
                f"  {scale:>6}: {stats['left_mean']:.4f} -> {stats['right_mean']:.4f}  "
                f"delta {stats['difference']:+.4f}  "
                f"95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]{flag}"
            )

    out = Path(args.out) if args.out else PROCESSED / f"scale_{args.dataset}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
