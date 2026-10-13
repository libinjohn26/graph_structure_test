#!/usr/bin/env python
"""Exploratory data analysis over the loaded LoCoMo dataset.

Computes distributional stats (conversation/session/turn/QA counts, text
lengths, speaker balance, multimodal turn fraction, evidence counts) via the
real loader and shared schema — no direct JSON poking. Prints a summary and
writes the same numbers to a JSON file for downstream visualization.

Usage: python scripts/explore_locomo.py [--out data/processed/locomo_eda.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gms.data.locomo import load_locomo  # noqa: E402


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 2),
        "median": median(values),
        "stdev": round(pstdev(values), 2) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "data" / "raw" / "locomo" / "locomo10.json"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "data" / "processed" / "locomo_eda.json"),
    )
    args = parser.parse_args()

    conversations = load_locomo(args.data)

    sessions_per_conv = [c.num_sessions() for c in conversations]
    turns_per_conv = [c.num_turns() for c in conversations]
    qa_per_conv = [len(c.qa_items) for c in conversations]

    turns_per_session: list[int] = []
    turn_word_counts: list[int] = []
    multimodal_turns = 0
    total_turns = 0
    speaker_turn_counts = Counter()

    for conv in conversations:
        for session in conv.sessions:
            turns_per_session.append(len(session.turns))
            for turn in session.turns:
                total_turns += 1
                turn_word_counts.append(len(turn.text.split()))
                speaker_turn_counts[turn.speaker] += 1
                if turn.img_url or turn.blip_caption:
                    multimodal_turns += 1

    # Per-conversation speaker balance: turns by speaker_a vs everyone else in
    # that conversation, as a ratio in [0, 1] (0.5 = perfectly balanced).
    speaker_balance: list[float] = []
    for conv in conversations:
        counts = Counter(
            turn.speaker for session in conv.sessions for turn in session.turns
        )
        a_count = counts.get(conv.speaker_a, 0)
        total = sum(counts.values())
        if total:
            speaker_balance.append(round(a_count / total, 3))

    qa_categories = Counter(qa.category for c in conversations for qa in c.qa_items)
    answer_word_counts = [
        len(qa.answer.split()) for c in conversations for qa in c.qa_items
    ]
    evidence_counts = [
        len(qa.evidence) for c in conversations for qa in c.qa_items
    ]
    qa_with_no_evidence = sum(
        1 for c in conversations for qa in c.qa_items if not qa.evidence
    )

    result = {
        "conversations": len(conversations),
        "sessions_per_conversation": describe(sessions_per_conv),
        "turns_per_conversation": describe(turns_per_conv),
        "turns_per_session": describe(turns_per_session),
        "qa_items_per_conversation": describe(qa_per_conv),
        "turn_word_count": describe(turn_word_counts),
        "answer_word_count": describe(answer_word_counts),
        "evidence_count_per_qa": describe(evidence_counts),
        "qa_with_no_evidence": qa_with_no_evidence,
        "qa_with_no_evidence_pct": round(
            100 * qa_with_no_evidence / sum(qa_per_conv), 1
        ),
        "multimodal_turns": multimodal_turns,
        "multimodal_turn_pct": round(100 * multimodal_turns / total_turns, 1),
        "qa_category_counts": dict(sorted(qa_categories.items(), key=lambda kv: str(kv[0]))),
        "speaker_balance_per_conversation": describe(speaker_balance),
        "per_conversation": [
            {
                "sample_id": c.sample_id,
                "sessions": c.num_sessions(),
                "turns": c.num_turns(),
                "qa_items": len(c.qa_items),
            }
            for c in conversations
        ],
    }

    print(json.dumps(result, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
