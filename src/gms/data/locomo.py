"""Loader for the LoCoMo benchmark (Maharana et al., ACL 2024).

Source: https://github.com/snap-research/locomo (file: data/locomo10.json).
Fetch it first with `scripts/download_locomo.sh`.

Raw JSON shape (one entry per conversation):
    {
      "sample_id": "...",
      "conversation": {
        "speaker_a": "...", "speaker_b": "...",
        "session_1": [{"speaker": ..., "dia_id": ..., "text": ..., "img_url": ..., "blip_caption": ...}, ...],
        "session_1_date_time": "...",
        "session_2": [...], "session_2_date_time": "...",
        ...
      },
      "qa": [{"question": ..., "answer": ..., "category": ..., "evidence": [...]}, ...],
      ...  # observation / session_summary / event_summary fields also present, not
           # needed for the loader itself; downstream code can read the raw JSON
           # directly if it wants those.
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from gms.data.schema import Conversation, QAItem, Session, Turn

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")

# QAItem.category is kept as LoCoMo's raw int (see schema.py) rather than
# remapped at load time. This dict documents what those ints mean, for
# whoever needs to map them into this project's own task taxonomy.
#
# Not from official docs: the LoCoMo paper (Maharana et al., ACL 2024, sec.
# 4.1) names five categories - single-hop, multi-hop, temporal reasoning,
# open-domain knowledge, adversarial - but never ties them to the integer
# labels in the released JSON, and the released locomo10.json's numeric
# order does NOT match the paper's prose order. Derived instead by
# inspecting data/raw/locomo/locomo10.json directly against two signals per
# category: how many evidence turns/sessions each answer draws on, and the
# actual question/answer text.
#
#   category | n (this 10-conv file) | avg evidence turns | % multi-session evidence
#   1        | 282                   | 3.13                | 95.4%
#   2        | 321                   | 1.17                | 8.7%
#   3        | 96                    | 2.08                | 32.3%
#   4        | 841                   | 1.07                | 0.1%
#   5        | 446                   | 1.03                | 0.0%  (answer is always null)
#
# Confidence: 5 is certain (every answer is null - matches "adversarial,
# expected to be identified as unanswerable" exactly). 1 and 4 are strong
# (evidence-hop count is a direct, unambiguous signal). 2 is strong (near
# every question is literally "When did X happen?"). 3 is the softest -
# inferred from its speculative/hypothetical phrasing ("Would X be likely
# to...") plus a middling hop count.
LOCOMO_QA_CATEGORIES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal_reasoning",
    3: "open_domain_knowledge",
    4: "single_hop",
    5: "adversarial",
}


def load_locomo(path: str | Path) -> list[Conversation]:
    """Load the LoCoMo dataset from its raw JSON file into our shared schema.

    Raises FileNotFoundError with a pointer to the download script if the
    file isn't there yet, rather than a bare stack trace.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo data not found at {path}. Run `bash scripts/download_locomo.sh` first."
        )

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    conversations: list[Conversation] = []
    for sample in raw:
        conv_data = sample.get("conversation", {})

        session_nums = sorted(
            int(m.group(1))
            for key in conv_data
            if (m := _SESSION_KEY_RE.match(key))
        )

        sessions: list[Session] = []
        for num in session_nums:
            turns_raw = conv_data.get(f"session_{num}", [])
            date_time = conv_data.get(f"session_{num}_date_time", "")
            turns = [
                Turn(
                    dia_id=t.get("dia_id", ""),
                    speaker=t.get("speaker", ""),
                    text=t.get("text", ""),
                    img_url=t.get("img_url"),
                    blip_caption=t.get("blip_caption"),
                    session_date=date_time,
                )
                for t in turns_raw
            ]
            sessions.append(Session(session_id=f"session_{num}", date_time=date_time, turns=turns))

        qa_items = [
            QAItem(
                question=q.get("question", ""),
                answer=str(q.get("answer", "")),
                category=q.get("category"),
                evidence=q.get("evidence") or [],
                sample_id=sample.get("sample_id", ""),
            )
            for q in sample.get("qa", [])
        ]

        conversations.append(
            Conversation(
                sample_id=sample.get("sample_id", ""),
                speaker_a=conv_data.get("speaker_a", ""),
                speaker_b=conv_data.get("speaker_b", ""),
                sessions=sessions,
                qa_items=qa_items,
            )
        )

    return conversations
