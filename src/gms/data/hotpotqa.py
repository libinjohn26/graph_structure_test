"""Loader for HotpotQA (Yang et al., EMNLP 2018), distractor setting.

Not conversational — this is the **entity-diverse control**. Every result so far
comes from LoCoMo, where two speakers appear in half the turns and the entity
joining a multi-hop question's evidence is almost always one of them. HotpotQA
is the opposite shape: facts meet at specific named entities, and no participant
dominates. Running the same architectures over both separates "graph memory does
not help conversational agents" from "this graph implementation is weak".

Mapping into the shared schema:

| HotpotQA | here |
|---|---|
| one sentence | `Turn`, `dia_id = "<title>::<sent_id>"` |
| paragraph title | `Turn.speaker` — the entity the paragraph is about |
| one paragraph | `Session` (no timestamps; `format_turn` omits the prefix) |
| a pool of questions' paragraphs | `Conversation` |
| `supporting_facts` | `QAItem.evidence`, same id format |
| `type` (bridge / comparison) | `QAItem.category`, kept raw like LoCoMo's ints |

**Questions are pooled rather than loaded one-per-conversation.** A single
HotpotQA item carries only 10 paragraphs (~50 sentences), so retrieving 5 from
50 would be a far easier task than LoCoMo's 5 from 419 and the comparison would
measure corpus size rather than architecture. Pooling `questions_per_pool`
items puts every question's evidence in a corpus of comparable size, with the
other questions' distractors as noise.
"""

from __future__ import annotations

from pathlib import Path

from gms.data.schema import Conversation, QAItem, Session, Turn

# Bridge questions hop through a connecting entity; comparison questions contrast
# two entities that need not be linked at all. Kept raw, like LoCoMo's ints.
HOTPOTQA_CATEGORIES: dict[str, str] = {
    "bridge": "bridge",
    "comparison": "comparison",
}


def _turn_id(title: str, sent_id: int) -> str:
    return f"{title}::{sent_id}"


def load_hotpotqa(
    path: str | Path,
    questions_per_pool: int = 20,
    limit: int | None = None,
) -> list[Conversation]:
    """Load HotpotQA dev (distractor) into pooled conversations.

    `limit` caps the number of *questions*, not pools — a full run over 7,405
    questions would be far more extraction than this project needs for a control.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"HotpotQA data not found at {path}. Run `bash scripts/download_hotpotqa.sh` first."
        )

    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if limit is not None:
        rows = rows[:limit]

    conversations: list[Conversation] = []
    for start in range(0, len(rows), questions_per_pool):
        pool = rows[start : start + questions_per_pool]

        sessions: list[Session] = []
        seen_titles: set[str] = set()
        qa_items: list[QAItem] = []

        for row in pool:
            context = row["context"]
            for title, sentences in zip(context["title"], context["sentences"]):
                # The same article often appears as a distractor for several
                # questions in a pool; keep one copy so ids stay unique.
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                sessions.append(
                    Session(
                        session_id=title,
                        date_time="",  # HotpotQA has no timestamps
                        turns=[
                            Turn(
                                dia_id=_turn_id(title, index),
                                speaker=title,
                                text=sentence.strip(),
                            )
                            for index, sentence in enumerate(sentences)
                        ],
                    )
                )

            facts = row["supporting_facts"]
            qa_items.append(
                QAItem(
                    question=row["question"],
                    answer=str(row["answer"]),
                    category=row["type"],
                    evidence=[
                        _turn_id(title, sent_id)
                        for title, sent_id in zip(facts["title"], facts["sent_id"])
                    ],
                    sample_id=row["id"],
                )
            )

        conversations.append(
            Conversation(
                sample_id=f"pool-{start // questions_per_pool:04d}",
                speaker_a="",  # no dialogue participants; speakers are article titles
                speaker_b="",
                sessions=sessions,
                qa_items=qa_items,
            )
        )

    return conversations
