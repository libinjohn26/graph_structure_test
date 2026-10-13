"""Common data schema shared by every dataset loader.

Every dataset (LoCoMo, LongMemEval, our own synthetic generators) gets adapted
into these dataclasses. Memory architectures and the eval harness only ever
see this schema, never a dataset's native format — that's what lets us swap
in a new benchmark without touching anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single utterance within a session."""

    dia_id: str
    speaker: str
    text: str
    img_url: str | None = None
    blip_caption: str | None = None
    # Denormalised from the parent Session so a turn carries its own timestamp.
    # Retrieval hands architectures individual turns, not sessions, and without
    # this a relative reference like "yesterday" reaches the generation LLM with
    # nothing to anchor it — which made every temporal question unanswerable.
    session_date: str = ""


@dataclass
class Session:
    """An ordered block of turns sharing one timestamp (one "sitting")."""

    session_id: str
    date_time: str
    turns: list[Turn] = field(default_factory=list)


@dataclass
class QAItem:
    """One evaluation question tied back to a conversation.

    `category` is kept as whatever raw label the source dataset uses (an int
    for LoCoMo, a string for others) rather than remapped to our own task
    taxonomy — that remapping happens explicitly in the eval harness once we
    build it, so we don't silently mislabel questions on a guess. For LoCoMo,
    what those raw ints mean is documented at
    `gms.data.locomo.LOCOMO_QA_CATEGORIES`.
    """

    question: str
    answer: str
    category: int | str | None
    evidence: list[str] = field(default_factory=list)
    sample_id: str = ""


@dataclass
class Conversation:
    """One full multi-session conversation plus its QA annotations."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session] = field(default_factory=list)
    qa_items: list[QAItem] = field(default_factory=list)

    def num_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    def num_sessions(self) -> int:
        return len(self.sessions)
