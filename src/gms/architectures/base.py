"""Common interface every memory architecture (L0-L3', hybrid) implements.

L0 (`flat.py`) and L1 (`vector.py`) implement it today; L2/L3/L3'/hybrid
follow the same contract. The eval harness (`gms.eval.harness`) drives every
architecture through these same two calls, which is what makes the comparison
apples-to-apples rather than a per-architecture bespoke run.

Timing convention: `add_turn` and `retrieve` each time their *whole*
operation into `CostLog`, not just the model call inside it — for the graph
rungs the surrounding index/traversal work is a real part of the cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from gms.data.schema import Turn


def turn_content(turn: Turn) -> str:
    """Speaker and words only — what a turn *says*.

    Used wherever a turn is matched or processed rather than shown: embeddings
    and extraction prompts. Deliberately excludes the timestamp, which is
    identical for every turn in a session and would dilute the semantic signal
    without helping, since LoCoMo's temporal questions ("When did X happen?")
    contain no date to match against.
    """
    return f"{turn.speaker}: {turn.text}"


def format_turn(turn: Turn) -> str:
    """Render one turn as a line of context for the generation LLM.

    Carries the session timestamp, because answers to temporal questions are
    the session date combined with a relative phrase in the turn — "yesterday"
    is worthless without knowing when it was said. Turns from sources that
    carry no timestamp render unchanged.

    Shared so that architectures returning raw turns present them identically —
    a rendering difference would otherwise show up as an architecture effect.
    """
    content = turn_content(turn)
    return f"[{turn.session_date}] {content}" if turn.session_date else content


@dataclass
class Retrieval:
    """What an architecture hands back for one query.

    `context` is the text the generation LLM sees — raw turns for L0/L1, but
    distilled attributes or triples for the structured rungs, which is the
    whole point of climbing the ladder.

    `source_turn_ids` names the turns that context was derived from. Retrieval
    metrics score against these ids rather than parsing `context`, so evidence
    recall stays exact even when an architecture never emits a turn verbatim.
    An architecture that surfaces the same turn twice reports it once.
    """

    context: str
    source_turn_ids: list[str] = field(default_factory=list)


@dataclass
class CostLog:
    """Accumulated cost for one architecture instance, per design doc section 2.4."""

    construction_tokens: int = 0
    construction_seconds: float = 0.0
    retrieval_tokens: int = 0
    retrieval_seconds: float = 0.0
    maintenance_tokens: int = 0
    storage_entries: int = 0
    failures: int = 0
    queries: int = 0


class MemoryArchitecture(ABC):
    """Base class every memory architecture implementation subclasses."""

    def __init__(self) -> None:
        self.cost = CostLog()

    @abstractmethod
    def add_turn(self, turn: Turn) -> None:
        """Incorporate one new turn into memory, updating self.cost."""

    @abstractmethod
    def retrieve(self, query: str) -> Retrieval:
        """Return the context for `query`, plus the turns it came from."""
