"""L0 memory architecture: no structure, retrieval is a fixed recency window.

The cheapest rung on the architecture ladder and the baseline every other
architecture is compared against — see the ladder table in the README and
`architectures: id L0` in configs/default.yaml.
"""

from __future__ import annotations

import time

from gms.architectures.base import MemoryArchitecture, Retrieval, format_turn
from gms.data.schema import Turn


class FlatRecencyMemory(MemoryArchitecture):
    """Stores every turn verbatim; retrieval ignores the query and returns the
    last `window` turns, formatted as "speaker: text" lines."""

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = window
        self._turns: list[Turn] = []

    def add_turn(self, turn: Turn) -> None:
        # Timed like every other rung even though the work is trivial: reporting
        # an exact 0.0s against real timings elsewhere overstates how cheap L0 is
        # on the cost-per-accuracy axis.
        start = time.perf_counter()
        self._turns.append(turn)
        self.cost.construction_seconds += time.perf_counter() - start
        self.cost.storage_entries += 1

    def retrieve(self, query: str) -> Retrieval:
        self.cost.queries += 1

        start = time.perf_counter()
        recent = self._turns[-self.window :]
        retrieval = Retrieval(
            context="\n".join(format_turn(t) for t in recent),
            source_turn_ids=[t.dia_id for t in recent],
        )
        self.cost.retrieval_seconds += time.perf_counter() - start
        return retrieval
