"""L1 memory architecture: chunk = one turn, embed each turn as it arrives,
retrieve by top-k cosine similarity against the query.

See the ladder table in the README and `architectures: id L1` in
configs/default.yaml.
"""

from __future__ import annotations

import time

import numpy as np

from gms.architectures.base import MemoryArchitecture, Retrieval, format_turn, turn_content
from gms.architectures.embeddings import BGEEmbedder, Embedder
from gms.data.schema import Turn


class VectorMemory(MemoryArchitecture):
    """Embeds every turn independently on arrival; retrieval returns the
    `top_k` most similar turns, in chronological order."""

    def __init__(self, embedder: Embedder | None = None, top_k: int = 5) -> None:
        super().__init__()
        self.embedder = embedder or BGEEmbedder()
        self.top_k = top_k
        self._turns: list[Turn] = []
        self._embeddings: list[np.ndarray] = []

    def add_turn(self, turn: Turn) -> None:
        # Embedded on content, not the rendered line: the timestamp is identical
        # across a session and would only dilute the match.
        text = turn_content(turn)

        start = time.perf_counter()
        vec = self.embedder.embed([text])[0]
        self._turns.append(turn)
        self._embeddings.append(vec)
        self.cost.construction_seconds += time.perf_counter() - start

        self.cost.construction_tokens += self.embedder.count_tokens(text)
        self.cost.storage_entries += 1

    def retrieve(self, query: str) -> Retrieval:
        self.cost.queries += 1
        if not self._turns:
            return Retrieval(context="")

        start = time.perf_counter()

        query_vec = self.embedder.embed_query(query)
        matrix = np.stack(self._embeddings)
        scores = matrix @ query_vec  # rows are L2-normalized, so dot product == cosine similarity
        top_idx = np.argsort(-scores)[: self.top_k]
        selected = [self._turns[i] for i in sorted(top_idx)]  # chronological, not rank order
        retrieval = Retrieval(
            context="\n".join(format_turn(t) for t in selected),
            source_turn_ids=[t.dia_id for t in selected],
        )

        self.cost.retrieval_seconds += time.perf_counter() - start
        # Tokenizing for the cost log is measurement, not retrieval — kept out of the timer.
        self.cost.retrieval_tokens += self.embedder.count_tokens(query)
        return retrieval
