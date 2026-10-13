"""Hybrid memory architecture: turn-embedding retrieval as the primary channel,
the entity graph used only to expand it.

See the ladder table in the README (`hybrid` / `vector_seeded_graph`).

**What the sweep over `vector_slots` established (finding #15).** This module was
written expecting L3's deficit to be its *retrieval unit* — ranking by rendered
triple similarity rather than by the turn itself. **That turned out to be wrong.**
At `vector_slots=0` every returned turn is ranked by PageRank over the same
triple-derived graph, and it beats both L1 (+0.012 recall) and L3 (+0.071). The
graph and the ranking are the same as L3's; only the seeding differs. So L3's
weakness was its *entry point* — matching the query against entities and triples
— and seeding the walk from the entities of vector-retrieved turns fixes it.

So this keeps L1's ranking for part of the budget and lets the graph fill the
rest. Two consequences worth stating:

- It cannot score far below L1, because L1's own top turns are included by
  construction. The interesting question is whether it scores *above* L1 — that
  is, whether graph expansion finds evidence turn-similarity misses.
- `vector_slots` spans the whole design space rather than picking a point.
  `vector_slots == top_k` is exactly L1; `vector_slots == 0` is the ladder's
  "vector finds seed entities, then graph traversal", since seeds always come
  from the vector channel. Sweeping it *is* the experiment.

**This is the most expensive rung**, not a cheaper one: construction pays L3's
extraction and triple embeddings *plus* L1's turn embeddings. A hybrid that only
matched L1's accuracy would therefore be strictly worse than L1 on R2's
cost-adjusted frontier.
"""

from __future__ import annotations

import time
from collections import defaultdict

import networkx as nx
import numpy as np

from gms.architectures.base import Retrieval, format_turn, turn_content
from gms.architectures.embeddings import Embedder
from gms.architectures.graph import GraphMemory
from gms.data.schema import Turn
from gms.llm import LLMClient

# How many of the top vector-ranked turns contribute their entities as PageRank
# seeds. Independent of how many budget slots the vector channel keeps, because
# seeding and answering are different jobs: a turn can be a good place to start
# walking without being worth returning.
_SEED_TURNS = 5


class HybridMemory(GraphMemory):
    """L3's graph plus L1's turn embeddings, with the vector channel leading.

    Subclasses `GraphMemory` rather than composing one so the graph, the
    inverted index and the relation-weighting machinery are shared outright —
    the hybrid *is* L3 with a second retrieval channel, and duplicating that
    state would be a way for the two to drift apart.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder | None = None,
        top_k: int = 5,
        vector_slots: int | None = None,
        seed_turns: int = _SEED_TURNS,
        **graph_kwargs,
    ) -> None:
        """`vector_slots` is how many of the `top_k` slots the vector channel
        keeps; the graph fills the remainder. `None` means half, rounded up, so
        a budget of 5 is 3 vector + 2 graph.
        """
        super().__init__(llm=llm, embedder=embedder, top_k=top_k, **graph_kwargs)
        self.vector_slots = (top_k + 1) // 2 if vector_slots is None else vector_slots
        self.seed_turns = seed_turns
        self._turn_list: list[Turn] = []
        self._turn_embeddings: list[np.ndarray] = []

    def add_turn(self, turn: Turn) -> None:
        # The graph half first, so extraction failures are logged exactly as L3
        # logs them and the two rungs stay comparable on failure rate.
        super().add_turn(turn)

        # Embedded on content, not the rendered line — same rule as L1, so the
        # vector channel here and L1 itself match on the same text.
        text = turn_content(turn)
        start = time.perf_counter()
        self._turn_list.append(turn)
        self._turn_embeddings.append(self.embedder.embed([text])[0])
        self.cost.construction_seconds += time.perf_counter() - start
        self.cost.construction_tokens += self.embedder.count_tokens(text)

    def _vector_ranking(self, query_vec: np.ndarray) -> list[int]:
        """Indices into `_turn_list`, best first."""
        if not self._turn_embeddings:
            return []
        scores = np.stack(self._turn_embeddings) @ query_vec
        return list(np.argsort(-scores))

    def _graph_scores(self, query: str, query_vec: np.ndarray, seed_turn_ids: list[str]):
        """PageRank turn scores, seeded from the entities of the vector hits.

        This is what makes it a *hybrid* rather than L3 with extra steps: L3
        seeds from entities named in the query and from query-to-triple linking,
        whereas here the walk starts wherever turn-level semantic similarity
        already points. When the vector channel yields no usable seed the walk
        falls back to L3's own seeding, so the graph half is never simply dead.
        """
        seeds: dict[str, float] = defaultdict(float)
        seed_turn_ids = set(seed_turn_ids)

        # Entities of the seed turns, weighted by specificity (1/|turns holding
        # the node|) for the same reason L3 weights its own seeds: a speaker
        # appearing in 43% of turns would otherwise absorb the walk.
        for node, turn_ids in self._turn_ids_by_node.items():
            overlap = len(turn_ids.intersection(seed_turn_ids))
            if overlap and turn_ids:
                seeds[node] += overlap / len(turn_ids)

        if not seeds:
            triple_scores = (
                np.stack(self._triple_embeddings) @ query_vec if self._triples else np.zeros(0)
            )
            seeds = self._seed_weights(query, query_vec, triple_scores)
        if not seeds:
            return {}

        total = sum(seeds.values())
        if total <= 0:
            return {}
        seeds = {node: weight / total for node, weight in seeds.items()}

        triple_scores = (
            np.stack(self._triple_embeddings) @ query_vec if self._triples else np.zeros(0)
        )
        weight_attr = self._apply_relation_weights(triple_scores)
        ranks = nx.pagerank(
            self._graph, alpha=self.alpha, personalization=seeds, weight=weight_attr
        )

        turn_scores: dict[str, float] = defaultdict(float)
        for node, rank in ranks.items():
            turn_ids = self._turn_ids_by_node[node]
            specificity = 1.0 / len(turn_ids) if turn_ids else 0.0
            for turn_id in turn_ids:
                turn_scores[turn_id] += rank * specificity
        return turn_scores

    def retrieve(self, query: str) -> Retrieval:
        self.cost.queries += 1
        if not self._turn_list:
            return Retrieval(context="")

        # Both deferred passes, in the same order as GraphMemory.retrieve. This
        # override is the only entry point the hybrid uses, so omitting either
        # here disables it silently for this rung alone — which is exactly what
        # happened: entity resolution produced a change of precisely 0.0000 with
        # a zero-width interval, because it was never running.
        self._ensure_entity_resolution()
        self._ensure_synonym_edges()
        start = time.perf_counter()

        query_vec = self.embedder.embed_query(query)
        ranking = self._vector_ranking(query_vec)

        kept = [self._turn_list[i] for i in ranking[: self.vector_slots]]
        chosen_ids = {t.dia_id for t in kept}

        remaining = self.top_k - len(kept)
        if remaining > 0 and self._graph.number_of_nodes():
            seed_turn_ids = [self._turn_list[i].dia_id for i in ranking[: self.seed_turns]]
            turn_scores = self._graph_scores(query, query_vec, seed_turn_ids)
            # Turns the vector channel already returned are excluded rather than
            # re-ranked: the graph's job here is to add what similarity missed,
            # and letting it re-surface the same turns would silently shrink the
            # effective budget and make the hybrid look like L1 with noise.
            ordered = sorted(
                (kv for kv in turn_scores.items() if kv[0] not in chosen_ids),
                key=lambda kv: -kv[1],
            )
            for turn_id, _ in ordered[:remaining]:
                kept.append(self._turns[turn_id])
                chosen_ids.add(turn_id)

        selected = sorted(kept, key=lambda t: self._turn_order[t.dia_id])
        retrieval = Retrieval(
            context="\n".join(format_turn(t) for t in selected),
            source_turn_ids=[t.dia_id for t in selected],
        )

        self.cost.retrieval_seconds += time.perf_counter() - start
        self.cost.retrieval_tokens += self.embedder.count_tokens(query)
        return retrieval
