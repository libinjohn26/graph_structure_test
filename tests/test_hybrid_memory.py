"""Tests for the hybrid rung (L1's turn embeddings + L3's graph expansion).

The two boundary cases matter most: `vector_slots == top_k` must reproduce L1
exactly, and `vector_slots == 0` must be a pure vector-seeded graph. If either
drifts, the sweep over `vector_slots` stops being an interpretable experiment
and becomes an arbitrary third architecture.
"""

import pytest

from gms.architectures.hybrid import HybridMemory
from gms.architectures.vector import VectorMemory
from gms.data.schema import Turn

from tests.fakes import ScriptedLLM, StableHashEmbedder


def triples_for(text: str) -> str:
    """One triple per turn, keyed off a marker word so tests can steer the graph."""
    def wrap(*triples):
        inner = ", ".join(
            f'{{"subject": "{s}", "relation": "{r}", "object": "{o}"}}' for s, r, o in triples
        )
        return f'{{"triples": [{inner}]}}'

    if "rex" in text.lower():
        return wrap(("caroline", "owns", "rex"))
    if "sweden" in text.lower():
        return wrap(("caroline", "moved from", "sweden"))
    if "vet" in text.lower():
        return wrap(("rex", "sees", "vet"))
    return wrap()


class SteeredLLM(ScriptedLLM):
    """Returns a triple chosen by the turn's content rather than a fixed script,
    so a test can steer which entities the graph ends up holding."""

    def complete(self, prompt, *, system=None):
        from gms.llm import LLMResponse

        self.prompts.append(prompt)
        text = triples_for(prompt)
        return LLMResponse(text=text, prompt_tokens=len(prompt.split()),
                           completion_tokens=len(text.split()))


def make_turns():
    return [
        Turn(dia_id="t1", speaker="Caroline", text="I adopted a dog called Rex."),
        Turn(dia_id="t2", speaker="Caroline", text="I moved from Sweden four years ago."),
        Turn(dia_id="t3", speaker="Mel", text="Does he see a vet regularly?"),
        Turn(dia_id="t4", speaker="Caroline", text="We talked about the weather."),
    ]


def build(**kwargs):
    memory = HybridMemory(llm=SteeredLLM(), embedder=StableHashEmbedder(), **kwargs)
    for turn in make_turns():
        memory.add_turn(turn)
    return memory


def test_full_vector_slots_reproduces_l1_exactly():
    """The design's left endpoint. If this drifts, `vector_slots` is not a dial
    between L1 and a graph-led architecture and the sweep means nothing."""
    query = "What dog does Caroline own?"

    hybrid = build(top_k=3, vector_slots=3)
    l1 = VectorMemory(embedder=StableHashEmbedder(), top_k=3)
    for turn in make_turns():
        l1.add_turn(turn)

    assert hybrid.retrieve(query).source_turn_ids == l1.retrieve(query).source_turn_ids


def test_zero_vector_slots_still_retrieves_via_the_graph():
    """The right endpoint: the ladder's 'vector finds seed entities, then graph
    traversal'. No vector slots must not mean no results."""
    memory = build(top_k=3, vector_slots=0)
    result = memory.retrieve("What dog does Caroline own?")
    assert result.source_turn_ids, "graph channel returned nothing with vector_slots=0"
    assert len(result.source_turn_ids) <= 3


def test_graph_slots_do_not_duplicate_vector_hits():
    """The graph's job is to add what similarity missed. Re-surfacing a turn the
    vector channel already returned would silently shrink the effective budget
    while looking like a full result."""
    memory = build(top_k=4, vector_slots=2)
    ids = memory.retrieve("What dog does Caroline own?").source_turn_ids
    assert len(ids) == len(set(ids))


def test_respects_the_budget():
    for slots in (0, 1, 2, 3):
        memory = build(top_k=3, vector_slots=slots)
        ids = memory.retrieve("Where did Caroline move from?").source_turn_ids
        assert len(ids) <= 3, f"budget exceeded with vector_slots={slots}"


def test_returns_turns_in_chronological_order():
    memory = build(top_k=4, vector_slots=2)
    ids = memory.retrieve("What dog does Caroline own?").source_turn_ids
    order = {t.dia_id: i for i, t in enumerate(make_turns())}
    assert ids == sorted(ids, key=lambda i: order[i])


def test_source_ids_match_the_rendered_context():
    """The harness scores ids, and renders turns with format_turn to match them.
    A private rendering here would score every retrieval as a miss."""
    from gms.architectures.base import format_turn

    memory = build(top_k=3, vector_slots=2)
    result = memory.retrieve("What dog does Caroline own?")
    by_id = {t.dia_id: t for t in make_turns()}
    expected = "\n".join(format_turn(by_id[i]) for i in result.source_turn_ids)
    assert result.context == expected


def test_construction_cost_exceeds_the_graph_rung_alone():
    """The hybrid pays L3's extraction *and* L1's turn embeddings. If it ever
    reported less than L3, the cost-adjusted comparison in R2 would be wrong in
    the hybrid's favour."""
    from gms.architectures.graph import GraphMemory

    hybrid = build(top_k=3)
    graph = GraphMemory(llm=SteeredLLM(), embedder=StableHashEmbedder(), top_k=3)
    for turn in make_turns():
        graph.add_turn(turn)

    assert hybrid.cost.construction_tokens > graph.cost.construction_tokens


def test_empty_memory_returns_empty_retrieval():
    memory = HybridMemory(llm=SteeredLLM(), embedder=StableHashEmbedder(), top_k=3)
    result = memory.retrieve("anything")
    assert result.context == ""
    assert result.source_turn_ids == []


def test_default_vector_slots_is_half_the_budget_rounded_up():
    memory = HybridMemory(llm=SteeredLLM(), embedder=StableHashEmbedder(), top_k=5)
    assert memory.vector_slots == 3
    assert HybridMemory(llm=SteeredLLM(), embedder=StableHashEmbedder(), top_k=4).vector_slots == 2


def test_queries_are_counted_even_when_memory_is_empty():
    memory = HybridMemory(llm=SteeredLLM(), embedder=StableHashEmbedder(), top_k=3)
    memory.retrieve("anything")
    assert memory.cost.queries == 1
