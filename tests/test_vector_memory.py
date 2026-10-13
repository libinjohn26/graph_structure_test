import pytest
from fakes import StableHashEmbedder

from gms.architectures.vector import VectorMemory
from gms.data.schema import Turn


def make_turn(dia_id: str, speaker: str, text: str) -> Turn:
    return Turn(dia_id=dia_id, speaker=speaker, text=text)


@pytest.fixture
def mem():
    return VectorMemory(embedder=StableHashEmbedder(), top_k=1)


def test_retrieve_returns_most_similar_turn(mem):
    mem.add_turn(make_turn("d1", "Alice", "I have a pet dog named Rex"))
    mem.add_turn(make_turn("d2", "Bob", "My favorite programming language is Python"))
    mem.add_turn(make_turn("d3", "Alice", "The weather today is sunny and warm"))

    result = mem.retrieve("dog Rex")
    assert result.context == "Alice: I have a pet dog named Rex"
    assert result.source_turn_ids == ["d1"]


def test_retrieve_preserves_chronological_order_among_selected(mem):
    mem.top_k = 2
    mem.add_turn(make_turn("d1", "Alice", "dog Rex pet"))
    mem.add_turn(make_turn("d2", "Bob", "completely unrelated programming"))
    mem.add_turn(make_turn("d3", "Alice", "dog Rex walk"))

    result = mem.retrieve("dog Rex")

    assert result.context == "Alice: dog Rex pet\nAlice: dog Rex walk"
    assert result.source_turn_ids == ["d1", "d3"]


def test_empty_memory_retrieves_empty_string(mem):
    result = mem.retrieve("anything")
    assert result.context == ""
    assert result.source_turn_ids == []
    assert mem.cost.queries == 1


def test_cost_tracking(mem):
    mem.add_turn(make_turn("d1", "Alice", "dog Rex pet"))
    mem.add_turn(make_turn("d2", "Bob", "another turn here"))
    mem.retrieve("dog Rex")

    assert mem.cost.storage_entries == 2
    assert mem.cost.queries == 1
    assert mem.cost.construction_tokens > 0
    assert mem.cost.retrieval_tokens > 0
    assert mem.cost.construction_seconds >= 0
    assert mem.cost.retrieval_seconds >= 0
