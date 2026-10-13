from gms.architectures.flat import FlatRecencyMemory
from gms.data.schema import Turn


def make_turn(dia_id: str, speaker: str, text: str) -> Turn:
    return Turn(dia_id=dia_id, speaker=speaker, text=text)


def test_retrieve_returns_all_turns_within_window():
    mem = FlatRecencyMemory(window=5)
    mem.add_turn(make_turn("d1", "Alice", "hi"))
    mem.add_turn(make_turn("d2", "Bob", "hello"))

    result = mem.retrieve("anything")
    assert result.context == "Alice: hi\nBob: hello"
    assert result.source_turn_ids == ["d1", "d2"]


def test_retrieve_truncates_to_window_keeping_most_recent():
    mem = FlatRecencyMemory(window=2)
    mem.add_turn(make_turn("d1", "Alice", "first"))
    mem.add_turn(make_turn("d2", "Bob", "second"))
    mem.add_turn(make_turn("d3", "Alice", "third"))

    result = mem.retrieve("ignored")
    assert result.context == "Bob: second\nAlice: third"
    assert result.source_turn_ids == ["d2", "d3"]


def test_retrieve_ignores_the_query():
    mem = FlatRecencyMemory(window=10)
    mem.add_turn(make_turn("d1", "Alice", "constant"))

    assert mem.retrieve("query one").context == mem.retrieve("a different query").context


def test_empty_memory_retrieves_empty_string():
    mem = FlatRecencyMemory()
    result = mem.retrieve("anything")
    assert result.context == ""
    assert result.source_turn_ids == []


def test_cost_tracking():
    mem = FlatRecencyMemory(window=10)
    mem.add_turn(make_turn("d1", "Alice", "hi"))
    mem.add_turn(make_turn("d2", "Bob", "hello"))
    mem.retrieve("q1")
    mem.retrieve("q2")

    assert mem.cost.storage_entries == 2
    assert mem.cost.queries == 2
    # L0 makes no LLM calls of its own.
    assert mem.cost.construction_tokens == 0
    assert mem.cost.retrieval_tokens == 0
