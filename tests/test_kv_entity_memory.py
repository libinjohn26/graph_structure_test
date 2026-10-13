import json

import pytest
from fakes import ScriptedLLM, StableHashEmbedder

from gms.architectures.kv_entity import KVEntityMemory, _parse_entities
from gms.data.schema import Turn


def extraction(*entities: tuple[str, list[str]]) -> str:
    return json.dumps({"entities": [{"name": n, "attributes": a} for n, a in entities]})


def make_turn(dia_id: str, speaker: str, text: str) -> Turn:
    return Turn(dia_id=dia_id, speaker=speaker, text=text)


def build(responses: list[str], top_k: int = 5) -> KVEntityMemory:
    return KVEntityMemory(
        llm=ScriptedLLM(responses), embedder=StableHashEmbedder(), top_k=top_k
    )


# --- extraction parsing ------------------------------------------------------


def test_parses_well_formed_json():
    assert _parse_entities(extraction(("Rex", ["is a dog"]))) == [("Rex", ["is a dog"])]


def test_parses_json_wrapped_in_markdown_fences():
    fenced = "```json\n" + extraction(("Rex", ["is a dog"])) + "\n```"
    assert _parse_entities(fenced) == [("Rex", ["is a dog"])]


@pytest.mark.parametrize(
    "raw",
    ["not json at all", "", "[1, 2, 3]", '{"entities": "not a list"}', '{"no_entities_key": 1}'],
)
def test_unusable_responses_are_distinguishable_failures(raw):
    assert _parse_entities(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"entities": []}',
        '{"entities": [{"name": "", "attributes": ["x"]}]}',
        '{"entities": [{"name": "Rex", "attributes": []}]}',
    ],
)
def test_valid_but_empty_extraction_is_not_a_failure(raw):
    # A greeting states no facts; the model saying so is correct behaviour.
    assert _parse_entities(raw) == []


# --- construction ------------------------------------------------------------


def test_stores_one_entry_per_extracted_attribute():
    mem = build([extraction(("Rex", ["is a dog", "is brown"]), ("Alice", ["owns Rex"]))])
    mem.add_turn(make_turn("D1:1", "Alice", "My brown dog Rex"))

    assert mem.cost.storage_entries == 3
    assert mem.cost.construction_tokens > 0


def test_malformed_extraction_counts_as_a_failure_without_crashing():
    mem = build(["total garbage, not json"])
    mem.add_turn(make_turn("D1:1", "Alice", "My dog Rex"))

    assert mem.cost.failures == 1
    assert mem.cost.storage_entries == 0


def test_factless_turn_stores_nothing_and_is_not_a_failure():
    # Roughly a fifth of real dialogue turns are greetings or backchannel;
    # counting them as failures would make the failure rate meaningless.
    mem = build(['{"entities": []}'])
    mem.add_turn(make_turn("D1:1", "Caroline", "Hey Mel! Good to see you!"))

    assert mem.cost.failures == 0
    assert mem.cost.storage_entries == 0


def test_extraction_prompt_forbids_relationships():
    # L2's defining constraint vs L3: attributes only, no edges.
    mem = build([extraction(("Rex", ["is a dog"]))])
    mem.add_turn(make_turn("D1:1", "Alice", "My dog Rex"))

    assert "relationships" in mem.llm.systems[0]


# --- retrieval ---------------------------------------------------------------


def test_retrieves_attributes_not_raw_turns():
    mem = build([extraction(("Rex", ["is a brown dog"]))])
    mem.add_turn(make_turn("D1:1", "Alice", "some quite differently worded utterance"))

    result = mem.retrieve("Rex")

    # The context is the distilled attribute; the turn text never appears.
    assert result.context == "Rex: is a brown dog"
    assert "differently worded" not in result.context
    # Provenance still points back at the source turn, which is what the
    # harness scores against.
    assert result.source_turn_ids == ["D1:1"]


def test_entity_name_match_filters_candidates():
    mem = build(
        [
            extraction(("Rex", ["is a dog"])),
            extraction(("Paris", ["is a city"])),
        ],
        top_k=1,
    )
    mem.add_turn(make_turn("D1:1", "Alice", "about the dog"))
    mem.add_turn(make_turn("D1:2", "Alice", "about the city"))

    assert mem.retrieve("Tell me about Rex").source_turn_ids == ["D1:1"]
    assert mem.retrieve("Tell me about Paris").source_turn_ids == ["D1:2"]


def test_entity_match_uses_word_boundaries():
    mem = build([extraction(("Mel", ["likes pottery"]))], top_k=1)
    mem.add_turn(make_turn("D1:1", "Alice", "about Mel"))

    # "Melanie" contains "Mel" but must not match it as an entity.
    assert mem._entity_matches("what does Melanie like") == []
    assert mem._entity_matches("what does Mel like") == [0]


def test_vector_fallback_when_query_names_no_known_entity():
    mem = build(
        [
            extraction(("Rex", ["enjoys long walks outdoors"])),
            extraction(("Paris", ["has excellent pastry shops"])),
        ],
        top_k=1,
    )
    mem.add_turn(make_turn("D1:1", "Alice", "turn one"))
    mem.add_turn(make_turn("D1:2", "Alice", "turn two"))

    # No entity named, so every attribute is a candidate and ranking decides.
    result = mem.retrieve("excellent pastry shops")

    assert result.source_turn_ids == ["D1:2"]


def test_context_is_chronological_not_rank_ordered():
    # Matches L0/L1/L3 so ordering never becomes an architecture variable;
    # relative time references only resolve against surrounding turns.
    mem = build(
        [
            extraction(("Rex", ["is a dog"])),
            extraction(("Paris", ["has pastry shops"])),
        ],
        top_k=2,
    )
    mem.add_turn(make_turn("D1:1", "Alice", "first"))
    mem.add_turn(make_turn("D1:2", "Bob", "second"))

    # Query matches the second turn's attribute more strongly, but the first
    # turn must still be rendered first.
    result = mem.retrieve("pastry shops")

    assert result.source_turn_ids == ["D1:1", "D1:2"]


def test_embedding_tokens_count_toward_construction_cost():
    # L1 bills embedding tokens, so L2 must too, or the expensive rung looks
    # cheaper than it is on the cost-per-accuracy axis.
    class LoudEmbedder(StableHashEmbedder):
        def count_tokens(self, text: str) -> int:
            return 1000

    mem = KVEntityMemory(
        llm=ScriptedLLM([extraction(("Rex", ["is a dog"]))]),
        embedder=LoudEmbedder(),
        top_k=5,
    )
    mem.add_turn(make_turn("D1:1", "Alice", "My dog Rex"))

    assert mem.cost.construction_tokens >= 1000


def test_attributes_sharing_a_turn_report_that_turn_once():
    mem = build([extraction(("Rex", ["is a dog", "is brown", "is friendly"]))], top_k=3)
    mem.add_turn(make_turn("D1:1", "Alice", "My friendly brown dog Rex"))

    result = mem.retrieve("Rex")

    assert len(result.context.split("\n")) == 3
    assert result.source_turn_ids == ["D1:1"]


def test_empty_memory_retrieves_nothing():
    mem = build([])
    result = mem.retrieve("anything")

    assert result.context == ""
    assert result.source_turn_ids == []
    assert mem.cost.queries == 1


def test_retrieval_cost_is_tracked():
    mem = build([extraction(("Rex", ["is a dog"]))])
    mem.add_turn(make_turn("D1:1", "Alice", "My dog Rex"))
    mem.retrieve("Rex")

    assert mem.cost.queries == 1
    assert mem.cost.retrieval_tokens > 0
    assert mem.cost.retrieval_seconds >= 0
