import json

import pytest
from fakes import ScriptedLLM, StableHashEmbedder

from gms.architectures.graph import GraphMemory, _parse_triples
from gms.data.schema import Turn


def extraction(*triples: tuple[str, str, str]) -> str:
    return json.dumps(
        {"triples": [{"subject": s, "relation": r, "object": o} for s, r, o in triples]}
    )


def make_turn(dia_id: str, speaker: str, text: str) -> Turn:
    return Turn(dia_id=dia_id, speaker=speaker, text=text)


def build(responses: list[str], top_k: int = 5) -> GraphMemory:
    return GraphMemory(llm=ScriptedLLM(responses), embedder=StableHashEmbedder(), top_k=top_k)


# --- parsing -----------------------------------------------------------------


def test_parses_well_formed_triples():
    parsed = _parse_triples(extraction(("Caroline", "owns", "Rex")))
    assert [(t.subject, t.relation, t.object) for t in parsed] == [("Caroline", "owns", "Rex")]


def test_parses_json_in_markdown_fences():
    fenced = "```json\n" + extraction(("A", "likes", "B")) + "\n```"
    assert len(_parse_triples(fenced)) == 1


@pytest.mark.parametrize(
    "raw", ["not json", "", "[1,2]", '{"triples": "nope"}', '{"other_key": []}']
)
def test_unusable_responses_are_distinguishable_failures(raw):
    assert _parse_triples(raw) is None


@pytest.mark.parametrize(
    "raw",
    ['{"triples": []}', '{"triples": [{"subject": "", "relation": "r", "object": "o"}]}'],
)
def test_valid_but_empty_is_not_a_failure(raw):
    # Conversational filler states no relationships; that is correct behaviour.
    assert _parse_triples(raw) == []


# --- construction ------------------------------------------------------------


def test_builds_a_graph_from_triples():
    mem = build([extraction(("Caroline", "owns", "Rex"), ("Rex", "is a", "dog"))])
    mem.add_turn(make_turn("D1:1", "Caroline", "My dog Rex"))

    stats = mem.graph_stats()
    assert stats["nodes"] == 3  # caroline, rex, dog
    assert stats["edges"] == 2
    assert mem.cost.construction_tokens > 0


def test_extraction_prompt_requires_relationships():
    # The defining contrast with L2, whose prompt forbids them.
    mem = build([extraction(("A", "knows", "B"))])
    mem.add_turn(make_turn("D1:1", "A", "text"))

    assert "relationships" in mem.llm.systems[0]
    assert "triples" in mem.llm.prompts[0]


def test_repeated_relations_strengthen_the_edge():
    mem = build([extraction(("A", "knows", "B")), extraction(("A", "trusts", "B"))])
    mem.add_turn(make_turn("D1:1", "A", "one"))
    mem.add_turn(make_turn("D1:2", "A", "two"))

    assert mem.graph_stats()["edges"] == 1
    assert mem._graph["a"]["b"]["weight"] == 2.0
    assert mem._graph["a"]["b"]["relations"] == {"knows", "trusts"}


def test_malformed_extraction_counts_as_a_failure():
    mem = build(["garbage"])
    mem.add_turn(make_turn("D1:1", "A", "text"))

    assert mem.cost.failures == 1
    assert mem.graph_stats()["nodes"] == 0


# --- retrieval ---------------------------------------------------------------


def test_returns_turns_not_triples():
    # HippoRAG2-style: the graph ranks, but the generator still sees the turn.
    mem = build([extraction(("Caroline", "owns", "Rex"))], top_k=1)
    mem.add_turn(make_turn("D1:1", "Caroline", "I adopted a dog last spring"))

    result = mem.retrieve("Caroline")

    assert result.context == "Caroline: I adopted a dog last spring"
    assert result.source_turn_ids == ["D1:1"]


def test_traversal_joins_facts_that_converge_on_a_shared_subject():
    """The canonical multi-hop shape, and the reason the graph is undirected.

    Both facts point *away* from Caroline. Answering from the Sweden end needs
    a step backward to Caroline before going forward to Rex — impossible while
    edges only run subject -> object.
    """
    mem = build(
        [
            extraction(("Caroline", "moved from", "Sweden")),
            extraction(("Caroline", "owns", "Rex")),
        ],
        top_k=2,
    )
    mem.add_turn(make_turn("D1:1", "Caroline", "about moving"))
    mem.add_turn(make_turn("D1:2", "Caroline", "about the dog"))

    # Seeding at Sweden must still reach the turn about Rex.
    ranks = mem.retrieve("Sweden")

    assert "D1:2" in ranks.source_turn_ids


def test_pagerank_reaches_turns_the_query_never_names():
    """The property that distinguishes a graph from an entity index.

    The query names only Caroline. Rex is never mentioned in it, and the turn
    about Rex shares no vocabulary with the query — but the edge
    Caroline -> Rex carries rank to it.
    """
    mem = build(
        [
            extraction(("Caroline", "owns", "Rex")),
            extraction(("Rex", "has", "fleas")),
        ],
        top_k=2,
    )
    mem.add_turn(make_turn("D1:1", "Caroline", "about the owner"))
    mem.add_turn(make_turn("D1:2", "Vet", "entirely different wording here"))

    result = mem.retrieve("Tell me about Caroline")

    assert "D1:2" in result.source_turn_ids


def test_seeds_match_on_word_boundaries():
    # Both names are in the graph, so a substring match would wrongly seed at
    # "mel" whenever the query says "Melanie".
    mem = build(
        [extraction(("Mel", "likes", "pottery"), ("Melanie", "likes", "running"))], top_k=1
    )
    mem.add_turn(make_turn("D1:1", "Mel", "text"))

    assert mem._seed_nodes("what does Mel like") == ["mel"]
    assert mem._seed_nodes("what does Melanie like") == ["melanie"]


def test_seed_weights_favour_specific_entities_over_hubs():
    """Node specificity, the fix that makes PPR discriminate at all.

    A speaker named in most turns carries almost no signal about which turn
    answers the question; an entity named in one turn carries a great deal.
    """
    mem = build(
        [
            extraction(("Caroline", "did", "research")),
            extraction(("Caroline", "likes", "coffee")),
            extraction(("Caroline", "visited", "Rome")),
        ],
        top_k=1,
    )
    for i in range(3):
        mem.add_turn(make_turn(f"D1:{i}", "Caroline", "text"))

    weights = mem._seed_weights("What did Caroline research?")

    # "caroline" is in all three turns, "research" in one.
    assert weights["research"] > weights["caroline"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_query_whose_key_term_is_a_relation_still_reaches_its_turn():
    """The HippoRAG1 -> HippoRAG2 gap, as a regression guard.

    "What did Caroline research?" carries its signal in the verb. Extraction
    puts that verb in the *relation* — (Caroline) -[is researching]-> (adoption
    agencies) — so an entity-only seeder never sees it and lands on unrelated
    turns that happen to contain a node literally named "research".
    """
    mem = build(
        [
            extraction(("Caroline", "is researching", "adoption agencies")),
            extraction(("Melanie", "owns", "research")),
        ],
        top_k=1,
    )
    mem.add_turn(make_turn("D1:1", "Caroline", "the turn that answers it"))
    mem.add_turn(make_turn("D1:2", "Melanie", "an unrelated turn"))

    seeds = mem._seed_weights("What did Caroline research?")

    # Linking on the rendered triple reaches the right entity, not just the
    # node that happens to share the query's wording.
    assert "adoption agencies" in seeds


def test_vector_fallback_when_query_names_no_entity():
    mem = build([extraction(("Paris", "is in", "France"))], top_k=1)
    mem.add_turn(make_turn("D1:1", "A", "text"))

    # No known entity named, so seeding falls back to embeddings rather than
    # returning nothing.
    assert mem.retrieve("somewhere entirely unrelated").source_turn_ids == ["D1:1"]


def test_chronological_order_among_selected():
    mem = build(
        [extraction(("A", "knows", "B")), extraction(("B", "knows", "C"))], top_k=2
    )
    mem.add_turn(make_turn("D1:1", "A", "first"))
    mem.add_turn(make_turn("D1:2", "B", "second"))

    result = mem.retrieve("A")

    assert result.source_turn_ids == ["D1:1", "D1:2"]


def test_empty_memory_retrieves_nothing():
    mem = build([])
    result = mem.retrieve("anything")

    assert result.context == ""
    assert result.source_turn_ids == []
    assert mem.cost.queries == 1


def test_embedding_tokens_count_toward_construction_cost():
    class LoudEmbedder(StableHashEmbedder):
        def count_tokens(self, text: str) -> int:
            return 1000

    from gms.architectures.graph import GraphMemory as _GraphMemory

    mem = _GraphMemory(
        llm=ScriptedLLM([extraction(("A", "knows", "B"))]),
        embedder=LoudEmbedder(),
        top_k=5,
    )
    mem.add_turn(make_turn("D1:1", "A", "text"))

    assert mem.cost.construction_tokens >= 1000


def test_query_is_embedded_once_per_retrieval():
    """Both seeding paths need the same vector; embedding twice pays twice
    while `retrieval_tokens` bills it once."""

    class CountingEmbedder(StableHashEmbedder):
        def __init__(self):
            super().__init__()
            self.query_embeds = 0

        def embed_query(self, text):
            self.query_embeds += 1
            return super().embed_query(text)

    embedder = CountingEmbedder()
    mem = GraphMemory(
        llm=ScriptedLLM([extraction(("A", "knows", "B"))]), embedder=embedder, top_k=1
    )
    mem.add_turn(make_turn("D1:1", "A", "text"))
    embedder.query_embeds = 0

    # A query naming no known entity takes the fallback path, which previously
    # embedded once for node seeding and again for triple linking.
    mem.retrieve("something the graph has never heard of")

    assert embedder.query_embeds == 1


class RelationAwareEmbedder(StableHashEmbedder):
    """Scores a triple by word overlap with the query, so a relation-matching
    edge is genuinely closer than an irrelevant one sharing the same subject."""

    def _vector(self, text: str):
        import numpy as np

        vocab = ["research", "adoption", "coffee", "likes", "caroline", "sweden", "moved"]
        vec = np.zeros(len(vocab) + 1, dtype=np.float32)
        lowered = text.lower()
        for i, word in enumerate(vocab):
            if word in lowered:
                vec[i] = 1.0
        vec[-1] = 0.1  # keeps unrelated text non-zero, like a real embedder's floor
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


def _hub_memory(power: float) -> GraphMemory:
    """One hub with three unrelated relations hanging off it — the structure that
    makes 75% of LoCoMo's multi-hop questions unreachable."""
    mem = GraphMemory(
        llm=ScriptedLLM(
            [
                extraction(("Caroline", "is researching", "adoption agencies")),
                extraction(("Caroline", "likes", "coffee")),
                extraction(("Caroline", "moved from", "Sweden")),
            ]
        ),
        embedder=RelationAwareEmbedder(),
        top_k=1,
        relation_weight_power=power,
    )
    for i, text in enumerate(["about research", "about coffee", "about sweden"]):
        mem.add_turn(make_turn(f"D1:{i}", "Caroline", text))
    return mem


def test_relation_weighting_is_off_by_default():
    mem = _hub_memory(power=0.0)
    mem.retrieve("What did Caroline research?")

    # Static counts: every edge off the hub looks identical.
    assert all("query_weight" not in d for _, _, d in mem._graph.edges(data=True))


def test_relation_weighting_prefers_the_edge_matching_the_query():
    mem = _hub_memory(power=4.0)
    mem.retrieve("What did Caroline research?")

    research_edge = mem._graph["caroline"]["adoption agencies"]["query_weight"]
    coffee_edge = mem._graph["caroline"]["coffee"]["query_weight"]

    # The hub node is uninformative; its edges are not.
    assert research_edge > coffee_edge


def test_an_edge_takes_the_best_of_its_relations_not_the_mean():
    mem = GraphMemory(
        llm=ScriptedLLM(
            [extraction(("Caroline", "is researching", "adoption agencies"), ("Caroline", "likes", "adoption agencies"))]
        ),
        embedder=RelationAwareEmbedder(),
        top_k=1,
        relation_weight_power=4.0,
    )
    mem.add_turn(make_turn("D1:1", "Caroline", "text"))
    mem.retrieve("What did Caroline research?")

    best_alone = mem._graph["caroline"]["adoption agencies"]["query_weight"]

    # One relevant relation must carry the edge even when an irrelevant one
    # shares the same pair; averaging would dilute it away.
    assert best_alone > 1e-6


def test_graph_stats_report_quality_metrics():
    mem = build([extraction(("A", "knows", "B"), ("B", "knows", "C"), ("A", "knows", "C"))])
    mem.add_turn(make_turn("D1:1", "A", "text"))

    stats = mem.graph_stats()
    assert stats["nodes"] == 3
    assert stats["edges"] == 3
    assert stats["avg_degree"] == pytest.approx(2.0)
    assert stats["clustering"] == pytest.approx(1.0)  # fully connected triangle
