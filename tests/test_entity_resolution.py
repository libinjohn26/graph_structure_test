"""Tests for entity resolution inside GraphMemory.

Resolution is destructive — it fuses two nodes permanently — so the failure mode
is not a crash but a quietly wrong graph: stranded edges, a stale inverted index,
or seeds pointing at nodes that no longer exist. Each of those would show up only
as slightly worse retrieval, which is indistinguishable from noise. Hence the
structural assertions here.
"""

import pytest

from gms.architectures.graph import GraphMemory, build_resolution_map, normalise_entity
from gms.data.schema import Turn

from tests.fakes import ScriptedLLM, StableHashEmbedder


def triple_json(*triples):
    inner = ", ".join(
        f'{{"subject": "{s}", "relation": "{r}", "object": "{o}"}}' for s, r, o in triples
    )
    return f'{{"triples": [{inner}]}}'


def build(resolve: bool, responses):
    memory = GraphMemory(
        llm=ScriptedLLM(responses),
        embedder=StableHashEmbedder(),
        top_k=3,
        resolve_entities=resolve,
    )
    for index, _ in enumerate(responses):
        memory.add_turn(Turn(dia_id=f"t{index}", speaker="S", text=f"turn {index}"))
    return memory


FRAGMENTED = [
    triple_json(("Tim Burton", "directed", "Batman")),
    triple_json(("Burton", "was born in", "Burbank")),
    triple_json(("Batman", "features", "Gotham")),
]


# --- the merge rule itself -------------------------------------------------

def test_single_token_name_folds_into_the_fuller_form():
    assert build_resolution_map({"burton", "tim burton"})["burton"] == "tim burton"


def test_merge_requires_a_whole_token_not_a_substring():
    """`art` must not be absorbed by `bart simpson` — substring matching here
    would fuse unrelated entities wholesale."""
    mapping = build_resolution_map({"art", "bart simpson"})
    assert mapping["art"] == "art"


def test_very_short_names_are_left_alone():
    assert build_resolution_map({"rex", "t rex"})["rex"] == "rex"


def test_multi_token_names_are_never_absorbed():
    mapping = build_resolution_map({"tim burton", "tim burton jr"})
    assert mapping["tim burton"] == "tim burton"


def test_normalisation_strips_articles_and_punctuation():
    assert normalise_entity("The Batman,") == "batman"
    assert normalise_entity("a dog") == "dog"
    # Never returns empty, even when every token is stripped.
    assert normalise_entity("the") == "the"


# --- what resolution does to the graph -------------------------------------

def test_resolution_is_off_by_default():
    memory = build(resolve=False, responses=FRAGMENTED)
    memory.graph_stats()
    assert "burton" in memory._graph and "tim burton" in memory._graph


def test_merging_removes_the_absorbed_node():
    memory = build(resolve=True, responses=FRAGMENTED)
    memory.graph_stats()
    assert "burton" not in memory._graph
    assert "tim burton" in memory._graph


def test_edges_of_the_absorbed_node_move_to_the_canonical_one():
    """A stranded edge is the quiet failure: connectivity would look improved in
    the node count while the path is actually gone."""
    memory = build(resolve=True, responses=FRAGMENTED)
    memory.graph_stats()
    assert memory._graph.has_edge("tim burton", "burbank")
    assert memory._graph.has_edge("tim burton", "batman")


def test_resolution_raises_connectivity():
    """The whole point: `burton` and `tim burton` sat in separate components."""
    import networkx as nx

    before = build(resolve=False, responses=FRAGMENTED)
    before.graph_stats()
    after = build(resolve=True, responses=FRAGMENTED)
    after.graph_stats()

    assert nx.number_connected_components(after._graph) < nx.number_connected_components(
        before._graph
    )


def test_inverted_index_merges_so_no_turn_becomes_unreachable():
    """Turn scoring sums rank over `_turn_ids_by_node`. If the absorbed node's
    turns are dropped, those turns can never be retrieved again."""
    memory = build(resolve=True, responses=FRAGMENTED)
    memory.graph_stats()
    assert memory._turn_ids_by_node["tim burton"] == {"t0", "t1"}
    assert "burton" not in memory._turn_ids_by_node


def test_specificity_falls_after_merging():
    """The cost side of the trade, asserted rather than assumed.

    Seeds are weighted by 1/|turns mentioning the node|, so fusing two names
    makes the result *less* discriminative. This is the effect that could make
    resolution a net loss, and it should be visible in the index.
    """
    unresolved = build(resolve=False, responses=FRAGMENTED)
    unresolved.graph_stats()
    resolved = build(resolve=True, responses=FRAGMENTED)
    resolved.graph_stats()

    before = 1.0 / len(unresolved._turn_ids_by_node["tim burton"])
    after = 1.0 / len(resolved._turn_ids_by_node["tim burton"])
    assert after < before


def test_triple_seeds_are_mapped_through_the_resolution_table():
    """Triples keep their original surface strings. Without mapping, a seed taken
    from the `Burton` triple names a node that no longer exists and is dropped by
    `_seed_weights` — losing seeds on exactly the queries resolution should help.
    """
    memory = build(resolve=True, responses=FRAGMENTED)
    memory.graph_stats()
    assert memory._canon("burton") == "tim burton"

    import numpy as np

    scores = np.ones(len(memory._triples))
    linked = memory._linked_triple_nodes(scores)
    assert all(node in memory._graph for node in linked), (
        "a linked triple node is not in the graph; seeds will be silently dropped"
    )


def test_edge_triples_are_rekeyed_so_relation_weighting_still_applies():
    """`_apply_relation_weights` skips keys whose edge is absent, so stale keys
    would silently disable relation weighting for merged edges."""
    memory = build(resolve=True, responses=FRAGMENTED)
    memory.graph_stats()
    for left, right in memory._edge_triples:
        assert memory._graph.has_edge(left, right), f"stale edge key {(left, right)}"


def test_retrieval_still_works_after_merging():
    memory = build(resolve=True, responses=FRAGMENTED)
    result = memory.retrieve("What did Tim Burton direct?")
    assert result.source_turn_ids
    assert len(result.source_turn_ids) <= 3


def test_resolution_runs_once_not_per_query():
    memory = build(resolve=True, responses=FRAGMENTED)
    memory.retrieve("anything")
    nodes_after_first = memory._graph.number_of_nodes()
    memory.retrieve("anything else")
    assert memory._graph.number_of_nodes() == nodes_after_first


def test_adding_a_turn_after_a_query_re_resolves():
    """Ingestion can resume after a query; the merge must not be left stale."""
    memory = build(resolve=True, responses=FRAGMENTED)
    memory.retrieve("warm up")
    memory.llm = ScriptedLLM([triple_json(("Burbank", "is in", "California"))])
    memory.add_turn(Turn(dia_id="t9", speaker="S", text="later"))
    assert memory._resolution_stale
    memory.retrieve("again")
    assert not memory._resolution_stale


def test_normalisation_is_applied_only_when_resolution_is_on():
    """Node ids must stay byte-identical with the flag off.

    Normalising inside `_add_triple` would have been simpler but would silently
    change the graph for every run that predates this feature, invalidating the
    published results.
    """
    responses = [triple_json(("The Batman", "appears in", "Gotham"))]
    off = build(resolve=False, responses=responses)
    off.graph_stats()
    assert "the batman" in off._graph

    on = build(resolve=True, responses=responses)
    on.graph_stats()
    assert "batman" in on._graph
    assert "the batman" not in on._graph


def test_a_renamed_node_keeps_its_embedding():
    """`_seed_nodes` falls back to nearest-node-by-embedding. A rename that drops
    the embedding removes that node from the fallback pool silently."""
    on = build(resolve=True, responses=[triple_json(("The Batman", "appears in", "Gotham"))])
    on.graph_stats()
    assert "batman" in on._node_embeddings
    assert "the batman" not in on._node_embeddings


def test_architecture_and_analysis_script_agree_on_the_merge():
    """Finding #12's reachability numbers come from the analysis script; this
    rung must build the same graph or the measurement describes nothing real."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "analyse_graphs", Path(__file__).resolve().parent.parent / "scripts" / "analyse_graphs.py"
    )
    analyse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyse)

    names = {"the batman", "tim burton", "burton", "gotham city", "gotham"}
    normalised = {analyse._normalise(n) for n in names}
    assert normalised == {normalise_entity(n) for n in names}
    assert analyse.build_resolution_map(normalised) == build_resolution_map(normalised)


def test_hybrid_runs_the_resolution_pass_too():
    """`HybridMemory` overrides `retrieve`, so it must invoke both deferred
    passes itself. It originally called only `_ensure_synonym_edges`, which
    disabled resolution for that rung alone — visible only as a measured effect
    of exactly 0.0000 with a zero-width confidence interval.
    """
    from gms.architectures.hybrid import HybridMemory

    memory = HybridMemory(
        llm=ScriptedLLM(FRAGMENTED),
        embedder=StableHashEmbedder(),
        top_k=3,
        resolve_entities=True,
    )
    for index in range(len(FRAGMENTED)):
        memory.add_turn(Turn(dia_id=f"t{index}", speaker="S", text=f"turn {index}"))

    assert memory._resolution_stale
    memory.retrieve("What did Tim Burton direct?")
    assert not memory._resolution_stale, "hybrid never ran entity resolution"
    assert "burton" not in memory._graph
