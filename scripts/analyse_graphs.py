#!/usr/bin/env python
"""Graph statistics and gold-evidence reachability over every corpus (A2, A3).

Answers two questions the paper needs and the one-conversation probes could not:

- **A2** — is the extracted graph a tree, a near-tree, or something with real
  topology? Reported per corpus with components and independent cycles, not just
  degree, because "forest" is a falsifiable claim and edges > nodes - 1 falsifies
  it.
- **A3** — can traversal reach the evidence at all? For every multi-hop question,
  whether its gold turns' entities share a connected component, and how far apart
  they are. If most gold pairs are disconnected, no retrieval tuning can help and
  the diagnosis is direct rather than inferred.

Builds graphs from the **cached extraction responses only** — no embeddings, no
LLM calls — so this runs in minutes over both datasets.

Usage:
    python scripts/analyse_graphs.py --out data/processed/graph_analysis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import networkx as nx  # noqa: E402

from gms.architectures.base import turn_content  # noqa: E402
from gms.architectures.graph import (  # noqa: E402
    _EXTRACTION_PROMPT,
    _EXTRACTION_SYSTEM,
    _parse_triples,
)
from gms.data import load_hotpotqa, load_locomo  # noqa: E402
from gms.llm import VLLMClient  # noqa: E402
from gms.utils.env import load_env_file  # noqa: E402


def load_cache(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.exists():
        return entries
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                entries[record["key"]] = record["text"]
            except (json.JSONDecodeError, KeyError):
                continue
    return entries


def extraction_key(fingerprint: str, turn) -> str:
    prompt = _EXTRACTION_PROMPT.format(turn=turn_content(turn))
    payload = "\x00".join([fingerprint, _EXTRACTION_SYSTEM, prompt])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_ARTICLES = {"the", "a", "an"}


def _normalise(name: str) -> str:
    """Surface normalisation applied to every node before anything else."""
    tokens = [t.strip(".,!?'\"()") for t in name.lower().split()]
    tokens = [t for t in tokens if t and t not in _ARTICLES]
    return " ".join(tokens) or name.lower()


def build_resolution_map(names: set[str]) -> dict[str, str]:
    """Post-hoc entity resolution over already-extracted node names (D1).

    Deliberately conservative and lexical — no embeddings, no LLM, no
    re-extraction. It merges a node into another when its token sequence is a
    contiguous subsequence of the other's *and* the shorter name is not a single
    very common word, which is the `Tim Burton` / `Burton` case that finding #10
    identified as the main source of node fragmentation.

    Conservative on purpose: this is meant to put a *lower bound* on what entity
    resolution could recover. If even an aggressive merge left connectivity flat
    that would be ambiguous, but a conservative merge that *does* move it is
    evidence the bottleneck is real and addressable.
    """
    canonical = {name: name for name in names}
    # Longest first, so shorter fragments attach to the fullest form available.
    ordered = sorted(names, key=lambda n: (-len(n.split()), -len(n)))
    by_token: dict[str, list[str]] = defaultdict(list)
    for name in ordered:
        for token in set(name.split()):
            by_token[token].append(name)

    for name in ordered:
        tokens = name.split()
        if len(tokens) != 1 or len(name) < 4:
            continue
        # Candidates share this token, so we never scan the whole vocabulary.
        for longer in by_token.get(name, ()):
            longer_tokens = longer.split()
            if len(longer_tokens) < 2 or canonical[longer] != longer:
                continue
            if name in longer_tokens:
                canonical[name] = longer
                break
    return canonical


def build_graph(turns, cache, fingerprint, resolve: bool = False):
    """Rebuild one corpus's graph from cached extractions."""
    raw_triples = []
    misses = 0
    for turn in turns:
        raw = cache.get(extraction_key(fingerprint, turn))
        if raw is None:
            misses += 1
            continue
        for triple in _parse_triples(raw) or []:
            raw_triples.append(
                (turn.dia_id, _normalise(triple.subject), _normalise(triple.object))
            )

    if resolve:
        names = {n for _, s, o in raw_triples for n in (s, o)}
        canonical = build_resolution_map(names)
    else:
        canonical = {}

    graph = nx.Graph()
    nodes_of_turn: dict[str, set[str]] = defaultdict(set)
    for dia_id, subject, obj in raw_triples:
        subject = canonical.get(subject, subject)
        obj = canonical.get(obj, obj)
        if subject == obj:
            # Resolution collapsed a triple onto itself; it carries no edge.
            nodes_of_turn[dia_id].add(subject)
            continue
        for node in (subject, obj):
            nodes_of_turn[dia_id].add(node)
        if graph.has_edge(subject, obj):
            graph[subject][obj]["weight"] += 1.0
        else:
            graph.add_edge(subject, obj, weight=1.0)
    return graph, nodes_of_turn, misses


def graph_stats(graph: nx.Graph, turns, nodes_of_turn) -> dict:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    components = nx.number_connected_components(graph) if nodes else 0
    largest = max((len(c) for c in nx.connected_components(graph)), default=0)

    # A forest has edges == nodes - components. Anything above that is cycles.
    cycles = edges - nodes + components if nodes else 0

    df = Counter()
    for node_set in nodes_of_turn.values():
        for node in node_set:
            df[node] += 1
    top_share = (df.most_common(1)[0][1] / len(turns)) if df and turns else 0.0

    return {
        "nodes": nodes,
        "edges": edges,
        "mean_degree": (2 * edges / nodes) if nodes else 0.0,
        "clustering": nx.average_clustering(graph) if nodes > 1 else 0.0,
        "components": components,
        "largest_component_share": (largest / nodes) if nodes else 0.0,
        "independent_cycles": cycles,
        "is_forest": cycles == 0,
        "top_entity": df.most_common(1)[0][0] if df else None,
        "top_entity_share_of_turns": top_share,
    }


def reachability(graph, nodes_of_turn, qa_items, valid_ids, speakers: set[str]) -> dict:
    """A3: can traversal get from one gold turn's entities to another's?"""
    stripped = graph.copy()
    stripped.remove_nodes_from([s for s in speakers if s in stripped])

    def analyse(target_graph):
        same_component = 0
        disconnected = 0
        no_entities = 0
        path_lengths: list[int] = []

        component_of = {}
        for index, component in enumerate(nx.connected_components(target_graph)):
            for node in component:
                component_of[node] = index

        total = 0
        for qa in qa_items:
            gold = [e for e in qa.evidence if e in valid_ids]
            if len(gold) < 2:
                continue
            total += 1
            per_turn = [
                {n for n in nodes_of_turn.get(g, set()) if n in target_graph} for g in gold
            ]
            if any(not s for s in per_turn):
                no_entities += 1
                continue

            components = [{component_of[n] for n in s} for s in per_turn]
            if set.intersection(*components):
                same_component += 1
                # Over half the questions have more than two gold turns, so the
                # distance between the first pair is not the distance traversal
                # has to cover. Report the *hardest leg*: for every pair of gold
                # turns, the closest their entity sets come, maximised over
                # pairs. That is the hop budget needed to reach all of the gold
                # starting from any one piece of it.
                legs = []
                for i in range(len(per_turn)):
                    for j in range(i + 1, len(per_turn)):
                        best = None
                        for source in per_turn[i]:
                            for destination in per_turn[j]:
                                if component_of[source] != component_of[destination]:
                                    continue
                                length = nx.shortest_path_length(
                                    target_graph, source, destination
                                )
                                best = length if best is None else min(best, length)
                        if best is not None:
                            legs.append(best)
                if legs:
                    path_lengths.append(max(legs))
            else:
                disconnected += 1

        return {
            "questions": total,
            "gold_entities_share_a_component": same_component,
            "gold_turns_disconnected": disconnected,
            "gold_turn_had_no_entities": no_entities,
            "shortest_path_distribution": dict(sorted(Counter(path_lengths).items())),
            "median_shortest_path": statistics.median(path_lengths) if path_lengths else None,
        }

    return {"with_speakers": analyse(graph), "without_speakers": analyse(stripped)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(REPO_ROOT / "data" / "processed" / "llm_cache.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "graph_analysis.json"))
    parser.add_argument(
        "--resolve-entities",
        action="store_true",
        help="D1: merge co-referent nodes in the already-extracted triples before measuring, "
             "to bound what entity resolution could recover without re-extraction.",
    )
    args = parser.parse_args()

    load_env_file()
    fingerprint = VLLMClient.from_env().fingerprint
    cache = load_cache(Path(args.cache))
    print(f"cache entries: {len(cache):,}")

    report: dict = {"locomo": [], "hotpotqa": []}

    for conv in load_locomo(REPO_ROOT / "data" / "raw" / "locomo" / "locomo10.json"):
        turns = [t for s in conv.sessions for t in s.turns]
        valid = {t.dia_id for t in turns}
        graph, nodes_of_turn, misses = build_graph(
            turns, cache, fingerprint, resolve=args.resolve_entities
        )
        if misses:
            print(f"  [WARN] {conv.sample_id}: {misses} turns missing from cache")
        speakers = {conv.speaker_a.lower(), conv.speaker_b.lower()}
        stats = graph_stats(graph, turns, nodes_of_turn)
        stats["sample_id"] = conv.sample_id
        stats["turns"] = len(turns)
        stats["reachability_multi_hop"] = reachability(
            graph, nodes_of_turn, [q for q in conv.qa_items if q.category == 1], valid, speakers
        )
        report["locomo"].append(stats)
        print(f"  {conv.sample_id}: {stats['nodes']} nodes, {stats['edges']} edges, "
              f"{stats['components']} components, {stats['independent_cycles']} cycles")

    hotpot = load_hotpotqa(
        REPO_ROOT / "data" / "raw" / "hotpotqa" / "validation_distractor.parquet",
        questions_per_pool=20,
        limit=200,
    )
    for pool in hotpot:
        turns = [t for s in pool.sessions for t in s.turns]
        valid = {t.dia_id for t in turns}
        graph, nodes_of_turn, misses = build_graph(
            turns, cache, fingerprint, resolve=args.resolve_entities
        )
        if misses:
            print(f"  [WARN] {pool.sample_id}: {misses} turns missing from cache")
        stats = graph_stats(graph, turns, nodes_of_turn)
        stats["sample_id"] = pool.sample_id
        stats["turns"] = len(turns)
        stats["reachability_bridge"] = reachability(
            graph, nodes_of_turn, [q for q in pool.qa_items if q.category == "bridge"], valid, set()
        )
        report["hotpotqa"].append(stats)
        print(f"  {pool.sample_id}: {stats['nodes']} nodes, {stats['edges']} edges, "
              f"{stats['components']} components, {stats['independent_cycles']} cycles")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
