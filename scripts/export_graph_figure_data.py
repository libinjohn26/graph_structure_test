#!/usr/bin/env python
"""Export subgraphs and degree/component distributions for the structure figure.

The figure's job is to make finding #10 visible: LoCoMo's extracted graph is a
hub-and-spoke star around two speakers, HotpotQA's is a field of disconnected
islands, and that difference is why traversal behaves differently on them.

**Sampling is deliberately naive, so the picture is honest.** Each panel shows
the subgraph induced by a contiguous block of turns — the first `--turns` of one
conversation or pool — rather than a hand-picked neighbourhood. A curated
"representative" subgraph would be the easiest way to draw a conclusion that the
full graph does not support. The quantitative panels (degree and component
distributions) are computed over the *complete* graphs for all 10 corpora, so
the claim never rests on the sample.

Positions are laid out here with a fixed seed rather than in the browser, so the
figure is reproducible and the page needs no physics engine.

Usage:
    python scripts/export_graph_figure_data.py --out data/processed/figure_graphs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

SEED = 7


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


def build(turns, cache, fingerprint):
    graph = nx.Graph()
    nodes_of_turn: dict[str, set[str]] = defaultdict(set)
    for turn in turns:
        raw = cache.get(extraction_key(fingerprint, turn))
        if raw is None:
            continue
        for triple in _parse_triples(raw) or []:
            subject, obj = triple.subject.lower(), triple.object.lower()
            nodes_of_turn[turn.dia_id].update((subject, obj))
            if graph.has_edge(subject, obj):
                graph[subject][obj]["weight"] += 1
            else:
                graph.add_edge(subject, obj, weight=1)
    return graph, nodes_of_turn


def sample_panel(turns, cache, fingerprint, n_turns, label):
    """Subgraph induced by the first `n_turns` turns — not a curated selection."""
    graph, nodes_of_turn = build(turns[:n_turns], cache, fingerprint)
    df = Counter()
    for node_set in nodes_of_turn.values():
        for node in node_set:
            df[node] += 1

    # Spring layout on the whole sample so disconnected pieces are spread out
    # rather than piled at the origin; a fixed seed keeps the figure stable.
    positions = nx.spring_layout(graph, seed=SEED, k=0.55, iterations=220)
    xs = [p[0] for p in positions.values()] or [0.0]
    ys = [p[1] for p in positions.values()] or [0.0]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0

    component_of = {}
    for index, component in enumerate(nx.connected_components(graph)):
        for node in component:
            component_of[node] = index

    nodes = [
        {
            "id": name,
            "x": round((positions[name][0] - min(xs)) / span_x, 4),
            "y": round((positions[name][1] - min(ys)) / span_y, 4),
            "deg": graph.degree(name),
            "turns": df[name],
            "component": component_of[name],
        }
        for name in graph
    ]
    index_of = {n["id"]: i for i, n in enumerate(nodes)}
    edges = [[index_of[a], index_of[b]] for a, b in graph.edges()]

    return {
        "label": label,
        "turns_shown": min(n_turns, len(turns)),
        "nodes": nodes,
        "edges": edges,
        "components": nx.number_connected_components(graph),
        "max_degree": max((n["deg"] for n in nodes), default=0),
        "top_entities": [
            {"id": name, "turns": count, "share": round(count / min(n_turns, len(turns)), 4)}
            for name, count in df.most_common(5)
        ],
    }


def corpus_stats(all_graphs, all_dfs, total_turns):
    """Degree and component distributions over the complete graphs."""
    degrees = Counter()
    components = []
    top_shares = []
    for graph, df, turns in zip(all_graphs, all_dfs, total_turns):
        for _, d in graph.degree():
            degrees[d] += 1
        components.extend(len(c) for c in nx.connected_components(graph))
        if df:
            top_shares.append(df.most_common(1)[0][1] / turns)

    component_bins = Counter()
    for size in components:
        if size <= 2:
            component_bins["1-2"] += 1
        elif size <= 5:
            component_bins["3-5"] += 1
        elif size <= 20:
            component_bins["6-20"] += 1
        elif size <= 100:
            component_bins["21-100"] += 1
        else:
            component_bins["100+"] += 1

    total_nodes = sum(degrees.values())
    return {
        "degree_distribution": [
            {"degree": d, "count": c, "share": round(c / total_nodes, 5)}
            for d, c in sorted(degrees.items())
            if d <= 20
        ],
        "degree_tail_max": max(degrees),
        "component_bins": [
            {"bin": b, "count": component_bins.get(b, 0)}
            for b in ("1-2", "3-5", "6-20", "21-100", "100+")
        ],
        "top_entity_share_mean": round(sum(top_shares) / len(top_shares), 4),
        "nodes_total": total_nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(REPO_ROOT / "data" / "processed" / "llm_cache.jsonl"))
    parser.add_argument("--turns", type=int, default=90, help="turns per sampled panel")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "figure_graphs.json"))
    args = parser.parse_args()

    load_env_file()
    fingerprint = VLLMClient.from_env().fingerprint
    cache = load_cache(Path(args.cache))
    print(f"cache entries: {len(cache):,}")

    locomo = load_locomo(REPO_ROOT / "data" / "raw" / "locomo" / "locomo10.json")
    hotpot = load_hotpotqa(
        REPO_ROOT / "data" / "raw" / "hotpotqa" / "validation_distractor.parquet",
        questions_per_pool=20,
    )[:10]

    report = {"seed": SEED, "turns_per_panel": args.turns}

    lo_turns = [t for s in locomo[0].sessions for t in s.turns]
    hq_turns = [t for s in hotpot[0].sessions for t in s.turns]
    report["locomo_sample"] = sample_panel(lo_turns, cache, fingerprint, args.turns, "LoCoMo")
    report["hotpotqa_sample"] = sample_panel(hq_turns, cache, fingerprint, args.turns, "HotpotQA")
    report["locomo_sample"]["speakers"] = [
        locomo[0].speaker_a.lower(), locomo[0].speaker_b.lower()
    ]
    report["hotpotqa_sample"]["speakers"] = []

    for key, corpora in (("locomo", locomo), ("hotpotqa", hotpot)):
        graphs, dfs, totals = [], [], []
        for conv in corpora:
            turns = [t for s in conv.sessions for t in s.turns]
            graph, nodes_of_turn = build(turns, cache, fingerprint)
            df = Counter()
            for node_set in nodes_of_turn.values():
                for node in node_set:
                    df[node] += 1
            graphs.append(graph)
            dfs.append(df)
            totals.append(len(turns))
        report[f"{key}_stats"] = corpus_stats(graphs, dfs, totals)
        report[f"{key}_stats"]["corpora"] = len(corpora)
        print(f"  {key}: {len(corpora)} corpora, "
              f"{report[f'{key}_stats']['nodes_total']:,} nodes total")

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nWritten to {out} ({out.stat().st_size/1024:.0f} KB)")
    for key in ("locomo_sample", "hotpotqa_sample"):
        p = report[key]
        print(f"  {p['label']:9s} panel: {len(p['nodes'])} nodes, {len(p['edges'])} edges, "
              f"{p['components']} components, max degree {p['max_degree']}")
        top = ", ".join(f"{e['id']} ({e['share']:.0%} of turns)" for e in p["top_entities"][:3])
        print(f"    top entities: {top}")


if __name__ == "__main__":
    main()
