"""L3 memory architecture: LLM-extracted triples retrieved by personalized PageRank.

The cheap graph rung, following HippoRAG2: the graph decides *ranking*, and the
turns themselves are what gets handed to the generation LLM. Construction pulls
subject-relation-object triples from each turn; retrieval seeds PageRank at the
entities named in the query, propagates through the graph, and scores a turn by
the summed rank of the entities it mentions.

Two deliberate contrasts with the rungs either side of it:

- **Against L2.** The extraction prompt here *requires* relationships, where
  L2's forbids them. Same model, same per-turn call, same cost shape.
  **Known confound, deliberately kept:** L3 also runs a vector search over all
  triples on every query, while L2 reaches its vector path only when the query
  names no known entity. So this pair varies retrieval channel as well as
  structure, and an L3-over-L2 margin cannot be attributed to edges alone. The
  asymmetry is kept because removing it would mean crippling L3's query linking,
  which is what makes it HippoRAG2 — see the README on comparing graph variants.
- **Against L1.** Both return raw turns, so a score difference is a difference
  in how candidates were ranked, not in what the generator was given. L3 vs L2
  is not that clean: it varies representation *and* structure at once.

Returning turns rather than triples is what HippoRAG2 does, and it keeps the
comparison with L1 interpretable. The community-summary variant (L3') is where
a distilled representation gets tested instead.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np

from gms.architectures.base import MemoryArchitecture, Retrieval, format_turn, turn_content
from gms.architectures.embeddings import BGEEmbedder, Embedder
from gms.data.schema import Turn
from gms.llm import LLMClient

_EXTRACTION_SYSTEM = (
    "You extract a knowledge graph from conversation turns. Report the "
    "relationships stated in the turn as subject-relation-object triples, using "
    "only what the turn says. Name entities explicitly rather than using pronouns."
)

_EXTRACTION_PROMPT = """Extract the relationships stated in this conversation turn as triples.

Turn:
{turn}

Respond with JSON only, in exactly this shape:
{{"triples": [{{"subject": "<entity>", "relation": "<relation>", "object": "<entity or value>"}}]}}

If the turn states no relationships, respond with {{"triples": []}}."""

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# Defaults for the seeding knobs. Exposed as constructor arguments rather than
# baked in, because none of them has been tuned and a module constant cannot be
# swept without editing the file.
#
# How many nodes to seed with when the query names no known entity. Kept small:
# a broad seed set flattens the personalization vector into plain PageRank,
# which ranks by global connectivity and ignores the question entirely.
_FALLBACK_SEEDS = 3

# PageRank damping. Set explicitly rather than inherited from networkx, so the
# traversal depth this project reports is a recorded choice and not whatever a
# future networkx release defaults to. `alpha=0` disables traversal entirely,
# which is the baseline every graph variant must be compared against.
_PAGERANK_ALPHA = 0.85

# Exponent on query-to-triple cosine used as edge weight. **Non-zero by
# default, deliberately**: 0.0 means static triple counts, and relevance-blind
# traversal measured *worse than not traversing at all* (106 vs 110 evidence
# hits on LoCoMo). Leaving the off switch as the default made it the setting a
# caller gets by not thinking about it, and that silently weakened the first
# HotpotQA control run. 8 was the best of {1, 2, 4, 8, 16, 32} on LoCoMo and
# improves HotpotQA recall 0.646 -> 0.661; the curve is flat enough above 8 that
# the exact value matters little. Pass 0.0 to reproduce pre-2026-10 runs.
_RELATION_WEIGHT_POWER = 8.0

# How many triples the query is linked to before seeding. This is the step that
# makes it HippoRAG2 rather than HippoRAG1: entity-only seeding cannot see a
# query whose key term is a *relation*. "What did Caroline research?" seeds at
# the entities "caroline" and "research", while the turn that answers it holds
# (Caroline) -[is researching]-> (adoption agencies), where "research" is an
# edge label and therefore invisible. Matching the whole rendered triple finds it.
_SEED_TRIPLES = 5


@dataclass
class Triple:
    """One extracted relationship, kept with the turn it came from."""

    subject: str
    relation: str
    object: str
    source_turn_id: str

    def render(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"


def _parse_triples(raw: str) -> list[Triple] | None:
    """Pull triples out of a model response, without their provenance.

    Returns `None` for an unusable response and `[]` for a turn that genuinely
    states no relationships — the same distinction L2 draws, and for the same
    reason: conversational filler legitimately yields nothing, and collapsing
    the two makes `CostLog.failures` mostly a record of correct behaviour.
    """
    try:
        payload = json.loads(_JSON_FENCE_RE.sub("", raw))
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict) or not isinstance(payload.get("triples"), list):
        return None

    parsed = []
    for item in payload["triples"]:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        relation = item.get("relation")
        obj = item.get("object")
        if all(isinstance(v, str) and v.strip() for v in (subject, relation, obj)):
            parsed.append(
                Triple(
                    subject=subject.strip(),
                    relation=relation.strip(),
                    object=obj.strip(),
                    source_turn_id="",
                )
            )
    return parsed


_ARTICLES = {"the", "a", "an"}


def normalise_entity(name: str) -> str:
    """Surface normalisation applied before a name becomes a node id."""
    tokens = [t.strip(".,!?'\"()") for t in name.lower().split()]
    tokens = [t for t in tokens if t and t not in _ARTICLES]
    return " ".join(tokens) or name.lower()


def build_resolution_map(names: set[str]) -> dict[str, str]:
    """Map each surface name to the node it should merge into.

    Deliberately conservative and lexical — no embeddings, no LLM, no
    re-extraction. A single-token name folds into a multi-token name that
    contains it as a whole token (`burton` -> `tim burton`), which is the
    fragmentation finding #10 identified as the main cause of disconnection.

    Narrow on purpose. A wrong merge fuses two distinct entities permanently and
    silently, which is worse than leaving them apart, so this errs toward
    under-merging and puts a *lower bound* on what resolution can recover.
    """
    canonical = {name: name for name in names}
    # Longest first, so a fragment attaches to the fullest form available.
    ordered = sorted(names, key=lambda n: (-len(n.split()), -len(n)))
    by_token: dict[str, list[str]] = defaultdict(list)
    for name in ordered:
        for token in set(name.split()):
            by_token[token].append(name)

    for name in ordered:
        if len(name.split()) != 1 or len(name) < 4:
            continue
        # Candidates share this token, so the whole vocabulary is never scanned.
        for longer in by_token.get(name, ()):
            if len(longer.split()) < 2 or canonical[longer] != longer:
                continue
            if name in longer.split():
                canonical[name] = longer
                break
    return canonical


class GraphMemory(MemoryArchitecture):
    """Entity graph built from triples, retrieved by personalized PageRank."""

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder | None = None,
        top_k: int = 5,
        synonym_threshold: float = 0.0,
        resolve_entities: bool = False,
        relation_weight_power: float = _RELATION_WEIGHT_POWER,
        alpha: float = _PAGERANK_ALPHA,
        seed_triples: int = _SEED_TRIPLES,
        fallback_seeds: int = _FALLBACK_SEEDS,
    ) -> None:
        """`synonym_threshold` adds HippoRAG-style synonym edges between nodes
        whose embeddings exceed that cosine similarity; 0.0 disables them.

        Per-turn extraction with no entity resolution leaves the graph barely
        connected — measured on one conversation, 824 of 933 nodes appeared in a
        single turn, average degree was 2.13, and ablating traversal entirely
        *improved* retrieval. "Mel" and "Melanie" are separate nodes with nothing
        joining them. Synonym edges are HippoRAG's answer: link near-identical
        entities instead of merging them, so a mistaken link adds a weak path
        rather than silently fusing two distinct entities.
        """
        super().__init__()
        self.llm = llm
        self.embedder = embedder or BGEEmbedder()
        self.top_k = top_k
        self.synonym_threshold = synonym_threshold
        # Off by default: every result published before this existed predates it,
        # was produced without it, and silently changing the graph would
        # invalidate them.
        self.resolve_entities = resolve_entities
        self.relation_weight_power = relation_weight_power
        self.alpha = alpha
        self.seed_triples = seed_triples
        self.fallback_seeds = fallback_seeds
        self._synonyms_stale = True
        self._resolution_stale = True
        # Any surface name ever seen -> the node it now lives in; identity for
        # anything unmerged. Triples keep their original strings, so a seed taken
        # from a triple must be mapped through this or it names a node that no
        # longer exists.
        self._canonical: dict[str, str] = {}
        # Canonical edge key -> indices into _triples, so an edge can be scored
        # by what its relations *say* rather than only by how many there are.
        self._edge_triples: dict[tuple[str, str], list[int]] = defaultdict(list)
        # Undirected on purpose. Edges run subject -> object, but relation
        # direction is an artifact of how the extractor phrased the triple, not a
        # constraint on retrieval — and keeping it blocks the canonical multi-hop
        # pattern, where two facts converge on a shared subject. Answering "what
        # does the person who moved from Sweden own?" means walking *back* from
        # Sweden to Caroline before going forward to Rex. Measured on one
        # conversation, 75% of nodes were sinks and directed traversal reached
        # 43.5 nodes at two hops against 184 undirected.
        self._graph = nx.Graph()
        self._turns: dict[str, Turn] = {}
        self._turn_order: dict[str, int] = {}
        # Lowercased entity -> ids of turns that mention it. Scoring a turn means
        # summing the rank of its entities, so this is the inverted index PPR needs.
        self._turn_ids_by_node: dict[str, set[str]] = defaultdict(set)
        self._node_embeddings: dict[str, np.ndarray] = {}
        # Triples kept alongside the graph so a query can be linked to a whole
        # relationship, not only to entity names.
        self._triples: list[Triple] = []
        self._triple_embeddings: list[np.ndarray] = []

    def add_turn(self, turn: Turn) -> None:
        prompt = _EXTRACTION_PROMPT.format(turn=turn_content(turn))

        start = time.perf_counter()
        response = self.llm.complete(prompt, system=_EXTRACTION_SYSTEM)
        triples = _parse_triples(response.text)

        if triples is None:
            self.cost.failures += 1
            triples = []

        self._turns[turn.dia_id] = turn
        self._turn_order[turn.dia_id] = len(self._turn_order)

        for triple in triples:
            triple.source_turn_id = turn.dia_id
            self._add_triple(triple)

        self._synonyms_stale = True
        self._resolution_stale = True
        self.cost.construction_seconds += time.perf_counter() - start
        self.cost.construction_tokens += response.total_tokens
        self.cost.storage_entries = self._graph.number_of_edges()

    def _canon(self, name: str) -> str:
        """The node a surface name currently lives in."""
        return self._canonical.get(name, name)

    def _ensure_entity_resolution(self) -> None:
        """Merge co-referent nodes, once, after ingestion settles.

        Lazy for the same reason synonym edges are: resolution is inherently
        retroactive — `burton` can only be known to fold into `tim burton` once
        the longer name has arrived — and turns stream in one at a time, so doing
        it per turn would mean repeatedly rewriting the graph.

        **This is a merge, not a link.** `synonym_threshold` adds an *edge*
        between similar nodes, which is HippoRAG's conservative answer: a wrong
        link adds a weak path, where a wrong merge silently fuses two distinct
        entities. Merging is the stronger intervention, which is why it is off by
        default and why the rule is lexical and narrow rather than
        embedding-based.

        **It cuts both ways, and the second effect is easy to miss.** Seeds and
        turn scores are weighted by node specificity, `1/|turns mentioning the
        node|`. Merging `burton` (3 turns) into `tim burton` (12) yields one node
        in 15, so its specificity *falls* — resolution buys connectivity and
        spends discriminative weighting, the very thing this rung depends on.
        Whether that nets out positive is exactly what measuring it settles.
        """
        if not self._resolution_stale:
            return
        self._resolution_stale = False
        if not self.resolve_entities or self._graph.number_of_nodes() < 2:
            return

        start = time.perf_counter()
        # Normalise first, then merge — the same two steps, in the same order, as
        # `scripts/analyse_graphs.py --resolve-entities`, so the reachability
        # measurement in finding #12 describes the graph this actually builds.
        # Normalisation is applied *here* rather than in `_add_triple` so that
        # leaving `resolve_entities` off keeps node ids byte-identical to every
        # result published before this existed.
        normalised = {node: normalise_entity(node) for node in self._graph}
        folded = build_resolution_map(set(normalised.values()))
        mapping = {
            node: folded.get(norm, norm) for node, norm in normalised.items()
        }
        # A canonical target need not already be a node — normalising "the batman"
        # to "batman" is a rename, not a merge — so targets are created as needed
        # rather than filtered out.
        merges = {n: c for n, c in mapping.items() if c != n}
        if not merges:
            self.cost.construction_seconds += time.perf_counter() - start
            return

        for canonical in set(merges.values()):
            if canonical not in self._graph:
                self._graph.add_node(canonical, display=canonical)

        for node, canonical in merges.items():
            if node not in self._graph:
                continue
            for neighbour, data in list(self._graph[node].items()):
                target = merges.get(neighbour, neighbour)
                if target == canonical:
                    continue  # the edge collapsed onto the merged node itself
                if self._graph.has_edge(canonical, target):
                    existing = self._graph[canonical][target]
                    existing["weight"] = existing.get("weight", 0.0) + data.get("weight", 0.0)
                    existing.setdefault("relations", set()).update(data.get("relations", set()))
                    existing["synonym"] = existing.get("synonym", False) and data.get(
                        "synonym", False
                    )
                else:
                    self._graph.add_edge(canonical, target, **dict(data))
            # The inverted index must merge too, or the absorbed node's turns
            # become unreachable and its specificity is never paid.
            self._turn_ids_by_node[canonical] |= self._turn_ids_by_node.pop(node, set())
            # A renamed node needs its embedding carried over, or the vector
            # fallback in `_seed_nodes` loses it entirely.
            moved = self._node_embeddings.pop(node, None)
            if moved is not None and canonical not in self._node_embeddings:
                self._node_embeddings[canonical] = moved
            self._graph.remove_node(node)

        rekeyed: dict[tuple[str, str], list[int]] = defaultdict(list)
        for (left, right), indices in self._edge_triples.items():
            new_left, new_right = merges.get(left, left), merges.get(right, right)
            if new_left == new_right:
                continue
            rekeyed[tuple(sorted((new_left, new_right)))].extend(indices)
        self._edge_triples = rekeyed

        # Compose, so names already pointing at an absorbed node follow it.
        for name, target in list(self._canonical.items()):
            if target in merges:
                self._canonical[name] = merges[target]
        self._canonical.update(merges)

        self.cost.construction_seconds += time.perf_counter() - start

    def _ensure_synonym_edges(self) -> None:
        """Link embedding-similar nodes, once, after ingestion settles.

        Built lazily rather than per turn because it is all-pairs: doing it on
        every `add_turn` would be quadratic in turns for no benefit, since only
        the final graph is ever queried. Timed into construction, not retrieval —
        it is graph building that merely happens late.
        """
        if not self._synonyms_stale:
            return
        self._synonyms_stale = False
        if self.synonym_threshold <= 0.0 or len(self._node_embeddings) < 2:
            return

        start = time.perf_counter()
        nodes = list(self._node_embeddings)
        matrix = np.stack([self._node_embeddings[n] for n in nodes])
        similarity = matrix @ matrix.T  # rows are L2-normalized
        np.fill_diagonal(similarity, 0.0)

        for i, j in zip(*np.where(similarity >= self.synonym_threshold)):
            if i >= j:  # the matrix is symmetric; take each pair once
                continue
            left, right = nodes[i], nodes[j]
            if not self._graph.has_edge(left, right):
                self._graph.add_edge(
                    left, right, weight=float(similarity[i, j]), relations=set(), synonym=True
                )

        self.cost.construction_seconds += time.perf_counter() - start
        self.cost.storage_entries = self._graph.number_of_edges()

    def _add_triple(self, triple: Triple) -> None:
        subject, obj = triple.subject.lower(), triple.object.lower()

        rendered = triple.render()
        self._edge_triples[tuple(sorted((subject, obj)))].append(len(self._triples))
        self._triples.append(triple)
        self._triple_embeddings.append(self.embedder.embed([rendered])[0])
        # Embedding is construction work too; counting only LLM tokens would
        # understate the structured rungs against L1, which does count them.
        self.cost.construction_tokens += self.embedder.count_tokens(rendered)

        for node, display in ((subject, triple.subject), (obj, triple.object)):
            if node not in self._graph:
                self._graph.add_node(node, display=display)
                self._node_embeddings[node] = self.embedder.embed([display])[0]
                self.cost.construction_tokens += self.embedder.count_tokens(display)
            self._turn_ids_by_node[node].add(triple.source_turn_id)

        if self._graph.has_edge(subject, obj):
            # Repeated relations between the same pair are evidence of a stronger
            # link, which PageRank reads through edge weight.
            self._graph[subject][obj]["weight"] += 1.0
            self._graph[subject][obj]["relations"].add(triple.relation)
        else:
            self._graph.add_edge(subject, obj, weight=1.0, relations={triple.relation})

    def retrieve(self, query: str) -> Retrieval:
        self.cost.queries += 1
        if self._graph.number_of_nodes() == 0:
            return Retrieval(context="")

        # Resolution first: synonym edges operate on the node set, so merging
        # afterwards would strand edges on nodes that no longer exist.
        self._ensure_entity_resolution()
        self._ensure_synonym_edges()

        start = time.perf_counter()

        query_vec = self.embedder.embed_query(query)
        triple_scores = (
            np.stack(self._triple_embeddings) @ query_vec if self._triples else np.zeros(0)
        )

        seeds = self._seed_weights(query, query_vec, triple_scores)
        if not seeds:
            self.cost.retrieval_seconds += time.perf_counter() - start
            return Retrieval(context="")

        weight_attr = self._apply_relation_weights(triple_scores)
        ranks = nx.pagerank(
            self._graph, alpha=self.alpha, personalization=seeds, weight=weight_attr
        )

        # HippoRAG's passage scoring — a turn is worth the summed rank of the
        # entities it mentions — but weighted by node specificity here too, not
        # only in the seeds. A random walk keeps landing on a hub wherever it
        # starts, so without this every turn naming a speaker inherits that rank
        # and the 226 turns mentioning "caroline" score alike again. Weighting by
        # 1/|turns containing the node| makes the sum behave like TF-IDF.
        turn_scores: dict[str, float] = defaultdict(float)
        for node, rank in ranks.items():
            turn_ids = self._turn_ids_by_node[node]
            specificity = 1.0 / len(turn_ids) if turn_ids else 0.0
            for turn_id in turn_ids:
                turn_scores[turn_id] += rank * specificity

        ranked = sorted(turn_scores.items(), key=lambda kv: -kv[1])[: self.top_k]
        selected = sorted(
            (self._turns[turn_id] for turn_id, _ in ranked),
            key=lambda t: self._turn_order[t.dia_id],  # chronological, not rank order
        )

        retrieval = Retrieval(
            context="\n".join(format_turn(t) for t in selected),
            source_turn_ids=[t.dia_id for t in selected],
        )

        self.cost.retrieval_seconds += time.perf_counter() - start
        self.cost.retrieval_tokens += self.embedder.count_tokens(query)
        return retrieval

    def _seed_nodes(self, query: str, query_vec: np.ndarray | None = None) -> list[str]:
        """Entities the query points at, before weighting.

        Named entities in the query first, matched on word boundaries so a short
        name doesn't fire inside a longer one. When the query names nothing the
        graph knows, fall back to the nearest nodes by embedding — the same
        vector fallback L2 uses, so neither rung gets an advantage from simply
        having a fallback.
        """
        lowered = query.lower()
        matched = [
            node for node in self._graph if re.search(rf"\b{re.escape(node)}\b", lowered)
        ]
        if matched:
            return matched

        if query_vec is None:
            query_vec = self.embedder.embed_query(query)
        nodes = list(self._node_embeddings)
        matrix = np.stack([self._node_embeddings[n] for n in nodes])
        scores = matrix @ query_vec
        return [nodes[i] for i in np.argsort(-scores)[: self.fallback_seeds]]

    def _linked_triple_nodes(self, triple_scores: np.ndarray) -> list[str]:
        """Entities of the triples most similar to the query.

        HippoRAG2's query linking: the unit matched against is the rendered
        triple, so a question whose key term is a relation still reaches the
        right part of the graph.
        """
        if not self._triples:
            return []

        nodes = []
        for index in np.argsort(-triple_scores)[: self.seed_triples]:
            triple = self._triples[index]
            # Mapped through the resolution table: triples keep their original
            # surface strings, so after a merge the raw name is not a node and
            # `_seed_weights` would drop it silently, costing seeds exactly on
            # the queries resolution was meant to help.
            nodes.extend([self._canon(triple.subject.lower()), self._canon(triple.object.lower())])
        return nodes

    def _apply_relation_weights(self, triple_scores: np.ndarray) -> str:
        """Re-weight edges by what their relations say about *this* query.

        Returns the edge attribute PageRank should walk on.

        The problem this addresses: a hub node is uninformative, but its edges
        are not. "caroline" sits on 363 edges, so walking through her reaches
        everything — yet only a handful of those edges are
        `-[is researching]->`. Static weights (how many triples join a pair)
        cannot tell them apart, so traversal through the one entity that
        actually connects most multi-hop evidence is pure diffusion.

        An edge takes the **best** score among its triples, not the mean: one
        highly relevant relation should make an edge worth walking even when
        the same pair is also joined by irrelevant ones.

        `relation_weight_power` sharpens the contrast. Sentence-embedding cosine
        has a high floor — unrelated text still scores ~0.3 — so raw scores give
        a relevant edge barely twice the weight of an irrelevant one, and
        PageRank normalises per node, leaving almost no differentiation.
        Exponentiating widens that gap. 0.0 disables the whole mechanism and
        falls back to static counts.
        """
        if self.relation_weight_power <= 0.0 or not self._triples:
            return "weight"

        for (left, right), indices in self._edge_triples.items():
            if not self._graph.has_edge(left, right):
                continue
            best = max(float(triple_scores[i]) for i in indices)
            self._graph[left][right]["query_weight"] = (
                max(best, 1e-6) ** self.relation_weight_power
            )

        # Synonym edges carry no relation, so they keep a small constant weight:
        # traversable, but never preferred over a relation that matches the query.
        for left, right, data in self._graph.edges(data=True):
            if "query_weight" not in data:
                data["query_weight"] = 1e-6

        return "query_weight"

    def _seed_weights(
        self, query: str, query_vec: np.ndarray | None = None, triple_scores: np.ndarray | None = None
    ) -> dict[str, float]:
        """Personalization vector over seed entities, weighted by specificity.

        Seeds come from two places: entities the query names outright, and the
        entities of the triples it most resembles. Both are needed — name
        matching alone misses relation-centric questions, and triple linking
        alone can drift when the query names an entity explicitly. Note that
        this makes the vector channel unconditional here but conditional in L2;
        the module docstring records why that confound is kept.

        A node's weight is the inverse of how many turns mention it, which is
        HippoRAG's node specificity. Without it the speakers swamp everything:
        measured on one LoCoMo conversation, "caroline" appeared in 226 of 419
        turns and took 32% of the rank mass, so every turn naming her scored
        alike and the ranking among them was arbitrary.
        """
        # Embedded once and reused: seeding, triple linking and relation weighting
        # all need the same vector, and embedding repeatedly would bill the query
        # once while paying for it several times.
        if query_vec is None:
            query_vec = self.embedder.embed_query(query)
        if triple_scores is None:
            triple_scores = (
                np.stack(self._triple_embeddings) @ query_vec if self._triples else np.zeros(0)
            )

        seeds = set(self._seed_nodes(query, query_vec)) | set(
            self._linked_triple_nodes(triple_scores)
        )
        seeds = {node for node in seeds if node in self._graph}
        if not seeds:
            return {}

        weights = {node: 1.0 / len(self._turn_ids_by_node[node]) for node in seeds}
        total = sum(weights.values())
        return {node: weight / total for node, weight in weights.items()}

    def graph_stats(self) -> dict[str, float]:
        """Graph-quality metrics from the metric stack in the README.

        Reported per architecture instance rather than aggregated, since a graph
        is built per conversation.
        """
        self._ensure_entity_resolution()
        self._ensure_synonym_edges()
        nodes = self._graph.number_of_nodes()
        edges = self._graph.number_of_edges()
        return {
            "nodes": nodes,
            "edges": edges,
            "avg_degree": (2 * edges / nodes) if nodes else 0.0,
            "clustering": nx.average_clustering(self._graph) if nodes > 1 else 0.0,
        }
