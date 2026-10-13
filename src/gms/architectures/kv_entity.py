"""L2 memory architecture: LLM-extracted entity → attribute lists, no edges.

The intermediate rung the literature skips: it buys structure
over L1's flat chunks, but deliberately stores no relationships between
entities — that omission is exactly what separates it from L3, so the
extraction prompt forbids relational output rather than merely not asking for
it.

Retrieval is entity-name match with a vector fallback, per the ladder table:
entity names found in the query select the candidate attributes, and if the
query names no known entity the candidate set widens to everything. Either way
the final cut to `top_k` is by cosine similarity, since a long-lived entity
accumulates far more attributes than any context budget can hold.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import numpy as np

from gms.architectures.base import MemoryArchitecture, Retrieval, turn_content
from gms.architectures.embeddings import BGEEmbedder, Embedder
from gms.data.schema import Turn
from gms.llm import LLMClient

_EXTRACTION_SYSTEM = (
    "You extract structured memory from conversation turns. Report only facts "
    "stated in the turn itself. Never infer or state relationships between "
    "entities — attributes of a single entity only."
)

_EXTRACTION_PROMPT = """Extract the entities mentioned in this conversation turn, and the attributes stated about each one.

Turn:
{turn}

Respond with JSON only, in exactly this shape:
{{"entities": [{{"name": "<entity>", "attributes": ["<attribute>"]}}]}}

If the turn states no facts, respond with {{"entities": []}}."""

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass
class Attribute:
    """One extracted fact, kept with the turn it came from.

    Provenance is stored at extraction time because it cannot be recovered
    later: the attribute text is the model's paraphrase, not a quote.
    """

    entity: str
    text: str
    source_turn_id: str
    # Carried so distilled attributes stay temporally grounded. Without it L2
    # would hand the generator "went to a support group yesterday" with no
    # anchor, losing against L0/L1 on temporal questions for a rendering reason
    # rather than an architectural one.
    source_date: str = ""

    def render(self) -> str:
        body = f"{self.entity}: {self.text}"
        return f"[{self.source_date}] {body}" if self.source_date else body


def _parse_entities(raw: str) -> list[tuple[str, list[str]]] | None:
    """Pull (entity, attributes) pairs out of a model response.

    Returns `None` when the response isn't usable at all, and `[]` when the
    model correctly reported that the turn states no facts. The distinction
    matters: conversational turns are full of greetings and backchannel that
    legitimately extract to nothing, so collapsing the two would report a
    failure rate made mostly of the model behaving exactly as asked.
    """
    try:
        payload = json.loads(_JSON_FENCE_RE.sub("", raw))
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        return None

    parsed = []
    for entity in payload["entities"]:
        if not isinstance(entity, dict):
            continue
        name = entity.get("name")
        attributes = entity.get("attributes") or []
        if isinstance(name, str) and name.strip() and isinstance(attributes, list):
            texts = [a.strip() for a in attributes if isinstance(a, str) and a.strip()]
            if texts:
                parsed.append((name.strip(), texts))
    return parsed


class KVEntityMemory(MemoryArchitecture):
    """Entity-keyed attribute store built by an LLM, one extraction per turn."""

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder | None = None,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.embedder = embedder or BGEEmbedder()
        self.top_k = top_k
        self._attributes: list[Attribute] = []
        self._embeddings: list[np.ndarray] = []
        # Lowercased entity name -> indices into _attributes.
        self._by_entity: dict[str, list[int]] = {}

    def add_turn(self, turn: Turn) -> None:
        prompt = _EXTRACTION_PROMPT.format(turn=turn_content(turn))

        start = time.perf_counter()
        response = self.llm.complete(prompt, system=_EXTRACTION_SYSTEM)
        extracted = _parse_entities(response.text)

        if extracted is None:
            # Unusable response. An empty-but-valid extraction is not a failure:
            # greetings and backchannel genuinely carry no facts.
            self.cost.failures += 1
            extracted = []

        for entity, texts in extracted:
            for text in texts:
                attribute = Attribute(
                    entity=entity,
                    text=text,
                    source_turn_id=turn.dia_id,
                    source_date=turn.session_date,
                )
                self._by_entity.setdefault(entity.lower(), []).append(len(self._attributes))
                self._attributes.append(attribute)
                # Embedded without the timestamp, for the same reason as L1.
                embedded = f"{entity}: {text}"
                self._embeddings.append(self.embedder.embed([embedded])[0])
                # Embedding is construction work too; counting only LLM tokens
                # would understate L2 against L1, which does count them.
                self.cost.construction_tokens += self.embedder.count_tokens(embedded)

        self.cost.construction_seconds += time.perf_counter() - start
        self.cost.construction_tokens += response.total_tokens
        self.cost.storage_entries = len(self._attributes)

    def retrieve(self, query: str) -> Retrieval:
        self.cost.queries += 1
        if not self._attributes:
            return Retrieval(context="")

        start = time.perf_counter()

        candidates = self._entity_matches(query)
        if not candidates:
            candidates = range(len(self._attributes))  # vector fallback

        candidates = list(candidates)
        query_vec = self.embedder.embed_query(query)
        matrix = np.stack([self._embeddings[i] for i in candidates])
        scores = matrix @ query_vec  # rows are L2-normalized, so dot product == cosine similarity
        ranked = [candidates[i] for i in np.argsort(-scores)[: self.top_k]]

        # Chronological, not rank order — matching L0/L1/L3, so context ordering
        # never becomes an accidental architecture variable. Order carries
        # meaning once the generator reads it: relative time references only
        # resolve against the turns around them.
        selected = [self._attributes[i] for i in sorted(ranked)]
        retrieval = Retrieval(
            context="\n".join(a.render() for a in selected),
            # Several attributes can share a turn; report each turn once.
            source_turn_ids=list(dict.fromkeys(a.source_turn_id for a in selected)),
        )

        self.cost.retrieval_seconds += time.perf_counter() - start
        self.cost.retrieval_tokens += self.embedder.count_tokens(query)
        return retrieval

    def _entity_matches(self, query: str) -> list[int]:
        """Indices of attributes whose entity is named in the query.

        Matched on word boundaries so a short name ("Mel") doesn't fire on a
        longer one ("Melanie") that happens to contain it.
        """
        lowered = query.lower()
        matched: list[int] = []
        for name, indices in self._by_entity.items():
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                matched.extend(indices)
        return matched
