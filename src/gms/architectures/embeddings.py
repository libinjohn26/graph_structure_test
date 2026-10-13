"""Text embedding backends for memory architectures that need vector retrieval
(L1, hybrid).

Kept separate from the architectures themselves so every vector-based
architecture shares one embedding call path, and so the backend — which
model, which device — is a single swappable piece behind the `Embedder`
interface. Production default is `BGEEmbedder`, matching `models.embedding`
in configs/default.yaml; tests inject a fake `Embedder` instead so they don't
need the real model downloaded.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Embedder(ABC):
    """Turns text into L2-normalized vectors. Implementations own model loading."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of passages. Returns an (len(texts), dim) float32 array."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single retrieval query — may differ from `embed` (e.g. an
        instruction prefix), so it isn't just `embed([text])[0]`."""

    def count_tokens(self, text: str) -> int:
        """Token count for cost tracking. Default is a word-count approximation;
        backends with a real tokenizer should override for accuracy."""
        return len(text.split())

    @property
    def fingerprint(self) -> str:
        """Identity of everything that changes this embedder's output, for cache
        keys. Anything omitted becomes a way to serve a stale vector."""
        return type(self).__name__


class BGEEmbedder(Embedder):
    """Wraps BAAI/bge-large-en-v1.5 via sentence-transformers.

    BGE's retrieval instructions call for prefixing the *query* only, not
    passages — see the model card. Loading the model is deferred to first use
    so importing this module doesn't require sentence-transformers/torch
    unless an architecture actually instantiates this class.
    """

    _QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            self._QUERY_INSTRUCTION + text, normalize_embeddings=True, convert_to_numpy=True
        )

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text)["input_ids"])

    @property
    def fingerprint(self) -> str:
        return f"bge|{self._model_name}"


class CachingEmbedder(Embedder):
    """Disk cache in front of another embedder.

    Embedding is the dominant CPU cost of a resumed run and, unlike the LLM
    calls, it was being re-paid in full every time: a full HotpotQA pass embeds
    ~8k turns plus ~30k triples and nodes, so an interrupted run — a degraded
    server, a closed laptop — cost 35-60 minutes to get back to where it was.
    Embedding is deterministic for a fixed model, so none of that was necessary.

    Stored as one `.npy` per vector under a sharded directory, keyed by a hash of
    the backend fingerprint and the text. The fingerprint is part of the key so
    changing the embedding model cannot silently serve vectors from the old one —
    the same correctness rule as the LLM cache.

    Vectors are small and numerous, so this trades inodes for time; a full run
    leaves roughly 40k files of ~4KB. Delete the directory to invalidate.
    """

    def __init__(self, inner: Embedder, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.hits = 0
        self.misses = 0

    @property
    def fingerprint(self) -> str:
        return self.inner.fingerprint

    def _vector_path(self, text: str, kind: str) -> Path:
        digest = hashlib.sha256(
            "\x00".join([self.fingerprint, kind, text]).encode("utf-8")
        ).hexdigest()
        # Sharded two levels: a single flat directory of 40k entries is slow to
        # stat on Windows.
        return self.path / digest[:2] / digest[2:4] / f"{digest}.npy"

    def _load(self, text: str, kind: str) -> np.ndarray | None:
        target = self._vector_path(text, kind)
        if not target.exists():
            return None
        try:
            return np.load(target)
        except (OSError, ValueError):
            # A partially written file from a killed run; treat as a miss.
            return None

    def _store(self, text: str, kind: str, vector: np.ndarray) -> None:
        target = self._vector_path(text, kind)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted write never leaves a half file that
        # a later run would load as a valid vector.
        #
        # The temp name carries pid and a random suffix because the target name
        # is a pure hash of the content: two processes embedding the same text
        # would otherwise pick the *same* temp path, and on Windows the second
        # os.replace fails with WinError 32 while the first still holds it open.
        # Two concurrent sweeps sharing a cache is normal here, so this has to be
        # safe rather than merely unlikely.
        temporary = target.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        # Written through a handle: np.save appends ".npy" to a path whose name
        # does not already end in it, which would leave the rename with nothing
        # to move.
        with temporary.open("wb") as handle:
            np.save(handle, vector)
        try:
            temporary.replace(target)
        except OSError:
            # Another process won the race and wrote an identical vector — the
            # content is a function of the key, so either copy is correct.
            temporary.unlink(missing_ok=True)
            if not target.exists():
                raise

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray | None] = [self._load(t, "passage") for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]

        if missing:
            computed = self.inner.embed([texts[i] for i in missing])
            for index, vector in zip(missing, computed):
                self._store(texts[index], "passage", vector)
                vectors[index] = vector

        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        return np.stack(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        # Keyed separately from passages: BGE prefixes queries, so the same
        # string embeds differently on each path.
        cached = self._load(text, "query")
        if cached is not None:
            self.hits += 1
            return cached

        vector = self.inner.embed_query(text)
        self._store(text, "query", vector)
        self.misses += 1
        return vector

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)
