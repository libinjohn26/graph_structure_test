"""Test doubles for the model-backed pieces.

Keeps the suite fast and offline: no 1.3GB embedding model, no LLM server.
Both fakes are deterministic *across* runs, not just within one — Python's
builtin `hash()` is salted per process, so a fake built on it makes retrieval
order vary between runs and tests pass or fail by luck of the seed.
"""

from __future__ import annotations

import hashlib

import numpy as np

from gms.architectures.embeddings import Embedder
from gms.llm import LLMClient, LLMResponse


def _bucket(word: str, dim: int) -> int:
    digest = hashlib.md5(word.encode("utf-8")).digest()[:4]
    return int.from_bytes(digest, "little") % dim


class StableHashEmbedder(Embedder):
    """Bag-of-words hashing embedder standing in for BGEEmbedder.

    Carries no semantics — similarity is literal word overlap — so tests that
    use it must assert on shared vocabulary rather than meaning.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            vec[_bucket(word, self.dim)] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class ScriptedLLM(LLMClient):
    """Replays queued responses in order, recording the prompts it was given."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        text = self._responses.pop(0) if self._responses else '{"entities": []}'
        return LLMResponse(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )
