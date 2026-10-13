import numpy as np
import pytest
from fakes import StableHashEmbedder

from gms.architectures.embeddings import CachingEmbedder


class CountingEmbedder(StableHashEmbedder):
    """Records how much work the backend was actually asked to do."""

    def __init__(self, dim: int = 64, name: str = "counting") -> None:
        super().__init__(dim)
        self.name = name
        self.passages = 0
        self.queries = 0

    @property
    def fingerprint(self) -> str:
        return self.name

    def embed(self, texts):
        self.passages += len(texts)
        return super().embed(texts)

    def embed_query(self, text):
        self.queries += 1
        return super().embed_query(text)


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "embeddings"


def test_repeated_text_is_not_re_embedded(cache_dir):
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner, cache_dir)

    first = cached.embed(["alpha", "beta"])
    second = cached.embed(["alpha", "beta"])

    assert inner.passages == 2  # the backend saw each text once
    assert np.array_equal(first, second)
    assert (cached.hits, cached.misses) == (2, 2)


def test_partial_hits_only_embed_the_missing_texts(cache_dir):
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner, cache_dir)

    cached.embed(["alpha"])
    result = cached.embed(["alpha", "beta", "gamma"])

    # "alpha" came from disk; only the two new texts cost anything.
    assert inner.passages == 3
    assert result.shape[0] == 3


def test_order_is_preserved_when_some_are_cached(cache_dir):
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner, cache_dir)

    cached.embed(["beta"])
    mixed = cached.embed(["alpha", "beta", "gamma"])
    direct = CountingEmbedder().embed(["alpha", "beta", "gamma"])

    # A cache that returned rows out of order would silently mis-assign every
    # vector to the wrong turn.
    assert np.allclose(mixed, direct)


def test_cache_survives_a_new_process(cache_dir):
    CachingEmbedder(CountingEmbedder(), cache_dir).embed(["alpha"])

    fresh = CountingEmbedder()
    CachingEmbedder(fresh, cache_dir).embed(["alpha"])

    assert fresh.passages == 0


def test_queries_and_passages_are_keyed_separately(cache_dir):
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner, cache_dir)

    cached.embed(["alpha"])
    cached.embed_query("alpha")

    # BGE prefixes queries, so the same string embeds differently on each path;
    # sharing a key would serve the passage vector for a query.
    assert inner.passages == 1
    assert inner.queries == 1


def test_a_different_model_does_not_reuse_vectors(cache_dir):
    CachingEmbedder(CountingEmbedder(name="model-a"), cache_dir).embed(["alpha"])

    other = CountingEmbedder(name="model-b")
    CachingEmbedder(other, cache_dir).embed(["alpha"])

    assert other.passages == 1


def test_a_truncated_vector_file_is_treated_as_a_miss(cache_dir):
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner, cache_dir)
    cached.embed(["alpha"])

    target = cached._vector_path("alpha", "passage")
    target.write_bytes(b"\x93NUMPY truncated")

    # A run killed mid-write must not poison the cache.
    assert cached.embed(["alpha"]).shape[0] == 1
    assert inner.passages == 2


def test_count_tokens_delegates(cache_dir):
    cached = CachingEmbedder(CountingEmbedder(), cache_dir)

    assert cached.count_tokens("one two three") == 3
