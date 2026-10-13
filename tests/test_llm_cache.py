import pytest
from fakes import ScriptedLLM

from gms.llm import CachingLLMClient, LLMResponse


class CountingLLM(ScriptedLLM):
    """Returns a distinct response per call, so a cache hit is visible."""

    def complete(self, prompt, *, system=None):
        response = super().complete(prompt, system=system)
        return LLMResponse(
            text=f"{response.text}#{len(self.prompts)}",
            prompt_tokens=7,
            completion_tokens=3,
        )


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "llm_cache.jsonl"


def test_repeated_call_is_served_from_cache(cache_path):
    inner = CountingLLM(["a", "a"])
    client = CachingLLMClient(inner, cache_path)

    first = client.complete("same prompt")
    second = client.complete("same prompt")

    assert first.text == second.text
    assert len(inner.prompts) == 1  # the backend was called once
    assert (client.hits, client.misses) == (1, 1)


def test_cached_entries_replay_token_counts(cache_path):
    client = CachingLLMClient(CountingLLM(["a", "a"]), cache_path)

    client.complete("p")
    cached = client.complete("p")

    # CostLog must stay a real measurement even when the call was replayed.
    assert cached.prompt_tokens == 7
    assert cached.completion_tokens == 3


def test_cache_survives_a_new_process(cache_path):
    CachingLLMClient(CountingLLM(["a"]), cache_path).complete("p")

    fresh_backend = CountingLLM(["different"])
    reloaded = CachingLLMClient(fresh_backend, cache_path)
    response = reloaded.complete("p")

    assert response.text == "a#1"
    assert fresh_backend.prompts == []


def test_a_different_prompt_is_a_different_entry(cache_path):
    inner = CountingLLM(["a", "b"])
    client = CachingLLMClient(inner, cache_path)

    client.complete("prompt one")
    client.complete("prompt two")

    assert len(inner.prompts) == 2
    assert client.hits == 0


def test_a_changed_system_prompt_invalidates(cache_path):
    inner = CountingLLM(["a", "b"])
    client = CachingLLMClient(inner, cache_path)

    client.complete("p", system="grade strictly")
    client.complete("p", system="grade leniently")

    # Editing a prompt must not keep serving answers from the old wording.
    assert len(inner.prompts) == 2


def test_a_different_model_invalidates(cache_path):
    class ModelA(CountingLLM):
        @property
        def fingerprint(self):
            return "model-a"

    class ModelB(CountingLLM):
        @property
        def fingerprint(self):
            return "model-b"

    CachingLLMClient(ModelA(["a"]), cache_path).complete("p")

    backend_b = ModelB(["b"])
    response = CachingLLMClient(backend_b, cache_path).complete("p")

    assert backend_b.prompts == ["p"]
    assert response.text == "b#1"


def test_truncated_final_line_does_not_lose_the_cache(cache_path):
    client = CachingLLMClient(CountingLLM(["a"]), cache_path)
    client.complete("p")
    # Simulate a run killed mid-write, which is why entries are appended.
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "partial", "te')

    reloaded = CachingLLMClient(CountingLLM(["x"]), cache_path)

    assert reloaded.complete("p").text == "a#1"
