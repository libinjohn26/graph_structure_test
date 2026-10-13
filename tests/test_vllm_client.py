import httpx
import pytest

from gms.llm import VLLMClient


def chat_response(text: str, prompt_tokens: int = 11, completion_tokens: int = 7) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build(handler) -> VLLMClient:
    return VLLMClient(
        base_url="http://vllm.test:8000",
        model="qwen",
        transport=httpx.MockTransport(handler),
        max_retries=3,
    )


def test_complete_returns_text_and_real_usage_tokens():
    client = build(lambda request: httpx.Response(200, json=chat_response("hello", 11, 7)))

    response = client.complete("hi")

    assert response.text == "hello"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7
    assert response.total_tokens == 18


def test_system_prompt_is_sent_as_a_system_message():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=chat_response("ok"))

    build(handler).complete("extract this", system="you extract things")

    assert seen["messages"] == [
        {"role": "system", "content": "you extract things"},
        {"role": "user", "content": "extract this"},
    ]


def test_defaults_are_deterministic():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=chat_response("ok"))

    build(handler).complete("hi")

    # Reproducibility matters more than sampling quality for extraction runs.
    assert seen["temperature"] == 0.0
    assert seen["seed"] == 0


def test_thinking_toggle_is_omitted_unless_configured():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=chat_response("ok"))

    build(handler).complete("hi")

    # Models without a thinking mode reject this parameter with a 400.
    assert "chat_template_kwargs" not in seen


def test_thinking_can_be_disabled_explicitly():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=chat_response("ok"))

    VLLMClient(
        base_url="http://vllm.test:8000",
        model="qwen",
        enable_thinking=False,
        transport=httpx.MockTransport(handler),
    ).complete("hi")

    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_reasoning_and_truncation_are_surfaced():
    payload = {
        "choices": [
            {
                "message": {"content": None, "reasoning_content": "thinking out loud..."},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 512},
    }
    client = build(lambda request: httpx.Response(200, json=payload))

    response = client.complete("hi")

    # The reasoning-model trap: budget spent thinking, nothing emitted.
    assert response.text == ""
    assert response.reasoning == "thinking out loud..."
    assert response.truncated is True


def test_missing_usage_block_does_not_crash():
    payload = {"choices": [{"message": {"content": "hi"}}]}
    client = build(lambda request: httpx.Response(200, json=payload))

    response = client.complete("hi")

    assert response.total_tokens == 0


def test_null_content_is_treated_as_empty_text():
    payload = {"choices": [{"message": {"content": None}}], "usage": {}}
    client = build(lambda request: httpx.Response(200, json=payload))

    assert client.complete("hi").text == ""


def test_transient_server_errors_are_retried(monkeypatch):
    monkeypatch.setattr("gms.llm.time.sleep", lambda _: None)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=chat_response("recovered"))

    assert build(handler).complete("hi").text == "recovered"
    assert attempts["n"] == 3


def test_client_errors_are_not_retried():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(httpx.HTTPStatusError):
        build(handler).complete("hi")

    # A bad model name fails identically every time — retrying just wastes a run.
    assert attempts["n"] == 1


def test_persistent_failure_raises_rather_than_returning_empty(monkeypatch):
    monkeypatch.setattr("gms.llm.time.sleep", lambda _: None)
    client = build(lambda request: httpx.Response(500))

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        client.complete("hi")


@pytest.mark.parametrize(
    "base_url",
    ["http://vllm.test:8000", "http://vllm.test:8000/", "http://vllm.test:8000/v1", "http://vllm.test:8000/v1/"],
)
def test_endpoint_accepts_base_url_with_or_without_v1_suffix(base_url):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json=chat_response("ok"))

    VLLMClient(
        base_url=base_url, model="qwen", transport=httpx.MockTransport(handler)
    ).complete("hi")

    assert seen["path"] == "/v1/chat/completions"


def test_available_models_lists_served_ids():
    payload = {"data": [{"id": "Qwen/Qwen3-27B"}, {"id": "other"}]}
    client = build(lambda request: httpx.Response(200, json=payload))

    assert client.available_models() == ["Qwen/Qwen3-27B", "other"]


def test_from_env_requires_configuration(monkeypatch):
    monkeypatch.delenv("GMS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("GMS_LLM_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="GMS_LLM_BASE_URL"):
        VLLMClient.from_env()


def test_explicit_thinking_override_beats_the_environment(monkeypatch):
    monkeypatch.setenv("GMS_LLM_BASE_URL", "http://vllm.test:8000")
    monkeypatch.setenv("GMS_LLM_MODEL", "qwen")
    monkeypatch.setenv("GMS_LLM_ENABLE_THINKING", "false")

    # A second model (e.g. the judge) must not inherit the first model's thinking
    # setting — models without a thinking mode reject the parameter with a 400.
    client = VLLMClient.from_env(model="other-model", enable_thinking=None)

    assert client.enable_thinking is None


def test_from_env_reads_environment(monkeypatch):
    monkeypatch.setenv("GMS_LLM_BASE_URL", "http://vllm.test:8000")
    monkeypatch.setenv("GMS_LLM_MODEL", "qwen")

    client = VLLMClient.from_env()

    assert client.model == "qwen"
