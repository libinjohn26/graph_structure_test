"""LLM client interface, shared by every part of the project that calls a model.

Two callers, deliberately behind one interface:

- **Extraction** for the structured memory rungs (L2/L3/hybrid). This project fixes
  the extraction model across all of them so the experiment measures structure
  rather than extraction quality — one client instance passed to each
  architecture is how that control is enforced in code.
- **Generation and judging** in the eval harness, once that layer lands.

Architectures take an `LLMClient` by injection, so the backend stays swappable
and tests pass a scripted fake instead of hitting a server. `VLLMClient` talks
to a vLLM server over its OpenAI-compatible API.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


@dataclass
class LLMResponse:
    """One completion plus the token counts the cost axis needs."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning: str = ""
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        """Whether the model hit the token ceiling mid-answer.

        Worth checking explicitly on reasoning models: their thinking counts
        against `max_tokens`, so a budget that looks generous can be consumed
        entirely before any answer is emitted, returning empty text that looks
        like a refusal rather than a truncation.
        """
        return self.finish_reason == "length"


class LLMClient(ABC):
    """Minimal contract: prompt in, text and token counts out."""

    @property
    def fingerprint(self) -> str:
        """Identity of everything about this client that changes its output.

        Used as part of a cache key, so anything omitted here becomes a way to
        serve a stale response after a config change.
        """
        return type(self).__name__

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        """Run one completion. Implementations report real token counts where
        the backend returns them, since `CostLog` is a measured quantity in this
        project rather than an estimate."""


class VLLMClient(LLMClient):
    """A vLLM server, via its OpenAI-compatible `/v1/chat/completions` route.

    Defaults chosen for experiment reproducibility rather than output quality:
    `temperature=0` and an explicit `seed`, so re-running an architecture over
    the same conversation yields the same extractions. Raise the temperature
    only for a run that deliberately studies extraction variance.

    Token counts come from the server's `usage` block — vLLM reports what it
    actually processed, so the cost axis needs no tokenizer-side estimate.

    Transport errors raise after `max_retries` rather than being swallowed into
    `CostLog.failures`: a run of several thousand extractions against a dead
    server should stop loudly, not quietly produce a degraded memory that looks
    like a finding about the architecture.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        max_tokens: int = 512,
        enable_thinking: bool | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Accept the endpoint with or without the /v1 suffix — both forms get
        # pasted from vLLM docs and deployment notes, and the doubled path a
        # naive join produces ("/v1/v1/...") fails as an opaque 404.
        base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    @property
    def fingerprint(self) -> str:
        return (
            f"vllm|{self.model}|temp={self.temperature}|seed={self.seed}"
            f"|think={self.enable_thinking}|max_tokens={self.max_tokens}"
        )

    @classmethod
    def from_env(cls, **overrides) -> VLLMClient:
        """Build from GMS_LLM_BASE_URL / GMS_LLM_MODEL / GMS_LLM_API_KEY.

        Keeps a server address out of the committed config, so a machine's
        endpoint never becomes a hardcoded default someone else inherits.
        """
        base_url = overrides.pop("base_url", None) or os.environ.get("GMS_LLM_BASE_URL")
        model = overrides.pop("model", None) or os.environ.get("GMS_LLM_MODEL")
        if not base_url or not model:
            raise RuntimeError(
                "Set GMS_LLM_BASE_URL and GMS_LLM_MODEL (or pass base_url=/model=). "
                "Run `python scripts/check_llm.py` to list what the server is serving."
            )
        overrides.setdefault("api_key", os.environ.get("GMS_LLM_API_KEY"))
        if "enable_thinking" not in overrides:
            flag = os.environ.get("GMS_LLM_ENABLE_THINKING")
            if flag is not None:
                overrides["enable_thinking"] = flag.lower() not in {"0", "false", "no"}
        return cls(base_url=base_url, model=model, **overrides)

    def available_models(self) -> list[str]:
        """Model ids the server is actually serving, from `/v1/models`."""
        response = self._http.get("/v1/models")
        response.raise_for_status()
        return [entry["id"] for entry in response.json().get("data", [])]

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.enable_thinking is not None:
            # Only sent when explicitly configured: models without a thinking
            # mode reject this parameter outright with a 400.
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        data = self._post_with_retry("/v1/chat/completions", payload)
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=choice["message"].get("content") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning=choice["message"].get("reasoning_content") or "",
            finish_reason=choice.get("finish_reason") or "",
        )

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._http.post(path, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                # A 4xx is our bug (bad model name, malformed payload) and will
                # fail identically on every retry — only back off on transient ones.
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"vLLM request failed after {self.max_retries} attempts") from last_error


class CachingLLMClient(LLMClient):
    """Disk-backed cache in front of another client.

    Worth having because the expensive calls are deterministic by construction:
    extraction runs at `temperature=0` with a fixed seed, so re-running an eval
    re-derives identical results from 5,882 identical calls. Caching turns a
    ~64-minute L2 construction into seconds on the second run, and because the
    file is appended as it goes, a run killed by a gateway timeout keeps the
    work it had already done.

    **The cache key covers the client's fingerprint, the system prompt and the
    prompt itself**, so changing the model, temperature, seed, or any prompt
    wording produces different keys rather than silently serving responses from
    the old configuration. That is the whole correctness argument: a cache that
    outlives a prompt change would quietly invalidate every result after it.

    Cached entries replay the original token counts, so `CostLog` stays a real
    measurement. They cannot replay latency — a cached run's `*_seconds` are
    wall-clock of the replay, not of the work. Read token counts, not seconds,
    from any run that had cache hits (which the cost axis already prefers; see
    the README on wall-clock instability).
    """

    def __init__(self, inner: LLMClient, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, LLMResponse] = {}
        self._load()

    @property
    def fingerprint(self) -> str:
        return self.inner.fingerprint

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._entries[record["key"]] = LLMResponse(
                        text=record["text"],
                        prompt_tokens=record.get("prompt_tokens", 0),
                        completion_tokens=record.get("completion_tokens", 0),
                        reasoning=record.get("reasoning", ""),
                        finish_reason=record.get("finish_reason", ""),
                    )
                except (json.JSONDecodeError, KeyError):
                    # A truncated final line is expected when a previous run was
                    # killed mid-write; skip it rather than lose the whole cache.
                    continue

    def _key(self, prompt: str, system: str | None) -> str:
        payload = "\x00".join([self.fingerprint, system or "", prompt])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        key = self._key(prompt, system)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        response = self.inner.complete(prompt, system=system)
        self.misses += 1
        self._entries[key] = response

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, **asdict(response)}) + "\n")
        return response
