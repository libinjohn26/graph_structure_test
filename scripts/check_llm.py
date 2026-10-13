#!/usr/bin/env python
"""Verify the vLLM server is reachable and behaving before a long run.

Checks, in order:
  1. The server answers `/v1/models` and reports what it's serving.
  2. The configured model is actually in that list.
  3. A real completion round-trips, returning text and usage token counts.
  4. The extraction prompt L2 uses returns parseable JSON from this model.

Step 4 is the one worth running after any model change: L2 stores whatever the
extraction returns, so a model that ignores the JSON instruction degrades the
architecture silently rather than erroring.

Usage:
    export GMS_LLM_BASE_URL=http://host:8000
    export GMS_LLM_MODEL=<id from step 1>
    python scripts/check_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import os  # noqa: E402
import time  # noqa: E402

from gms.architectures.kv_entity import _EXTRACTION_PROMPT, _EXTRACTION_SYSTEM, _parse_entities  # noqa: E402
from gms.eval.judge import judge_answer  # noqa: E402
from gms.llm import VLLMClient  # noqa: E402
from gms.utils.env import load_env_file  # noqa: E402

SAMPLE_TURN = "Caroline: I moved here from Sweden four years ago and I'm training to be a counselor."


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def main() -> None:
    loaded = load_env_file()
    if loaded:
        print(f"Loaded from .env: {', '.join(loaded)}\n")

    base_url = os.environ.get("GMS_LLM_BASE_URL")
    if not base_url:
        fail("GMS_LLM_BASE_URL is not set (e.g. http://10.0.0.5:8000).")

    model = os.environ.get("GMS_LLM_MODEL")
    print(f"Checking vLLM at {base_url}...\n")

    # Model listing doesn't need a valid model name, so probe with a placeholder.
    probe = VLLMClient(base_url=base_url, model=model or "unknown", api_key=os.environ.get("GMS_LLM_API_KEY"))

    try:
        served = probe.available_models()
    except Exception as exc:  # noqa: BLE001 — surface the real reason, whatever it is
        fail(f"Could not reach {base_url}/v1/models: {exc!r}")

    if not served:
        fail("Server responded but is serving no models.")
    ok(f"Server reachable. Serving: {served}")

    if not model:
        fail(f"GMS_LLM_MODEL is not set. Pick one of: {served}")
    if model not in served:
        fail(f"GMS_LLM_MODEL={model!r} is not served. Available: {served}")
    ok(f"Configured model {model!r} is being served")

    client = VLLMClient.from_env()

    start = time.perf_counter()
    try:
        response = client.complete("Reply with exactly the word: ready")
    except Exception as exc:  # noqa: BLE001
        fail(f"Completion request failed: {exc!r}")
    elapsed = time.perf_counter() - start

    if not response.text.strip():
        fail("Completion returned empty text.")
    ok(f"Completion round-trip: {response.text.strip()[:60]!r} in {elapsed:.1f}s")

    # Latency on this shared server has swung by two orders of magnitude between
    # sessions, and a long run that meets a degraded server dies on a 504 with no
    # checkpoint. Cheap to check first; expensive to discover an hour in.
    if elapsed > 10:
        print(f"[WARN] A trivial completion took {elapsed:.0f}s — the server is heavily loaded.")
        print("       A full sweep would likely fail on a gateway timeout. Wait it out.")
    elif elapsed > 3:
        print(f"[WARN] A trivial completion took {elapsed:.1f}s — slower than usual (~0.3s).")

    if response.total_tokens == 0:
        print("[WARN] Server reported no usage tokens — CostLog will undercount.")
    else:
        ok(f"Usage reported: {response.prompt_tokens} prompt + {response.completion_tokens} completion tokens")

    if response.reasoning:
        print(f"[INFO] Reasoning model: {len(response.reasoning)} chars of hidden reasoning, "
              f"billed inside completion_tokens.")
        if os.environ.get("GMS_LLM_ENABLE_THINKING", "").lower() not in {"0", "false"}:
            print("       Extraction is mechanical work; thinking multiplies its token cost and "
                  "latency for no gain. Consider GMS_LLM_ENABLE_THINKING=false.")

    extraction = client.complete(
        _EXTRACTION_PROMPT.format(turn=SAMPLE_TURN), system=_EXTRACTION_SYSTEM
    )
    entities = _parse_entities(extraction.text)

    if entities is None:
        print("\n[WARN] L2's extraction prompt did not return usable JSON from this model.")
        if extraction.truncated:
            print(f"       finish_reason='length' — the {extraction.completion_tokens}-token budget "
                  "ran out before an answer was emitted.")
            if extraction.reasoning:
                print("       The budget went on reasoning. Disable thinking or raise max_tokens.")
        print(f"       Raw response: {extraction.text[:300]!r}")
        print("       L2 would record every such turn as an extraction failure.")
    elif not entities:
        print("\n[WARN] Extraction returned valid JSON but found no facts in a fact-bearing turn.")
        print(f"       Raw response: {extraction.text[:300]!r}")
    else:
        ok(f"L2 extraction prompt returns usable JSON: {entities}")

    check_judge(served, model)

    print("\nAll checks passed.")


def check_judge(served: list[str], generation_model: str) -> None:
    """The judge is a second model with its own quirks — verify it separately.

    Worth its own check because the thinking toggle is per-model: a judge that
    inherits the generation model's `enable_thinking` setting fails with an
    opaque 400, and only after a long run has already started.
    """
    judge_model = os.environ.get("GMS_JUDGE_MODEL")
    if not judge_model:
        print("\n[INFO] GMS_JUDGE_MODEL unset — judging would reuse the generation model, "
              "which is self-grading.")
        return

    print(f"\nChecking judge model {judge_model!r}...")
    if judge_model not in served:
        fail(f"GMS_JUDGE_MODEL={judge_model!r} is not served. Available: {served}")
    if judge_model == generation_model:
        print("[WARN] Judge and generation model are the same — self-grading inflates "
              "Answer Accuracy.")

    thinking = os.environ.get("GMS_JUDGE_ENABLE_THINKING")
    judge = VLLMClient.from_env(
        model=judge_model,
        enable_thinking=(None if thinking is None else thinking.lower() not in {"0", "false", "no"}),
    )

    try:
        verdict, response = judge_answer(judge, "What pet does Alice have?", "a dog", "a dog")
    except Exception as exc:  # noqa: BLE001
        fail(f"Judge request failed: {exc!r}. If this is a 400, the model likely rejects "
             f"chat_template_kwargs — unset GMS_JUDGE_ENABLE_THINKING.")

    if verdict is None:
        print(f"[WARN] Judge did not return a readable verdict: {response.text[:200]!r}")
        print("       Those judgements are excluded, so Answer Accuracy would rest on fewer items.")
    elif verdict is True:
        ok(f"Judge grades a correct answer as correct ({response.text.strip()[:40]!r})")
    else:
        print("[WARN] Judge graded an obviously correct answer as INCORRECT — check the prompt.")


if __name__ == "__main__":
    main()
