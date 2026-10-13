import numpy as np
import pytest

from gms.eval.generation import REFUSAL_SENTINEL, build_prompt, is_refusal
from gms.eval.metrics import cosine_similarity, rouge_l


# --- ROUGE-L -----------------------------------------------------------------


def test_identical_strings_score_one():
    assert rouge_l("Rex the dog", "Rex the dog") == 1.0


def test_no_overlap_scores_zero():
    assert rouge_l("completely unrelated", "Rex") == 0.0


def test_case_and_punctuation_are_ignored():
    assert rouge_l("7 May, 2023!", "7 may 2023") == 1.0


def test_subsequence_need_not_be_contiguous():
    # "the dog" survives as a subsequence despite the interruption.
    assert rouge_l("the brown dog", "the dog") == pytest.approx(2 * (2 / 3) * 1.0 / (2 / 3 + 1.0))


def test_verbosity_is_penalised():
    # The reason the prompt demands brevity: same facts, padded out, scores worse.
    terse = rouge_l("Rex", "Rex")
    verbose = rouge_l("I believe the dog is probably called Rex", "Rex")
    assert verbose < terse


def test_empty_strings_score_zero():
    assert rouge_l("", "Rex") == 0.0
    assert rouge_l("Rex", "") == 0.0


# --- cosine similarity -------------------------------------------------------


def test_cosine_of_identical_vectors_is_one():
    v = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_normalises_unnormalised_input():
    a = np.array([3.0, 0.0], dtype=np.float32)
    b = np.array([9.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_zero_vector_does_not_divide_by_zero():
    zero = np.zeros(3, dtype=np.float32)
    assert cosine_similarity(zero, np.ones(3, dtype=np.float32)) == 0.0


# --- refusal detection -------------------------------------------------------


def test_sentinel_is_detected():
    assert is_refusal(REFUSAL_SENTINEL)
    assert is_refusal(f"  {REFUSAL_SENTINEL}  ")
    assert is_refusal("not_answerable")


def test_a_real_answer_is_not_a_refusal():
    assert not is_refusal("Rex")
    assert not is_refusal("7 May 2023")


def test_prose_refusals_are_not_counted():
    # Detection is sentinel-only on purpose: fuzzy matching would hide a model
    # ignoring the instruction, which is a measurable compliance problem.
    assert not is_refusal("The context does not mention this.")


# --- prompt ------------------------------------------------------------------


def test_prompt_carries_context_and_question():
    prompt = build_prompt("Alice: I have a dog", "What pet?")
    assert "Alice: I have a dog" in prompt
    assert "What pet?" in prompt


def test_empty_context_is_marked_rather_than_blank():
    assert "(no context retrieved)" in build_prompt("", "What pet?")
