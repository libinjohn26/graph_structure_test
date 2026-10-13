"""Tests for the paired bootstrap behind every interval in the paper.

A bootstrap that is subtly wrong does not crash — it quietly produces intervals
of the wrong width, which is exactly the failure that would let an overclaim
through. So these check calibration against known effects rather than just
exercising the code path.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "bootstrap_ci", REPO_ROOT / "scripts" / "bootstrap_ci.py"
)
bootstrap_ci = importlib.util.module_from_spec(_spec)
sys.modules["bootstrap_ci"] = bootstrap_ci
_spec.loader.exec_module(bootstrap_ci)


def make_grouped(pairs_by_conversation):
    grouped = {name: list(pairs) for name, pairs in pairs_by_conversation.items()}
    return sorted(grouped), grouped


def run(pairs_by_conversation, unit="question", seed=0):
    conversations, grouped = make_grouped(pairs_by_conversation)
    return bootstrap_ci.bootstrap(
        grouped, conversations, unit, np.random.default_rng(seed)
    )


def test_identical_scores_give_zero_difference_and_zero_width():
    """No difference anywhere means the interval must collapse onto zero."""
    pairs = [(0.5, 0.5)] * 50
    result = run({"conv-1": pairs})
    assert result["difference"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert not result["significant"]


def test_large_consistent_effect_is_significant():
    pairs = [(0.2, 0.8)] * 200
    result = run({"conv-1": pairs})
    assert result["difference"] == pytest.approx(0.6)
    assert result["significant"]
    assert result["ci_low"] > 0


def test_false_positive_rate_is_near_five_percent():
    """Calibration: with no true effect, a 95% interval should wrongly exclude
    zero about 5% of the time — not never, and not often.

    Deliberately checked across many samples rather than one. A single draw from
    a zero-effect distribution lands outside its own interval 5% of the time by
    construction, so asserting "this one sample is not significant" tests the
    seed, not the estimator. (The first version of this test did exactly that
    and failed on a draw that sat 2.55 standard errors from zero — which is the
    same mistake, in miniature, as reading a result off a 4-question gap.)
    """
    false_positives = 0
    trials = 60
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        left = rng.normal(0.5, 0.2, 200)
        pairs = list(zip(left, left + rng.normal(0.0, 0.2, 200)))
        if run({"conv-1": pairs}, seed=seed)["significant"]:
            false_positives += 1
    assert false_positives <= trials * 0.15, (
        f"{false_positives}/{trials} false positives — intervals are too narrow"
    )


def test_interval_brackets_the_observed_difference():
    rng = np.random.default_rng(3)
    pairs = [(float(a), float(a + 0.1)) for a in rng.normal(0.5, 0.3, 400)]
    result = run({"conv-1": pairs})
    assert result["ci_low"] <= result["difference"] <= result["ci_high"]


def test_conversation_unit_is_wider_when_conversations_differ():
    """The whole reason for the second resampling unit.

    Questions inside a conversation share a graph, so a difference that is
    consistent within conversations but varies between them is less certain than
    per-question resampling implies. Resampling conversations must reflect that.
    """
    by_conversation = {
        f"conv-{i}": [(0.5, 0.5 + delta)] * 40
        for i, delta in enumerate([0.0, 0.05, 0.10, 0.30, 0.45, -0.10])
    }
    per_question = run(by_conversation, unit="question")
    per_conversation = run(by_conversation, unit="conversation")

    assert per_question["difference"] == pytest.approx(per_conversation["difference"])
    question_width = per_question["ci_high"] - per_question["ci_low"]
    conversation_width = per_conversation["ci_high"] - per_conversation["ci_low"]
    assert conversation_width > question_width


def test_pairing_drops_questions_missing_on_either_side():
    """An unscored question on one side must remove it from both, never compare
    a scored question against a missing one."""
    left = [
        {"sample_id": "c", "question": "q1", "evidence_recall": 1.0},
        {"sample_id": "c", "question": "q2", "evidence_recall": 1.0},
        {"sample_id": "c", "question": "q3", "evidence_recall": 0.0},
    ]
    right = [
        {"sample_id": "c", "question": "q1", "evidence_recall": 0.0},
        {"sample_id": "c", "question": "q2", "evidence_recall": None},
        {"sample_id": "c", "question": "q4", "evidence_recall": 1.0},
    ]
    _, grouped = bootstrap_ci.paired_units(left, right, "evidence_recall", False)
    assert [p for pairs in grouped.values() for p in pairs] == [(1.0, 0.0)]


def test_answerable_only_excludes_expected_refusals():
    left = [
        {"sample_id": "c", "question": "q1", "factual_correctness": 1.0},
        {"sample_id": "c", "question": "q2", "factual_correctness": 1.0,
         "expected_refusal": True},
    ]
    right = [
        {"sample_id": "c", "question": "q1", "factual_correctness": 0.0},
        {"sample_id": "c", "question": "q2", "factual_correctness": 0.0,
         "expected_refusal": True},
    ]
    _, grouped = bootstrap_ci.paired_units(left, right, "factual_correctness", True)
    assert [p for pairs in grouped.values() for p in pairs] == [(1.0, 0.0)]


def test_answer_accuracy_is_derived_not_read():
    record = {"factual_correctness": 1.0, "semantic_similarity": 0.4}
    assert bootstrap_ci.metric_value(record, None) == pytest.approx(0.85)


def test_answer_accuracy_is_none_without_a_judgement():
    """Three quarters of the formula is factual correctness, so without the judge
    there is no Answer Accuracy to report — not a score of one quarter."""
    assert bootstrap_ci.metric_value({"semantic_similarity": 0.8}, None) is None


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    pairs = [(0.3, 0.45)] * 120
    first = run({"conv-1": pairs}, seed=11)
    second = run({"conv-1": pairs}, seed=11)
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]


def test_empty_input_reports_no_data_rather_than_crashing():
    assert run({})["n"] == 0


def test_repeated_question_text_is_not_collapsed():
    """LoCoMo repeats some question strings inside one conversation.

    Keying on (sample_id, question) silently dropped one row of each repeated
    pair from both sides — 12 of 1,977 questions at full scale. The counts stayed
    plausible and no error was raised, so this is exactly the kind of defect that
    only a count assertion catches.
    """
    left = [
        {"sample_id": "c", "question": "same?", "evidence_recall": 1.0},
        {"sample_id": "c", "question": "same?", "evidence_recall": 0.0},
        {"sample_id": "c", "question": "other?", "evidence_recall": 1.0},
    ]
    right = [
        {"sample_id": "c", "question": "same?", "evidence_recall": 0.5},
        {"sample_id": "c", "question": "same?", "evidence_recall": 0.25},
        {"sample_id": "c", "question": "other?", "evidence_recall": 1.0},
    ]
    _, grouped = bootstrap_ci.paired_units(left, right, "evidence_recall", False)
    pairs = sorted(p for pairs in grouped.values() for p in pairs)

    assert len(pairs) == 3, "a repeated question string must not drop a row"
    # Order is preserved per side, so the nth occurrence pairs with the nth.
    assert pairs == [(0.0, 0.25), (1.0, 0.5), (1.0, 1.0)]
