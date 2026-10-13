import pytest

from gms.eval.metrics import context_relevance, evidence_recall, relevance_ceiling


def test_evidence_recall_perfect():
    assert evidence_recall(["a", "b", "c"], ["a", "b"]) == 1.0


def test_evidence_recall_partial():
    assert evidence_recall(["a", "x"], ["a", "b"]) == 0.5


def test_evidence_recall_zero():
    assert evidence_recall(["x", "y"], ["a", "b"]) == 0.0


def test_context_relevance_is_precision_not_recall():
    # Everything gold was retrieved, but buried in noise: recall 1.0, relevance 0.25.
    retrieved = ["a", "x", "y", "z"]
    gold = ["a"]
    assert evidence_recall(retrieved, gold) == 1.0
    assert context_relevance(retrieved, gold) == 0.25


def test_context_relevance_empty_context():
    assert context_relevance([], ["a"]) == 0.0


def test_duplicate_gold_refs_counted_once():
    assert evidence_recall(["a"], ["a", "a"]) == 1.0


def test_relevance_ceiling_is_bounded_by_available_gold():
    # One gold turn, five returned: perfect retrieval still scores 0.20.
    assert relevance_ceiling(n_retrieved=5, n_gold=1) == pytest.approx(0.2)
    assert relevance_ceiling(n_retrieved=4, n_gold=1) == pytest.approx(0.25)
    # Enough gold to fill the budget means a perfect score is reachable.
    assert relevance_ceiling(n_retrieved=5, n_gold=5) == 1.0
    assert relevance_ceiling(n_retrieved=5, n_gold=9) == 1.0


def test_relevance_ceiling_handles_empty_retrieval():
    assert relevance_ceiling(n_retrieved=0, n_gold=3) == 0.0


def test_ceiling_makes_fewer_returned_units_a_non_advantage():
    """Why the share is reported instead of raw relevance.

    An architecture that dedups five units down to two scores far higher on raw
    precision for identical gold coverage. Normalising by the ceiling removes
    that, which is what makes L2 comparable with the turn-returning rungs.
    """
    gold = ["a", "b"]
    few_raw = context_relevance(["a", "b"], gold)          # 1.00 over 2 units
    many_raw = context_relevance(["a", "b", "x", "y", "z"], gold)  # 0.40 over 5

    assert few_raw > many_raw

    few_share = few_raw / relevance_ceiling(2, len(gold))
    many_share = many_raw / relevance_ceiling(5, len(gold))

    assert few_share == pytest.approx(many_share)
