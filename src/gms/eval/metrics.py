"""Scoring functions, per the metric stack described in the README.

Retrieval metrics (`evidence_recall`, `context_relevance`) work off the
retrieved context alone, so they score any architecture on the ladder without
a backbone LLM.

Generation metrics here are the **deterministic** half — `rouge_l` and
`cosine_similarity`, which feed the semantic-similarity term of Answer Accuracy.
The judged half lives in `judge.py`, which supplies factual correctness and
composes Answer Accuracy as `0.75·factual + 0.25·semantic`. Faithfulness and
Evidence Coverage are still unimplemented: no formula for them exists in-repo,
and inventing one would quietly become the published definition.

**Every metric here is per question.** The harness macro-averages them — an
unweighted mean over questions, not a pooled count — and chooses the subset per
metric (retrieval over all scored questions, generation over answerable ones,
refusal accuracy over unanswerable ones). See `gms.eval.harness.EvalResult`.

Every function is pure: embedding and generation happen in the harness, so
these stay testable without a model.

Both take sequences of *turn ids*: the ids an architecture reports as the
sources behind its context (`Retrieval.source_turn_ids`), scored against the
gold evidence ids the dataset annotates. Working in ids rather than text keeps
the comparison exact across rungs that return raw turns and rungs that return
distilled representations.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


def evidence_recall(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of the gold evidence that made it into the retrieved context.

    The headline retrieval metric: if the evidence never gets retrieved, no
    downstream generation can be right for the right reason. Caller guarantees
    `gold` is non-empty (the harness skips QA items with no resolvable
    evidence).
    """
    gold_set = set(gold)
    retrieved_set = set(retrieved)
    return len(gold_set & retrieved_set) / len(gold_set)


def context_relevance(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of the retrieved context that is actually gold evidence.

    The precision counterpart to recall — an architecture that dumps the whole
    conversation scores perfect recall and near-zero relevance, which is what
    separates "found it" from "buried it in noise".
    """
    if not retrieved:
        return 0.0
    gold_set = set(gold)
    return sum(1 for unit in retrieved if unit in gold_set) / len(retrieved)


def relevance_ceiling(n_retrieved: int, n_gold: int) -> float:
    """Best context relevance achievable given how much was returned.

    Precision is bounded by how much gold evidence exists: a question with one
    evidence turn, answered with five turns, cannot exceed 0.20 however perfect
    the retriever. Questions here average 1.27 gold turns against 4-5 returned,
    so raw relevance sits near this ceiling and is easy to misread as poor.

    It also makes raw relevance incomparable in two ways that matter: it falls
    as the budget grows (bigger denominator, near-fixed numerator), and it
    rewards an architecture for returning *fewer* units. Reporting relevance as
    a share of this ceiling removes both.
    """
    if n_retrieved <= 0:
        return 0.0
    return min(n_gold, n_retrieved) / n_retrieved


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length, two rows of DP.

    Quadratic, which is irrelevant here: LoCoMo answers run to a few words.
    """
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            current.append(
                previous[j] + 1 if token_a == token_b else max(current[j], previous[j + 1])
            )
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure over the longest common subsequence of word tokens.

    Implemented rather than pulled in as a dependency: it's a dozen lines, and
    a package here would be one more thing pinning the reported numbers to a
    version nobody records.
    """
    pred, ref = _tokenize(prediction), _tokenize(reference)
    if not pred or not ref:
        return 0.0

    lcs = _lcs_length(pred, ref)
    if not lcs:
        return 0.0

    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two embeddings, normalising defensively.

    Note when reading the numbers: a sentence embedder's cosine has a high
    floor — unrelated short strings routinely score 0.3-0.6 — so this is useful
    for ranking architectures against each other, not as an absolute "fraction
    correct".
    """
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0
