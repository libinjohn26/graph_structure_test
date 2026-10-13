"""LLM-judged factual correctness — the 0.75 term of Answer Accuracy.

Exists because the deterministic metrics systematically under-credit correct
answers: measured on real L1 generations, "trans woman" scores ROUGE-L 0.50
against "Transgender woman", and "national park" scores 0.50 against "National
park; she likes the outdoors". A judge grades meaning, which is what the
research question actually asks about.

**The judge is kept distinct from the generator.** `local-main` generates,
`local-critic` grades, which avoids self-grading bias — a model recognises its
own phrasing and inflates its own scores. An earlier configuration shared one
model for both and that warning used to live here; every judged run in
`data/processed` now records `generation_model` and `judge_model` separately in
its `config` block, so the split is checkable per run rather than asserted.

**Still unvalidated against humans.** No agreement study has been run, so the
judge's absolute calibration is unknown. Relative comparisons between
architectures are less exposed, since every rung is graded by the same judge.

The verdict is binary rather than graded. An LLM asked for a three-way or
numeric score is markedly noisier per item, and Answer Accuracy is a mean over
~2,000 questions, so granularity comes from the average rather than from each
judgement.
"""

from __future__ import annotations

import re

from gms.llm import LLMClient, LLMResponse

JUDGE_SYSTEM = (
    "You grade answers to questions about a conversation. Judge meaning only — "
    "ignore wording, capitalisation, formatting, and extra detail that does not "
    "contradict the reference. An answer that does not state the reference's "
    "information is INCORRECT, including one that declines to answer or says the "
    "information is unavailable. Reply with exactly one word: CORRECT or INCORRECT."
)

JUDGE_PROMPT = """Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Is the answer to grade correct?"""

# INCORRECT contains CORRECT, so a naive substring test silently grades every
# wrong answer as right. Anchor on word boundaries and check the negative first.
_INCORRECT_RE = re.compile(r"\bincorrect\b", re.IGNORECASE)
_CORRECT_RE = re.compile(r"\bcorrect\b", re.IGNORECASE)


def parse_verdict(text: str) -> bool | None:
    """True/False for a readable verdict, None when the judge didn't comply."""
    if _INCORRECT_RE.search(text):
        return False
    if _CORRECT_RE.search(text):
        return True
    return None


def judge_answer(
    llm: LLMClient, question: str, reference: str, answer: str
) -> tuple[bool | None, LLMResponse]:
    """Grade one answer. Returns (verdict, response) so the caller can bill the
    tokens and count non-compliant verdicts rather than scoring them as wrong."""
    response = llm.complete(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer),
        system=JUDGE_SYSTEM,
    )
    return parse_verdict(response.text), response


def answer_accuracy(factual_correctness: float, semantic_similarity: float) -> float:
    """Answer Accuracy: 0.75·factual-correctness + 0.25·semantic-sim."""
    return 0.75 * factual_correctness + 0.25 * semantic_similarity
