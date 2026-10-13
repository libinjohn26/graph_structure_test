"""Eval harness: run a memory architecture over conversations and score retrieval.

The harness owns the experiment loop, not the metrics: it feeds turns in order
into a fresh architecture per conversation, queries it once per QA item, and
records retrieval quality plus the architecture's own `CostLog`. Every
architecture on the ladder is driven through the same `add_turn`/`retrieve`
calls, so the comparison is apples-to-apples by construction.

Two data-quality rules matter for reading the numbers:

Scoring reads the `source_turn_ids` an architecture reports rather than
parsing its context text, so a rung that returns distilled attributes (L2) or
triples (L3) is scored on the same footing as one returning raw turns.

- **Unresolvable evidence.** ~0.3% of LoCoMo's evidence ids don't name a real
  turn (malformed like `"D8:6; D9:17"` or `"D"`, or dangling past the end of
  the conversation). They're counted and excluded rather than scored as misses,
  so a dataset bug doesn't silently penalise every architecture.
- **Out-of-scale QA.** When `max_turns` truncates a conversation for a scale
  point, QA items whose evidence lies past the cut are skipped, not scored —
  their answer simply isn't in memory yet, so scoring them would measure the
  truncation rather than the architecture.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from statistics import mean

from gms.architectures.base import CostLog, MemoryArchitecture
from gms.architectures.embeddings import Embedder
from gms.data.schema import Conversation, QAItem, Turn
from gms.eval.generation import generate_answer, is_refusal
from gms.eval.judge import answer_accuracy, judge_answer
from gms.eval.metrics import (
    context_relevance,
    cosine_similarity,
    evidence_recall,
    relevance_ceiling,
    rouge_l,
)
from gms.llm import LLMClient


@dataclass
class QAResult:
    """Retrieval scores for one QA item."""

    sample_id: str
    question: str
    category: int | str | None
    evidence_recall: float
    context_relevance: float
    relevance_ceiling: float
    n_gold: int
    n_retrieved: int

    # Generation, populated only when the harness was given an LLM.
    generated_answer: str | None = None
    rouge_l: float | None = None
    semantic_similarity: float | None = None
    # None when no judge ran, or when the judge returned an unreadable verdict —
    # which is excluded rather than scored as incorrect.
    factual_correctness: float | None = None
    # Driven by the gold answer being empty, not by the category label: two
    # LoCoMo items are tagged adversarial yet carry a real answer.
    expected_refusal: bool = False
    refused: bool = False


@dataclass
class EvalResult:
    """Everything one architecture produced over one run."""

    architecture: str
    conversations: int = 0
    turns_ingested: int = 0
    qa_skipped_out_of_scale: int = 0
    qa_skipped_no_evidence: int = 0
    unresolvable_evidence_refs: int = 0
    cost: CostLog = field(default_factory=CostLog)
    per_qa: list[QAResult] = field(default_factory=list)
    # Generation is the harness's cost, not the architecture's, so it stays out
    # of CostLog — which measures what a memory system costs to build and query.
    generation_tokens: int = 0
    generation_seconds: float = 0.0
    judge_tokens: int = 0
    judge_seconds: float = 0.0
    judge_unparseable: int = 0

    @property
    def qa_evaluated(self) -> int:
        return len(self.per_qa)

    def mean_evidence_recall(self) -> float:
        return mean(r.evidence_recall for r in self.per_qa) if self.per_qa else 0.0

    def mean_context_relevance(self) -> float:
        return mean(r.context_relevance for r in self.per_qa) if self.per_qa else 0.0

    def relevance_share(self) -> float:
        """Context relevance as a fraction of what was achievable.

        The comparable form. Raw relevance divides by units returned, so an
        architecture that surfaces fewer units scores higher for the same gold
        coverage — L2's attribute dedup does exactly that. Normalising by the
        per-question ceiling makes precision comparable across rungs that return
        different numbers of units, and across different budgets.

        **It also collapses to evidence recall whenever the gold evidence fits
        inside the budget**, which is ~99% of LoCoMo items (1.27 gold turns
        against 4-5 returned): share = |G∩R| / min(|G|,|R|), which is recall
        when |G| <= |R|. So report it as the comparable form of precision, but
        never alongside recall as if it were independent evidence. Precision
        only carries its own signal once an architecture is scored on the units
        it actually hands the generator — the open token-budget question.
        """
        scored = [
            r.context_relevance / r.relevance_ceiling
            for r in self.per_qa
            if r.relevance_ceiling > 0
        ]
        return mean(scored) if scored else 0.0

    def mean_relevance_ceiling(self) -> float:
        return mean(r.relevance_ceiling for r in self.per_qa) if self.per_qa else 0.0

    def mean_turns_returned(self) -> float:
        """Distinct source turns surfaced per query.

        The budget-comparability check: `top_k` counts an architecture's own
        retrieval units, which are turns for L0/L1 but attributes for L2 and
        triples for L3. When several units share a turn, the architecture
        surfaces fewer distinct turns than its k suggests and its recall is not
        comparable to a rung retrieving k whole turns.
        """
        return mean(r.n_retrieved for r in self.per_qa) if self.per_qa else 0.0

    @property
    def answerable(self) -> list[QAResult]:
        return [r for r in self.per_qa if not r.expected_refusal]

    @property
    def unanswerable(self) -> list[QAResult]:
        return [r for r in self.per_qa if r.expected_refusal]

    def mean_rouge_l(self) -> float:
        """ROUGE-L over answerable questions only — a refusal has no reference
        string to overlap with, so including them would score a *correct*
        refusal as zero."""
        scored = [r.rouge_l for r in self.answerable if r.rouge_l is not None]
        return mean(scored) if scored else 0.0

    def mean_semantic_similarity(self) -> float:
        scored = [r.semantic_similarity for r in self.answerable if r.semantic_similarity is not None]
        return mean(scored) if scored else 0.0

    def mean_factual_correctness(self) -> float:
        scored = [r.factual_correctness for r in self.answerable if r.factual_correctness is not None]
        return mean(scored) if scored else 0.0

    def mean_answer_accuracy(self) -> float | None:
        """Answer Accuracy over answerable questions.

        `None` when no judge ran: the formula is three-quarters factual
        correctness, so reporting the semantic-similarity term alone under this
        name would overstate what was measured.
        """
        scored = [
            answer_accuracy(r.factual_correctness, r.semantic_similarity or 0.0)
            for r in self.answerable
            if r.factual_correctness is not None
        ]
        return mean(scored) if scored else None

    def refusal_accuracy(self) -> float:
        """Share of unanswerable questions the model correctly declined."""
        items = self.unanswerable
        return sum(r.refused for r in items) / len(items) if items else 0.0

    def false_refusal_rate(self) -> float:
        """Share of *answerable* questions the model wrongly declined.

        Reported beside `refusal_accuracy` because that metric alone is gamed
        by refusing everything: a model that never answers scores a perfect
        1.0 on it. The pair has to be read together.
        """
        items = self.answerable
        return sum(r.refused for r in items) / len(items) if items else 0.0

    def by_category(self) -> dict[int | str | None, dict[str, float]]:
        """Scores grouped by the source dataset's raw category label.

        Kept raw (not remapped to this project's task taxonomy) for the same
        reason `QAItem.category` is — see `gms.data.locomo.LOCOMO_QA_CATEGORIES`
        for what LoCoMo's ints mean.
        """
        buckets: dict[int | str | None, list[QAResult]] = defaultdict(list)
        for result in self.per_qa:
            buckets[result.category].append(result)
        return {category: _category_stats(items) for category, items in buckets.items()}


def _category_stats(items: list[QAResult]) -> dict[str, float | None]:
    """Retrieval and generation scores for one QA category.

    Answerable and unanswerable questions are kept apart throughout: a category
    is scored on answer quality where there is a reference answer and on refusal
    behaviour where there isn't. Averaging the two together would make a correct
    refusal look like a zero-scoring answer. LoCoMo's categories are not cleanly
    one or the other — the adversarial bucket holds a couple of answerable items
    — so the split is driven per item, and either side can be `None` when a
    category has nothing of that kind.
    """
    answerable = [i for i in items if not i.expected_refusal]
    unanswerable = [i for i in items if i.expected_refusal]
    rouged = [i for i in answerable if i.rouge_l is not None]
    judged = [i for i in answerable if i.factual_correctness is not None]

    return {
        "n": len(items),
        "evidence_recall": mean(i.evidence_recall for i in items),
        "context_relevance": mean(i.context_relevance for i in items),
        "relevance_share": (
            mean(i.context_relevance / i.relevance_ceiling for i in items if i.relevance_ceiling > 0)
            if any(i.relevance_ceiling > 0 for i in items)
            else 0.0
        ),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "rouge_l": mean(i.rouge_l for i in rouged) if rouged else None,
        "factual_correctness": mean(i.factual_correctness for i in judged) if judged else None,
        "answer_accuracy": (
            mean(answer_accuracy(i.factual_correctness, i.semantic_similarity or 0.0) for i in judged)
            if judged
            else None
        ),
        "false_refusal_rate": (
            sum(i.refused for i in answerable) / len(answerable) if answerable else None
        ),
        "refusal_accuracy": (
            sum(i.refused for i in unanswerable) / len(unanswerable) if unanswerable else None
        ),
    }


def _accumulate_cost(total: CostLog, part: CostLog) -> None:
    for f in fields(CostLog):
        setattr(total, f.name, getattr(total, f.name) + getattr(part, f.name))


def _ordered_turns(conversation: Conversation) -> list[Turn]:
    return [turn for session in conversation.sessions for turn in session.turns]


def _score_generation(
    result: EvalResult,
    qa_result: QAResult,
    qa: QAItem,
    context: str,
    llm: LLMClient,
    embedder: Embedder | None,
    judge: LLMClient | None,
) -> None:
    start = time.perf_counter()
    response = generate_answer(llm, context, qa.question)
    result.generation_seconds += time.perf_counter() - start
    result.generation_tokens += response.total_tokens

    answer = response.text.strip()
    qa_result.generated_answer = answer
    qa_result.refused = is_refusal(answer)
    qa_result.expected_refusal = not qa.answer.strip()

    if qa_result.expected_refusal:
        # No reference string to overlap with; correctness is the refusal itself.
        return

    qa_result.rouge_l = rouge_l(answer, qa.answer)
    if embedder is not None:
        vectors = embedder.embed([answer or "", qa.answer])
        qa_result.semantic_similarity = cosine_similarity(vectors[0], vectors[1])

    if judge is not None:
        if qa_result.refused:
            # Declining a question that *has* an answer is incorrect, and that is
            # decidable here without a model. Judges rate a refusal CORRECT against
            # any reference — it contradicts nothing — which silently counted every
            # false refusal as a right answer.
            qa_result.factual_correctness = 0.0
            return

        start = time.perf_counter()
        verdict, judge_response = judge_answer(judge, qa.question, qa.answer, answer)
        result.judge_seconds += time.perf_counter() - start
        result.judge_tokens += judge_response.total_tokens
        if verdict is None:
            result.judge_unparseable += 1
        else:
            qa_result.factual_correctness = 1.0 if verdict else 0.0


def evaluate(
    name: str,
    build_memory: Callable[[], MemoryArchitecture],
    conversations: Iterable[Conversation],
    max_turns: int | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    judge: LLMClient | None = None,
) -> EvalResult:
    """Score one architecture over `conversations`.

    `build_memory` is a factory, not an instance: each conversation gets its own
    memory so nothing leaks between them, while expensive shared state (an
    `Embedder` and its loaded model) stays hoisted outside the factory.

    `max_turns` truncates each conversation to its first N turns, which is how
    the scale axis in configs/default.yaml gets swept.
    """
    result = EvalResult(architecture=name)

    for conversation in conversations:
        all_turns = _ordered_turns(conversation)
        ingested = all_turns if max_turns is None else all_turns[:max_turns]

        turn_by_id = {turn.dia_id: turn for turn in all_turns}
        ingested_ids = {turn.dia_id for turn in ingested}

        memory = build_memory()
        for turn in ingested:
            memory.add_turn(turn)

        result.conversations += 1
        result.turns_ingested += len(ingested)

        for qa in conversation.qa_items:
            gold_turns = []
            out_of_scale = False
            for ref in qa.evidence:
                turn = turn_by_id.get(ref)
                if turn is None:
                    result.unresolvable_evidence_refs += 1
                elif ref in ingested_ids:
                    gold_turns.append(turn)
                else:
                    out_of_scale = True

            if out_of_scale:
                result.qa_skipped_out_of_scale += 1
                continue
            if not gold_turns:
                result.qa_skipped_no_evidence += 1
                continue

            retrieval = memory.retrieve(qa.question)
            retrieved = retrieval.source_turn_ids
            gold = [turn.dia_id for turn in gold_turns]

            qa_result = QAResult(
                sample_id=conversation.sample_id,
                question=qa.question,
                category=qa.category,
                evidence_recall=evidence_recall(retrieved, gold),
                context_relevance=context_relevance(retrieved, gold),
                relevance_ceiling=relevance_ceiling(len(retrieved), len(set(gold))),
                # Distinct, matching what every metric above actually uses. One
                # LoCoMo item ("What are Dave's dreams?") annotates the same turn
                # twice, so the raw count would disagree with the scores and make
                # |G n R| non-integral for anyone re-deriving it from this field.
                n_gold=len(set(gold)),
                n_retrieved=len(retrieved),
            )
            if llm is not None:
                _score_generation(result, qa_result, qa, retrieval.context, llm, embedder, judge)

            result.per_qa.append(qa_result)

        _accumulate_cost(result.cost, memory.cost)

    return result
