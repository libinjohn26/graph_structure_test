import pytest
from fakes import ScriptedLLM, StableHashEmbedder

from gms.architectures.base import MemoryArchitecture, Retrieval
from gms.architectures.flat import FlatRecencyMemory
from gms.data.schema import Conversation, QAItem, Session, Turn
from gms.eval.generation import REFUSAL_SENTINEL
from gms.eval.harness import evaluate


def turn(dia_id: str, speaker: str, text: str) -> Turn:
    return Turn(dia_id=dia_id, speaker=speaker, text=text)


@pytest.fixture
def conversation():
    """Four turns across two sessions, with QA pointing at known evidence."""
    return Conversation(
        sample_id="conv-test",
        speaker_a="Alice",
        speaker_b="Bob",
        sessions=[
            Session(
                session_id="session_1",
                date_time="noon",
                turns=[turn("D1:1", "Alice", "I got a dog"), turn("D1:2", "Bob", "nice")],
            ),
            Session(
                session_id="session_2",
                date_time="later",
                turns=[turn("D2:1", "Alice", "his name is Rex"), turn("D2:2", "Bob", "cute")],
            ),
        ],
        qa_items=[
            QAItem(question="dog name?", answer="Rex", category=1, evidence=["D2:1"]),
        ],
    )


def test_scores_retrieval_against_gold_evidence(conversation):
    # window=4 returns every turn, so the gold evidence is always present.
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation])

    assert result.qa_evaluated == 1
    assert result.mean_evidence_recall() == 1.0
    # One of four returned turns is gold evidence.
    assert result.mean_context_relevance() == 0.25


def test_recency_window_that_misses_evidence_scores_zero(conversation):
    conversation.qa_items = [
        QAItem(question="dog name?", answer="Rex", category=1, evidence=["D1:1"])
    ]
    # window=1 returns only the final turn, which is not the evidence turn.
    result = evaluate("L0", lambda: FlatRecencyMemory(window=1), [conversation])

    assert result.mean_evidence_recall() == 0.0
    assert result.mean_context_relevance() == 0.0


def test_unresolvable_evidence_is_counted_not_scored(conversation):
    conversation.qa_items = [
        QAItem(question="q", answer="a", category=1, evidence=["D2:1", "D9:99"]),
    ]
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation])

    assert result.unresolvable_evidence_refs == 1
    # Scored on the one resolvable ref rather than penalised for the bad one.
    assert result.qa_evaluated == 1
    assert result.mean_evidence_recall() == 1.0


def test_qa_with_no_resolvable_evidence_is_skipped(conversation):
    conversation.qa_items = [QAItem(question="q", answer="a", category=1, evidence=[])]
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation])

    assert result.qa_evaluated == 0
    assert result.qa_skipped_no_evidence == 1


def test_max_turns_truncates_and_skips_out_of_scale_qa(conversation):
    # Only the first two turns are ingested, so D2:1 evidence is out of range.
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation], max_turns=2)

    assert result.turns_ingested == 2
    assert result.qa_evaluated == 0
    assert result.qa_skipped_out_of_scale == 1


def test_memory_is_rebuilt_per_conversation(conversation):
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation, conversation])

    # Cost accumulates across both runs, but neither saw the other's turns.
    assert result.conversations == 2
    assert result.turns_ingested == 8
    assert result.cost.storage_entries == 8
    assert result.cost.queries == 2
    assert result.mean_context_relevance() == 0.25


class ParaphrasingMemory(MemoryArchitecture):
    """Returns context that quotes no turn verbatim, like L2/L3 do."""

    def __init__(self) -> None:
        super().__init__()
        self._ids: list[str] = []

    def add_turn(self, turn: Turn) -> None:
        self._ids.append(turn.dia_id)

    def retrieve(self, query: str) -> Retrieval:
        return Retrieval(context="a distilled fact in the model's own words", source_turn_ids=self._ids)


def test_scores_architectures_whose_context_quotes_no_turn(conversation):
    # Guards the provenance contract: scoring reads source_turn_ids, so a rung
    # returning distilled attributes isn't scored 0.0 for structural reasons.
    result = evaluate("paraphrasing", ParaphrasingMemory, [conversation])

    assert result.qa_evaluated == 1
    assert result.mean_evidence_recall() == 1.0


def test_generation_scores_answerable_questions(conversation):
    llm = ScriptedLLM(["Rex"])
    result = evaluate(
        "L0", lambda: FlatRecencyMemory(window=4), [conversation],
        llm=llm, embedder=StableHashEmbedder(),
    )

    scored = result.per_qa[0]
    assert scored.generated_answer == "Rex"
    assert scored.expected_refusal is False
    assert scored.refused is False
    assert scored.rouge_l == 1.0
    assert scored.semantic_similarity == pytest.approx(1.0)
    assert result.generation_tokens > 0


def test_unanswerable_question_is_scored_on_refusal_not_overlap(conversation):
    # LoCoMo's adversarial items omit the answer entirely.
    conversation.qa_items = [
        QAItem(question="what did Bob realise?", answer="", category=5, evidence=["D1:1"])
    ]
    llm = ScriptedLLM([REFUSAL_SENTINEL])
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation], llm=llm)

    scored = result.per_qa[0]
    assert scored.expected_refusal is True
    assert scored.refused is True
    # No reference string exists, so overlap metrics are not computed at all.
    assert scored.rouge_l is None
    assert result.refusal_accuracy() == 1.0


def test_refusal_accuracy_and_false_refusals_are_reported_separately(conversation):
    conversation.qa_items = [
        QAItem(question="answerable", answer="Rex", category=4, evidence=["D1:1"]),
        QAItem(question="unanswerable", answer="", category=5, evidence=["D1:2"]),
    ]
    # A model that refuses everything: perfect on refusals, useless overall.
    llm = ScriptedLLM([REFUSAL_SENTINEL, REFUSAL_SENTINEL])
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation], llm=llm)

    assert result.refusal_accuracy() == 1.0
    assert result.false_refusal_rate() == 1.0  # which is what exposes it


def test_rouge_averages_exclude_unanswerable_items(conversation):
    conversation.qa_items = [
        QAItem(question="answerable", answer="Rex", category=4, evidence=["D1:1"]),
        QAItem(question="unanswerable", answer="", category=5, evidence=["D1:2"]),
    ]
    llm = ScriptedLLM(["Rex", REFUSAL_SENTINEL])
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation], llm=llm)

    # A correct refusal must not drag the mean down as if it were a 0.0 answer.
    assert result.mean_rouge_l() == 1.0


def test_no_generation_without_an_llm(conversation):
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation])

    assert result.per_qa[0].generated_answer is None
    assert result.generation_tokens == 0


def test_by_category_groups_raw_labels(conversation):
    conversation.qa_items = [
        QAItem(question="q1", answer="a", category=1, evidence=["D2:1"]),
        QAItem(question="q2", answer="a", category=4, evidence=["D1:1"]),
        QAItem(question="q3", answer="a", category=4, evidence=["D1:2"]),
    ]
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation])

    by_cat = result.by_category()
    assert by_cat[1]["n"] == 1
    assert by_cat[4]["n"] == 2


def test_by_category_splits_answerable_from_unanswerable(conversation):
    conversation.qa_items = [
        QAItem(question="answerable", answer="Rex", category=4, evidence=["D1:1"]),
        QAItem(question="trick", answer="", category=5, evidence=["D1:2"]),
    ]
    result = evaluate(
        "L0", lambda: FlatRecencyMemory(window=4), [conversation],
        llm=ScriptedLLM(["Rex", REFUSAL_SENTINEL]), embedder=StableHashEmbedder(),
    )

    answerable_cat = result.by_category()[4]
    assert answerable_cat["n_answerable"] == 1
    assert answerable_cat["rouge_l"] == 1.0
    # No unanswerable items here, so refusal accuracy is undefined, not zero.
    assert answerable_cat["refusal_accuracy"] is None

    adversarial_cat = result.by_category()[5]
    assert adversarial_cat["n_unanswerable"] == 1
    assert adversarial_cat["refusal_accuracy"] == 1.0
    # Correspondingly, no answer-quality score for a category with no references.
    assert adversarial_cat["rouge_l"] is None
    assert adversarial_cat["answer_accuracy"] is None


def test_category_generation_stats_are_none_without_generation(conversation):
    result = evaluate("L0", lambda: FlatRecencyMemory(window=4), [conversation])

    stats = result.by_category()[1]
    assert stats["rouge_l"] is None
    assert stats["answer_accuracy"] is None
    assert stats["evidence_recall"] == 1.0  # retrieval still scored


def test_repeated_gold_reference_is_counted_once():
    """LoCoMo annotates the same evidence turn twice on one item.

    Every metric already de-duplicates, but `n_gold` reported the raw count, so
    `evidence_recall * n_gold` — the obvious way to re-derive the number of gold
    turns retrieved — came out at 1.5 for that question. The reported field must
    agree with what the metrics used.
    """
    from gms.architectures.flat import FlatRecencyMemory
    from gms.data.schema import Conversation, QAItem, Session, Turn
    from gms.eval.harness import evaluate

    turns = [Turn(dia_id=f"t{i}", speaker="S", text=f"line {i}") for i in range(3)]
    conversation = Conversation(
        sample_id="c",
        speaker_a="S",
        speaker_b="T",
        sessions=[Session(session_id="s", date_time="", turns=turns)],
        qa_items=[QAItem(question="q?", answer="a", category=1,
                         evidence=["t0", "t0", "t1"], sample_id="c")],
    )

    result = evaluate("L0", lambda: FlatRecencyMemory(window=3), [conversation])
    row = result.per_qa[0]

    assert row.n_gold == 2, "a repeated evidence ref must count once"
    hits = row.evidence_recall * row.n_gold
    assert abs(hits - round(hits)) < 1e-9, "|G and R| must be a whole number of turns"
    assert hits == 2


def test_relevance_and_recall_agree_on_the_intersection():
    """|G n R| is derivable from recall and from relevance; the two must match.

    They divide by different denominators, so a de-duplication bug on either
    side shows up here and nowhere else.
    """
    from gms.architectures.flat import FlatRecencyMemory
    from gms.data.schema import Conversation, QAItem, Session, Turn
    from gms.eval.harness import evaluate

    turns = [Turn(dia_id=f"t{i}", speaker="S", text=f"line {i}") for i in range(5)]
    conversation = Conversation(
        sample_id="c", speaker_a="S", speaker_b="T",
        sessions=[Session(session_id="s", date_time="", turns=turns)],
        qa_items=[QAItem(question="q?", answer="a", category=1,
                         evidence=["t3", "t4"], sample_id="c")],
    )
    result = evaluate("L0", lambda: FlatRecencyMemory(window=3), [conversation])
    row = result.per_qa[0]

    assert row.evidence_recall * row.n_gold == row.context_relevance * row.n_retrieved
