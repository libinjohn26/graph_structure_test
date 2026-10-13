import pytest
from fakes import ScriptedLLM, StableHashEmbedder

from gms.architectures.flat import FlatRecencyMemory
from gms.data.schema import Conversation, QAItem, Session, Turn
from gms.eval.generation import REFUSAL_SENTINEL
from gms.eval.harness import evaluate
from gms.eval.judge import answer_accuracy, judge_answer, parse_verdict


# --- verdict parsing ---------------------------------------------------------


@pytest.mark.parametrize("raw", ["CORRECT", "correct", "  Correct.  ", "The answer is correct"])
def test_positive_verdicts(raw):
    assert parse_verdict(raw) is True


@pytest.mark.parametrize("raw", ["INCORRECT", "incorrect", "The answer is incorrect."])
def test_negative_verdicts(raw):
    # The trap: "INCORRECT" contains "CORRECT", so a substring test grades every
    # wrong answer as right.
    assert parse_verdict(raw) is False


@pytest.mark.parametrize("raw", ["", "maybe", "I cannot tell", "42"])
def test_non_compliant_verdicts_are_none_not_false(raw):
    # Scoring an unreadable verdict as incorrect would silently depress accuracy
    # for a judge that simply ignored the format.
    assert parse_verdict(raw) is None


def test_judge_answer_returns_verdict_and_billable_response():
    llm = ScriptedLLM(["CORRECT"])

    verdict, response = judge_answer(llm, "What pet?", "a dog", "a dog")

    assert verdict is True
    assert response.total_tokens > 0
    assert "a dog" in llm.prompts[0]


# --- the composite -----------------------------------------------------------


def test_answer_accuracy_weights_match_the_spec():
    assert answer_accuracy(1.0, 1.0) == pytest.approx(1.0)
    assert answer_accuracy(1.0, 0.0) == pytest.approx(0.75)
    assert answer_accuracy(0.0, 1.0) == pytest.approx(0.25)


# --- harness integration -----------------------------------------------------


@pytest.fixture
def conversation():
    return Conversation(
        sample_id="c1",
        speaker_a="Alice",
        speaker_b="Bob",
        sessions=[
            Session(
                session_id="session_1",
                date_time="noon",
                turns=[Turn(dia_id="D1:1", speaker="Alice", text="I have a dog named Rex")],
            )
        ],
        qa_items=[QAItem(question="pet name?", answer="Rex", category=4, evidence=["D1:1"])],
    )


def run(conversation, answers, verdicts):
    return evaluate(
        "L0",
        lambda: FlatRecencyMemory(window=4),
        [conversation],
        llm=ScriptedLLM(answers),
        embedder=StableHashEmbedder(),
        judge=ScriptedLLM(verdicts),
    )


def test_judged_run_reports_answer_accuracy(conversation):
    result = run(conversation, ["Rex"], ["CORRECT"])

    assert result.mean_factual_correctness() == 1.0
    # Semantic similarity of identical strings is 1.0, so the composite is 1.0.
    assert result.mean_answer_accuracy() == pytest.approx(1.0)
    assert result.judge_tokens > 0


def test_incorrect_verdict_lowers_accuracy(conversation):
    result = run(conversation, ["a cat"], ["INCORRECT"])

    assert result.mean_factual_correctness() == 0.0
    # Only the semantic term survives, weighted 0.25.
    assert result.mean_answer_accuracy() < 0.25 + 1e-9


def test_unparseable_verdicts_are_excluded_not_counted_wrong(conversation):
    result = run(conversation, ["Rex"], ["no idea really"])

    assert result.judge_unparseable == 1
    assert result.per_qa[0].factual_correctness is None
    # Nothing was gradeable, so there is no Answer Accuracy to report.
    assert result.mean_answer_accuracy() is None


def test_refusing_an_answerable_question_scores_zero_without_consulting_the_judge(conversation):
    judge = ScriptedLLM(["CORRECT"])  # judges rate refusals CORRECT; never ask one
    result = evaluate(
        "L0", lambda: FlatRecencyMemory(window=4), [conversation],
        llm=ScriptedLLM([REFUSAL_SENTINEL]), embedder=StableHashEmbedder(), judge=judge,
    )

    assert result.per_qa[0].refused is True
    assert result.per_qa[0].factual_correctness == 0.0
    assert judge.prompts == []
    assert result.judge_tokens == 0


def test_answer_accuracy_is_none_without_a_judge(conversation):
    result = evaluate(
        "L0", lambda: FlatRecencyMemory(window=4), [conversation],
        llm=ScriptedLLM(["Rex"]), embedder=StableHashEmbedder(),
    )

    # The formula is 75% judged correctness; the semantic term alone is not it.
    assert result.mean_answer_accuracy() is None


def test_unanswerable_questions_are_not_sent_to_the_judge(conversation):
    conversation.qa_items = [
        QAItem(question="trick question", answer="", category=5, evidence=["D1:1"])
    ]
    judge = ScriptedLLM(["CORRECT"])
    evaluate(
        "L0", lambda: FlatRecencyMemory(window=4), [conversation],
        llm=ScriptedLLM([REFUSAL_SENTINEL]), judge=judge,
    )

    # Correctness for these is the refusal metric's job — there is no reference
    # answer to grade against.
    assert judge.prompts == []
