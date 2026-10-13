import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from gms.data.hotpotqa import load_hotpotqa  # noqa: E402


def write_parquet(tmp_path, rows):
    table = pyarrow.Table.from_pylist(rows)
    path = tmp_path / "validation_distractor.parquet"
    pq.write_table(table, path)
    return path


def row(qid, question, answer, qtype, titles, sentences, facts_title, facts_sent):
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": qtype,
        "level": "hard",
        "supporting_facts": {"title": facts_title, "sent_id": facts_sent},
        "context": {"title": titles, "sentences": sentences},
    }


@pytest.fixture
def sample(tmp_path):
    return write_parquet(
        tmp_path,
        [
            row(
                "q1", "Who directed it?", "Tim Burton", "bridge",
                ["Ed Wood (film)", "Tim Burton"],
                [["Ed Wood is a 1994 film.", "It was directed by Tim Burton."],
                 ["Tim Burton is an American director."]],
                ["Ed Wood (film)", "Tim Burton"], [1, 0],
            ),
            row(
                "q2", "Same nationality?", "yes", "comparison",
                ["Scott Derrickson", "Ed Wood (film)"],
                [["Scott Derrickson is American."], ["Ed Wood is a 1994 film.", "It was directed by Tim Burton."]],
                ["Scott Derrickson"], [0],
            ),
        ],
    )


def test_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_hotpotqa.sh"):
        load_hotpotqa(tmp_path / "nope.parquet")


def test_sentences_become_turns_with_matching_ids(sample):
    conv = load_hotpotqa(sample, questions_per_pool=20)[0]
    turns = {t.dia_id: t for s in conv.sessions for t in s.turns}

    assert "Ed Wood (film)::1" in turns
    assert turns["Ed Wood (film)::1"].text == "It was directed by Tim Burton."
    # The article title stands in for the speaker — the entity the text is about.
    assert turns["Ed Wood (film)::1"].speaker == "Ed Wood (film)"


def test_evidence_ids_resolve_to_real_turns(sample):
    conv = load_hotpotqa(sample, questions_per_pool=20)[0]
    ids = {t.dia_id for s in conv.sessions for t in s.turns}

    for qa in conv.qa_items:
        assert qa.evidence
        for ref in qa.evidence:
            assert ref in ids, f"{ref} does not name a turn"


def test_repeated_articles_are_stored_once(sample):
    # "Ed Wood (film)" is a distractor for q2 and gold for q1.
    conv = load_hotpotqa(sample, questions_per_pool=20)[0]
    titles = [s.session_id for s in conv.sessions]

    assert titles.count("Ed Wood (film)") == 1
    assert len(titles) == len(set(titles))


def test_questions_are_pooled_into_one_conversation(sample):
    conversations = load_hotpotqa(sample, questions_per_pool=20)

    assert len(conversations) == 1
    assert len(conversations[0].qa_items) == 2


def test_pool_size_splits_conversations(sample):
    conversations = load_hotpotqa(sample, questions_per_pool=1)

    # A pool per question, so each corpus is only that question's paragraphs.
    assert len(conversations) == 2
    assert all(len(c.qa_items) == 1 for c in conversations)


def test_category_is_the_raw_hotpot_type(sample):
    conv = load_hotpotqa(sample, questions_per_pool=20)[0]

    assert {qa.category for qa in conv.qa_items} == {"bridge", "comparison"}


def test_turns_carry_no_timestamp(sample):
    conv = load_hotpotqa(sample, questions_per_pool=20)[0]
    turn = conv.sessions[0].turns[0]

    # HotpotQA has no times; format_turn must then render without a prefix.
    from gms.architectures.base import format_turn

    assert turn.session_date == ""
    assert not format_turn(turn).startswith("[")


def test_limit_caps_questions(sample):
    conversations = load_hotpotqa(sample, questions_per_pool=20, limit=1)

    assert sum(len(c.qa_items) for c in conversations) == 1
