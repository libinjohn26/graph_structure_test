import json

import pytest

from gms.data.locomo import load_locomo

RAW_SAMPLE = [
    {
        "sample_id": "conv-01",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [
                {"dia_id": "d1", "speaker": "Alice", "text": "Hi Bob, I got a new dog."},
                {"dia_id": "d2", "speaker": "Bob", "text": "Nice, what breed?"},
            ],
            "session_1_date_time": "1:00 pm on 1 January, 2024",
            "session_2": [
                {"dia_id": "d3", "speaker": "Alice", "text": "The dog's name is Rex."},
            ],
            "session_2_date_time": "3:00 pm on 3 January, 2024",
        },
        "qa": [
            {
                "question": "What is the name of Alice's dog?",
                "answer": "Rex",
                "category": 1,
                "evidence": ["d3"],
            }
        ],
    }
]


@pytest.fixture
def locomo_file(tmp_path):
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(RAW_SAMPLE))
    return path


def test_missing_file_raises_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="download_locomo.sh"):
        load_locomo(missing)


def test_loads_conversations(locomo_file):
    conversations = load_locomo(locomo_file)
    assert len(conversations) == 1

    conv = conversations[0]
    assert conv.sample_id == "conv-01"
    assert conv.speaker_a == "Alice"
    assert conv.speaker_b == "Bob"


def test_sessions_ordered_and_parsed_correctly(locomo_file):
    conv = load_locomo(locomo_file)[0]
    assert conv.num_sessions() == 2
    assert [s.session_id for s in conv.sessions] == ["session_1", "session_2"]
    assert conv.sessions[0].date_time == "1:00 pm on 1 January, 2024"
    assert conv.num_turns() == 3


def test_turn_fields_preserved(locomo_file):
    conv = load_locomo(locomo_file)[0]
    first_turn = conv.sessions[0].turns[0]
    assert first_turn.dia_id == "d1"
    assert first_turn.speaker == "Alice"
    assert first_turn.text == "Hi Bob, I got a new dog."


def test_qa_items_parsed(locomo_file):
    conv = load_locomo(locomo_file)[0]
    assert len(conv.qa_items) == 1
    qa = conv.qa_items[0]
    assert qa.question == "What is the name of Alice's dog?"
    assert qa.answer == "Rex"
    assert qa.category == 1
    assert qa.evidence == ["d3"]


def test_missing_qa_and_evidence_default_to_empty(tmp_path):
    minimal = [
        {
            "sample_id": "conv-02",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"dia_id": "d1", "speaker": "A", "text": "hello"}],
                "session_1_date_time": "noon",
            },
            # no "qa" key at all
        }
    ]
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(minimal))
    conv = load_locomo(path)[0]
    assert conv.qa_items == []
