"""The timestamp path, end to end.

LoCoMo's temporal answers are the session date plus a relative phrase in the
turn ("yesterday" in a session dated 8 May 2023 -> "7 May 2023"), so a turn that
reaches the generation LLM without its timestamp cannot be answered, only
guessed.
"""

import json

from fakes import ScriptedLLM, StableHashEmbedder

from gms.architectures.base import format_turn, turn_content
from gms.architectures.flat import FlatRecencyMemory
from gms.architectures.kv_entity import KVEntityMemory
from gms.architectures.vector import VectorMemory
from gms.data.locomo import load_locomo
from gms.data.schema import Turn

DATED = Turn(
    dia_id="D1:3",
    speaker="Caroline",
    text="I went to a support group yesterday",
    session_date="1:56 pm on 8 May, 2023",
)
UNDATED = Turn(dia_id="D1:1", speaker="Alice", text="hello")


# --- rendering ---------------------------------------------------------------


def test_context_rendering_carries_the_timestamp():
    assert format_turn(DATED) == "[1:56 pm on 8 May, 2023] Caroline: I went to a support group yesterday"


def test_turns_without_a_timestamp_render_unchanged():
    # Synthetic generators and other datasets need not supply one.
    assert format_turn(UNDATED) == "Alice: hello"


def test_matching_text_excludes_the_timestamp():
    # Every turn in a session shares a date; embedding it dilutes the match
    # without helping, since the questions contain no date to match against.
    assert turn_content(DATED) == "Caroline: I went to a support group yesterday"


# --- the loader populates it -------------------------------------------------


def test_loader_denormalises_session_date_onto_turns(tmp_path):
    raw = [
        {
            "sample_id": "c1",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "hi"}],
                "session_1_date_time": "1:56 pm on 8 May, 2023",
            },
        }
    ]
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(raw))

    turn = load_locomo(path)[0].sessions[0].turns[0]

    assert turn.session_date == "1:56 pm on 8 May, 2023"


# --- every architecture surfaces it ------------------------------------------


def test_flat_memory_surfaces_the_timestamp():
    mem = FlatRecencyMemory(window=5)
    mem.add_turn(DATED)

    assert "8 May, 2023" in mem.retrieve("when?").context


def test_vector_memory_surfaces_the_timestamp_but_does_not_embed_it():
    embedder = StableHashEmbedder()
    mem = VectorMemory(embedder=embedder, top_k=1)
    mem.add_turn(DATED)

    assert "8 May, 2023" in mem.retrieve("support group").context
    # The stored vector is the content's, not the dated line's.
    expected = embedder.embed([turn_content(DATED)])[0]
    assert (mem._embeddings[0] == expected).all()


def test_kv_entity_attributes_stay_temporally_grounded():
    extraction = json.dumps(
        {"entities": [{"name": "Caroline", "attributes": ["went to a support group yesterday"]}]}
    )
    mem = KVEntityMemory(
        llm=ScriptedLLM([extraction]), embedder=StableHashEmbedder(), top_k=1
    )
    mem.add_turn(DATED)

    context = mem.retrieve("Caroline").context
    assert context == "[1:56 pm on 8 May, 2023] Caroline: went to a support group yesterday"


def test_extraction_prompt_sees_content_without_the_timestamp():
    llm = ScriptedLLM(['{"entities": []}'])
    KVEntityMemory(llm=llm, embedder=StableHashEmbedder()).add_turn(DATED)

    assert "8 May" not in llm.prompts[0]
