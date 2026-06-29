import pytest
from main import parse_generated_text
from dialog_parser import flatten_sessions


# ── parse_generated_text ──────────────────────────────────────────────────────

def test_parse_basic_two_turns():
    text = "speaker1: 안녕하세요\nspeaker2: 반갑습니다"
    turns = parse_generated_text(text)
    assert turns == [
        {"speaker": "speaker1", "utterance": "안녕하세요"},
        {"speaker": "speaker2", "utterance": "반갑습니다"},
    ]


def test_parse_colons_in_utterance():
    """Colons in the utterance body must be preserved (split on first colon only)."""
    text = "speaker1: 주소: 서울 강남구"
    turns = parse_generated_text(text)
    assert turns == [{"speaker": "speaker1", "utterance": "주소: 서울 강남구"}]


def test_parse_ignores_non_speaker_lines():
    """Header lines, blank lines, and prose with ':' must not become turns."""
    text = (
        "## Rules:\n"
        "speaker1: 안녕\n"
        "This is prose: not a turn\n"
        "\n"
        "speaker2: 네"
    )
    turns = parse_generated_text(text)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "speaker1"
    assert turns[1]["speaker"] == "speaker2"


def test_parse_case_insensitive_normalises_to_lower():
    """Speaker label matching is case-insensitive; stored label is lowercased."""
    text = "Speaker1: 안녕\nSPEAKER2: 네"
    turns = parse_generated_text(text)
    assert turns[0]["speaker"] == "speaker1"
    assert turns[1]["speaker"] == "speaker2"


def test_parse_speaker_filter_drops_unknown():
    text = "speaker1: 안녕\nspeaker3: 침입자\nspeaker2: 네"
    turns = parse_generated_text(text, known_speakers={"speaker1", "speaker2"})
    assert len(turns) == 2
    assert all(t["speaker"] in {"speaker1", "speaker2"} for t in turns)


def test_parse_empty_text():
    assert parse_generated_text("") == []


def test_parse_whitespace_only():
    assert parse_generated_text("   \n  \n  ") == []


# ── flatten_sessions ──────────────────────────────────────────────────────────

def _make_data(*sessions):
    return {"sessionInfo": [{"dialog": list(s)} for s in sessions]}


def test_flatten_basic():
    data = _make_data(
        [{"speaker": "speaker1", "utterance": "hi"},
         {"speaker": "speaker2", "utterance": "hello"}],
        [{"speaker": "speaker1", "utterance": "bye"}],
    )
    result = flatten_sessions(data)
    assert len(result) == 3
    assert result[0] == {"speaker": "speaker1", "utterance": "hi"}
    assert result[2] == {"speaker": "speaker1", "utterance": "bye"}


def test_flatten_empty_sessions():
    assert flatten_sessions({"sessionInfo": []}) == []


def test_flatten_missing_session_info_key():
    with pytest.raises(KeyError, match="sessionInfo"):
        flatten_sessions({"other": []})


def test_flatten_session_with_empty_dialog():
    data = {"sessionInfo": [{"dialog": []}, {"dialog": [{"speaker": "speaker1", "utterance": "x"}]}]}
    result = flatten_sessions(data)
    assert result == [{"speaker": "speaker1", "utterance": "x"}]
