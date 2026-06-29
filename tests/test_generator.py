import unittest.mock as mock

import pytest

from llm_generator import build_prompt, generate_next_turn, load_template


# ── load_template ─────────────────────────────────────────────────────────────

def test_load_template_reads_content(tmp_path):
    tpl = tmp_path / "tpl.txt"
    tpl.write_text("Hello {{topic}}", encoding="utf-8")
    assert load_template(str(tpl)) == "Hello {{topic}}"


def test_load_template_missing_file():
    with pytest.raises(FileNotFoundError):
        load_template("/nonexistent/path/tpl.txt")


# ── build_prompt ──────────────────────────────────────────────────────────────

def _make_template(tmp_path, content="{{persona_summary}}\n{{topic}}\n{{dialog_history}}"):
    tpl = tmp_path / "tpl.txt"
    tpl.write_text(content, encoding="utf-8")
    return load_template(str(tpl))


def test_build_prompt_substitutes_all_placeholders(tmp_path):
    content = "{{topic}} {{persona1}} {{persona2}} {{profile1}} {{profile2}} {{persona_summary}} {{dialog_history}}"
    template = _make_template(tmp_path, content)
    dialog = [{"speaker": "speaker1", "utterance": "hi"}]
    prompt = build_prompt(dialog, "주제", "P1", "P2", "PR1", "PR2", template, "summary")
    assert "주제" in prompt
    assert "P1" in prompt
    assert "P2" in prompt
    assert "PR1" in prompt
    assert "PR2" in prompt
    assert "summary" in prompt
    assert "speaker1: hi" in prompt


def test_build_prompt_no_double_wrap(tmp_path):
    content = "<|begin_of_text|>\n{{topic}}\n{{persona_summary}}\n{{dialog_history}}\n{{persona1}}{{persona2}}{{profile1}}{{profile2}}"
    template = _make_template(tmp_path, content)
    dialog = [{"speaker": "speaker1", "utterance": "hi"}]
    prompt = build_prompt(dialog, "t", "p1", "p2", "pr1", "pr2", template)
    assert prompt.count("<|begin_of_text|>") == 1


def test_build_prompt_applies_sliding_window(tmp_path):
    """Only the last WINDOW_SIZE utterances should appear in the prompt."""
    from config import WINDOW_SIZE

    template = _make_template(tmp_path, "{{dialog_history}}{{topic}}{{persona_summary}}{{persona1}}{{persona2}}{{profile1}}{{profile2}}")
    # Build a dialog longer than WINDOW_SIZE
    dialog = [{"speaker": "speaker1", "utterance": f"msg{i}"} for i in range(WINDOW_SIZE + 5)]
    prompt = build_prompt(dialog, "t", "p1", "p2", "pr1", "pr2", template)
    assert "msg0" not in prompt              # oldest utterance excluded
    assert f"msg{WINDOW_SIZE + 4}" in prompt  # most recent utterance present


def test_build_prompt_empty_persona_summary(tmp_path):
    template = _make_template(tmp_path, "{{persona_summary}}|{{topic}}|{{dialog_history}}|{{persona1}}|{{persona2}}|{{profile1}}|{{profile2}}")
    dialog = [{"speaker": "speaker1", "utterance": "hi"}]
    prompt = build_prompt(dialog, "t", "p1", "p2", "pr1", "pr2", template)
    assert prompt.startswith("|")   # empty summary replaced with ""


# ── generate_next_turn ────────────────────────────────────────────────────────

def _mock_response(text):
    resp = mock.MagicMock()
    resp.json.return_value = {"choices": [{"text": text}]}
    resp.raise_for_status = mock.MagicMock()
    return resp


def test_generate_returns_text():
    with mock.patch("llm_generator.requests.post", return_value=_mock_response("speaker1: 안녕")):
        result = generate_next_turn("dummy")
    assert result == "speaker1: 안녕"


def test_generate_empty_choices():
    resp = mock.MagicMock()
    resp.json.return_value = {"choices": []}
    resp.raise_for_status = mock.MagicMock()
    with mock.patch("llm_generator.requests.post", return_value=resp):
        result = generate_next_turn("dummy")
    assert result == ""


def test_generate_strips_whitespace():
    with mock.patch("llm_generator.requests.post", return_value=_mock_response("  speaker1: 안녕  \n")):
        result = generate_next_turn("dummy")
    assert result == "speaker1: 안녕"


def test_generate_retries_on_connection_error():
    import requests as req_lib
    side_effects = [req_lib.exceptions.ConnectionError("down")] * 2 + [_mock_response("speaker1: 안녕")]
    with mock.patch("llm_generator.requests.post", side_effect=side_effects):
        with mock.patch("llm_generator.time.sleep"):   # skip real sleep
            result = generate_next_turn("dummy")
    assert result == "speaker1: 안녕"


def test_generate_returns_empty_after_max_retries():
    import requests as req_lib
    from config import MAX_RETRIES
    with mock.patch(
        "llm_generator.requests.post",
        side_effect=req_lib.exceptions.ConnectionError("down"),
    ):
        with mock.patch("llm_generator.time.sleep"):
            result = generate_next_turn("dummy")
    assert result == ""
