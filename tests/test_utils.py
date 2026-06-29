import json
import os

import pytest

from utils import count_turns, ensure_dir, load_json, save_json


def test_count_turns_basic():
    dialog = [
        {"speaker": "speaker1", "utterance": "a"},
        {"speaker": "speaker2", "utterance": "b"},
    ]
    assert count_turns(dialog) == 1


def test_count_turns_empty():
    assert count_turns([]) == 0


def test_count_turns_odd_utterances():
    # 3 utterances → 1 complete turn; the odd one doesn't count
    dialog = [{"speaker": "s", "utterance": str(i)} for i in range(3)]
    assert count_turns(dialog) == 1


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "test.json")
    obj = [{"speaker": "speaker1", "utterance": "안녕"}]
    save_json(obj, path)
    assert load_json(path) == obj


def test_save_json_korean_not_escaped(tmp_path):
    path = str(tmp_path / "kr.json")
    save_json({"text": "안녕하세요"}, path)
    raw = path and open(path, encoding="utf-8").read()
    assert "안녕하세요" in raw   # ensure_ascii=False is working


def test_ensure_dir_creates_nested(tmp_path):
    new_dir = str(tmp_path / "a" / "b" / "c")
    ensure_dir(new_dir)
    assert os.path.isdir(new_dir)


def test_ensure_dir_idempotent(tmp_path):
    d = str(tmp_path / "x")
    ensure_dir(d)
    ensure_dir(d)   # should not raise


def test_ensure_dir_empty_string():
    ensure_dir("")   # should be a no-op, not crash
