import json
import os
from typing import Any, List, Dict


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def count_turns(dialog: List[Dict[str, str]]) -> int:
    """Return number of complete turns (1 turn = 2 utterances, one per speaker)."""
    return len(dialog) // 2


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist. No-op if path is empty."""
    if path:
        os.makedirs(path, exist_ok=True)
