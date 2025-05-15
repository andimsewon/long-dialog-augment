#utils.py
import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def count_turns(dialog):
    return len(dialog) // 2  # 1턴은 두 화자의 한 왕복
