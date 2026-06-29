from typing import Any, Dict, List


def flatten_sessions(json_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Flatten the nested AI Hub sessionInfo structure into a plain turn list.

    Input schema:  {"sessionInfo": [{"dialog": [{"speaker": ..., "utterance": ...}]}]}
    Output schema: [{"speaker": ..., "utterance": ...}, ...]
    """
    if "sessionInfo" not in json_data:
        raise KeyError(
            "'sessionInfo' key not found. "
            "Make sure the input is an AI Hub multi-session dialogue JSON."
        )

    result: List[Dict[str, str]] = []
    for session in json_data["sessionInfo"]:
        for utter in session.get("dialog", []):
            result.append({
                "speaker": utter["speaker"],
                "utterance": utter["utterance"],
            })
    return result
