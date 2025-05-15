#dialogue_parser.py
def flatten_sessions(json_data):
    result = []
    for session in json_data["sessionInfo"]:
        for utter in session["dialog"]:
            result.append({
                "speaker": utter["speaker"],
                "utterance": utter["utterance"]
            })
    return result
