#main.py
import os
from dialog_parser import flatten_sessions
from utils import load_json, save_json, count_turns
from llm_generator import build_prompt, generate_next_turn
from config import TARGET_TURNS

def parse_generated_text(text):
    turns = []
    for line in text.strip().split("\n"):
        if ":" in line:
            speaker, utterance = line.split(":", 1)
            turns.append({"speaker": speaker.strip(), "utterance": utterance.strip()})
    return turns

def main():
    # 파일 경로
    input_path = "data/02.라벨링데이터/session4/K4-00001-CL91073-CP94160-12-10-S4.json"
    persona_path = "data/persona_info.json"
    prompt_template = "templates/prompt_template.txt"
    output_path = "outputs/augmented/K4-00001-augmented.json"

    raw_data = load_json(input_path)
    persona = load_json(persona_path)
    dialog = flatten_sessions(raw_data)

    while count_turns(dialog) < TARGET_TURNS:
        prompt = build_prompt(dialog, topic="호캉스",
                              persona1=persona["speaker1"],
                              persona2=persona["speaker2"],
                              template_path=prompt_template)
        generated = generate_next_turn(prompt)
        new_turns = parse_generated_text(generated)
        dialog.extend(new_turns)

    save_json(dialog, output_path)
    print(f"✅ 대화 증강 완료: {count_turns(dialog)} 턴 → {output_path}")

if __name__ == "__main__":
    main()
