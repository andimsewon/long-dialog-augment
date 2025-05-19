import argparse
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

def extract_persona_and_profile(json_data):
    cl = json_data["participantsInfo"]["speaker1"]
    cp = json_data["participantsInfo"]["speaker2"]
    cl_info = json_data["personaInfo"]["clInfo"]["personaFeatures"]
    cp_info = json_data["personaInfo"]["cpInfo"]["personaFeatures"]

    persona1 = " ".join(cl_info)
    persona2 = " ".join(cp_info)

    profile1 = f"{cl['gender']} / {cl['age']} / {cl['occupation']} / {cl['educationLevel']} / 출신: {cl['bPlace']}"
    profile2 = f"{cp['gender']} / {cp['age']} / {cp['occupation']} / {cp['educationLevel']} / 출신: {cp['bPlace']}"

    return persona1, persona2, profile1, profile2

def main():
    parser = argparse.ArgumentParser(description="Long dialog augmentation with persona & LLaMA")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--topic", required=True, help="Topic string (ex. '호캉스')")
    parser.add_argument("--template", default="templates/prompt_template.txt", help="Prompt template path")
    args = parser.parse_args()

    raw_data = load_json(args.input)
    persona1, persona2, profile1, profile2 = extract_persona_and_profile(raw_data)
    dialog = flatten_sessions(raw_data)

    while count_turns(dialog) < TARGET_TURNS:
        prompt = build_prompt(
            dialog, topic=args.topic,
            persona1=persona1, persona2=persona2,
            profile1=profile1, profile2=profile2,
            template_path=args.template
        )
        generated = generate_next_turn(prompt)
        new_turns = parse_generated_text(generated)
        dialog.extend(new_turns)

    save_json(dialog, args.output)
    print(f"✅ 대화 증강 완료: {count_turns(dialog)} 턴 → {args.output}")

if __name__ == "__main__":
    main()
