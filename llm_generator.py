import requests
from config import LLAMA_SERVER_URL, TEMPERATURE, MAX_NEW_TOKENS

def build_prompt(dialog_history, topic, persona1, persona2, profile1, profile2, template_path):
    with open(template_path, "r", encoding='utf-8') as f:
        template = f.read()

    history_text = "\n".join([f"{d['speaker']}: {d['utterance']}" for d in dialog_history])

    prompt = (
        template.replace("{{dialog_history}}", history_text)
                .replace("{{topic}}", topic)
                .replace("{{persona1}}", persona1)
                .replace("{{persona2}}", persona2)
                .replace("{{profile1}}", profile1)
                .replace("{{profile2}}", profile2)
    )
    return prompt

def generate_next_turn(prompt):
    try:
        headers = {"Content-Type": "application/json"}

        system_prompt = (
            "You are a Korean dialogue generation model. "
            "Continue a long-form conversation between two speakers using the given profile, persona, and prior dialogue. "
            "The output must be in natural Korean and reflect each speaker’s tone, personality, and background. "
            "Avoid restarting or summarizing. Continue naturally."
        )

        # LLaMA 3 OpenChat format
        final_input = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>"
        )

        payload = {
            "model": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
            "prompt": final_input,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "stop": ["<|eot_id|>"]  # Optional: ensures generation halts properly
        }

        response = requests.post(LLAMA_SERVER_URL, json=payload, headers=headers)
        response.raise_for_status()

        output_text = response.json().get("text", "").strip()
        return output_text

    except Exception as e:
        print("🔥 LLaMA 서버 오류:", e)
        return ""
