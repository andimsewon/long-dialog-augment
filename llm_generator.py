#llm_generator.py
import requests
from config import LLAMA_SERVER_URL, TEMPERATURE, MAX_NEW_TOKENS

def build_prompt(dialog_history, topic, persona1, persona2, template_path):
    with open(template_path, "r", encoding='utf-8') as f:
        template = f.read()
    history_text = "\n".join([f"{d['speaker']}: {d['utterance']}" for d in dialog_history])
    prompt = template.replace("{{dialog_history}}", history_text)
    prompt = prompt.replace("{{topic}}", topic)
    prompt = prompt.replace("{{persona1}}", persona1)
    prompt = prompt.replace("{{persona2}}", persona2)
    return prompt

def generate_next_turn(prompt):
    try:
        headers = {"Content-Type": "application/json"}
        final_input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>당신은 두 사람의 대화를 이어서 자연스럽게 작성합니다.<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        payload = {
            "model": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
            "prompt": final_input,
            "max_tokens": MAX_NEW_TOKENS
        }

        response = requests.post(LLAMA_SERVER_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("text", "")
    except Exception as e:
        print("🔥 LLaMA 서버 오류:", e)
        return ""
