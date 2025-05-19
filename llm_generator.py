import requests
import time
import json
import os
from typing import List, Dict
import argparse
from dialog_parser import flatten_sessions
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

def generate_next_turn(prompt, max_retries=3, timeout=60):
    headers = {"Content-Type": "application/json"}

    system_prompt = (
        "You are a Korean dialogue generation model. "
        "Continue a long-form conversation between two speakers using the given profile, persona, and prior dialogue. "
        "The output must be in natural Korean and reflect each speaker’s tone, personality, and background. "
        "Avoid restarting or summarizing. Continue naturally."
    )

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
        "stop": ["<|eot_id|>"]
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 LLaMA 요청 시도 {attempt} / {max_retries}")
            response = requests.post(LLAMA_SERVER_URL, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json().get("text", "").strip()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"⏱️ 서버 응답 지연 또는 연결 오류 (시도 {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)  # 점진적 대기
            else:
                print("❌ 최대 재시도 횟수 초과. 빈 응답 반환.")
                return ""
        except Exception as e:
            print("🔥 LLaMA 서버 기타 오류:", e)
            return ""