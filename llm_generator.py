import logging
import time
from typing import List, Dict

import requests

from config import (
    LLAMA_SERVER_URL, TEMPERATURE, MAX_NEW_TOKENS,
    WINDOW_SIZE, MODEL_ID, MAX_RETRIES, REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


def load_template(template_path: str) -> str:
    """Read the prompt template from disk once; callers cache the result."""
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(
    dialog_history: List[Dict[str, str]],
    topic: str,
    persona1: str,
    persona2: str,
    profile1: str,
    profile2: str,
    template: str,          # template content string, not a file path
    persona_summary: str = "",
) -> str:
    # Send only the most recent WINDOW_SIZE utterances to keep token count bounded.
    window = dialog_history[-WINDOW_SIZE:]
    history_text = "\n".join(
        f"{d['speaker']}: {d['utterance']}" for d in window
    )
    return (
        template
        .replace("{{dialog_history}}", history_text)
        .replace("{{persona_summary}}", persona_summary)
        .replace("{{topic}}", topic)
        .replace("{{persona1}}", persona1)
        .replace("{{persona2}}", persona2)
        .replace("{{profile1}}", profile1)
        .replace("{{profile2}}", profile2)
    )


def generate_next_turn(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}

    # prompt already contains the full LLaMA-3 chat format produced by build_prompt()
    # (the template includes <|begin_of_text|>...<|start_header_id|>assistant<|end_header_id|>)
    # — do NOT wrap it again or the model receives doubled special tokens.
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stop": ["<|eot_id|>"],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("LLM request attempt %d/%d", attempt, MAX_RETRIES)
            response = requests.post(
                LLAMA_SERVER_URL,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            # vLLM /v1/completions returns {"choices": [{"text": "..."}]}
            choices = response.json().get("choices", [])
            return choices[0]["text"].strip() if choices else ""
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning("Connection error on attempt %d/%d: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
            else:
                logger.error("Max retries exceeded; returning empty response.")
                return ""
        except Exception as e:
            logger.error("Unexpected error from LLM server: %s", e)
            return ""
