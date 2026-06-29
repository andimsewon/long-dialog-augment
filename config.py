# config.py
import os

TARGET_TURNS    = 60
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.7
WINDOW_SIZE     = 20   # utterances sent per request (sliding window)
MAX_RETRIES     = 3
REQUEST_TIMEOUT = 60   # seconds

MODEL_ID = os.getenv(
    "LLAMA_MODEL_ID",
    "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
)
LLAMA_SERVER_URL = os.getenv(
    "LLAMA_SERVER_URL",
    "http://localhost:8001/v1/completions",
)
