# long-dialog-augment

A Python pipeline that extends Korean multi-session dialogue data (AI Hub) into 60+ turn conversations by repeatedly calling a LLaMA-3.3-70B model served via vLLM.

---

## How it works

1. **Parse** — `dialog_parser.py` flattens the nested `sessionInfo[].dialog` structure into a flat list of `{speaker, utterance}` dicts.
2. **Build prompt** — `llm_generator.py` fills `templates/prompt_template.txt` with the topic, speaker profiles/personas, and recent dialogue history.
3. **Generate** — the filled template (already in LLaMA-3 chat format) is sent to a vLLM `/v1/completions` endpoint.
4. **Loop** — `main.py` repeats steps 2–3 until the dialogue reaches `TARGET_TURNS` (default 60), then saves a JSON turn list to the output path.

---

## Project structure

```
long-dialog-augment/
├── config.py             # 5 params: TARGET_TURNS, MAX_NEW_TOKENS, TEMPERATURE, WINDOW_SIZE, LLAMA_SERVER_URL
├── dialog_parser.py      # flatten_sessions(): sessionInfo[] → flat turn list
├── llm_generator.py      # build_prompt() + generate_next_turn()
├── main.py               # CLI entry point
├── utils.py              # load_json / save_json / count_turns
├── templates/
│   └── prompt_template.txt   # LLaMA-3 chat format template
└── requirements.txt
```

Exploratory notebooks (`test.ipynb`, `test2.ipynb`) are development scratch files; the canonical pipeline is `main.py`.

---

## Setup

**1. Start a vLLM server**

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
    --port 8001
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set the server URL** (defaults to `localhost:8001` if unset)

```bash
export LLAMA_SERVER_URL=http://<your-server>:8001/v1/completions
```

---

## Usage

```bash
python main.py \
  --input  data/your_dialog.json \
  --output outputs/augmented/your_dialog-augmented.json \
  --topic  "호캉스"
```

Optional flag:

| Flag | Default | Description |
|------|---------|-------------|
| `--template` | `templates/prompt_template.txt` | Path to the prompt template |

The output is a JSON array of `{speaker, utterance}` objects (same schema as the flattened input).

---

## Input data format

Expected: AI Hub Korean Multi-Session Dialogue JSON.  
Key paths used by the pipeline:

| Field | Path |
|-------|------|
| Speaker profiles | `participantsInfo.speaker1 / speaker2` |
| Persona features | `personaInfo.clInfo.personaFeatures` / `cpInfo` |
| Dialogue turns | `sessionInfo[].dialog[].{speaker, utterance}` |
| Per-session persona summary | `sessionInfo[].sessionPersonaSummary` |

**Note:** The raw data files are not committed to this repo due to redistribution restrictions. Download from [AI Hub Multi-Session Dialogue (71630)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71630) and place files under `data/`.

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_TURNS` | `60` | Target turn count (1 turn = 2 utterances) |
| `MAX_NEW_TOKENS` | `256` | Max tokens per generation call |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `WINDOW_SIZE` | `20` | Recent utterances sent per request (sliding window) |
| `LLAMA_SERVER_URL` | `$LLAMA_SERVER_URL` env var, fallback `http://localhost:8001/v1/completions` | vLLM endpoint |

---

## Data sources

- [AI Hub Korean Multi-Session Conversations (71630)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71630)
- [AI Hub Topic-Based Text Dialogue (543)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=543)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact

Sewon Kim — Computer Science & Engineering, Jeonbuk National University  
[andimsewon.github.io](https://andimsewon.github.io) · [github.com/andimsewon](https://github.com/andimsewon)
