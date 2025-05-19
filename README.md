# Long Dialog Augment 🧠💬

멀티세션 대화를 최대 60턴 이상으로 자연스럽게 이어붙여 **긴 대화 데이터셋**을 생성하는 Python 기반 시스템입니다.
본 저장소는 전북대학교 인공지능 대화요약 연구 과제의 일부로 개발되었습니다.
LLM은 Meta LLaMA-3 기반 FastAPI 서버와 연동되어 동작합니다.

👉 [프로젝트 소개 발표 자료 보기](./과제제안발표자료_세원님.pptx)

---
# 한국어 멀티세션 대화 https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71630


# 주제별 텍스트 일상 대화 데이터 https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=543

## 📁 프로젝트 구조

```
├── config.py                # 설정 파일 (LLM 서버 주소, 생성 파라미터 등)
├── dialogue_parser.py       # 멀티세션 JSON 대화 파싱 → flat list 변환
├── llm_generator.py         # 프롬프트 생성 및 LLaMA 서버 호출 함수 정의
├── main.py                  # 전체 증강 파이프라인 실행 메인 스크립트
├── utils.py                 # JSON 입출력 및 턴 수 계산 유틸 함수
├── templates/
│   └── prompt_template.txt  # 프롬프트 포맷 정의 텍스트 템플릿
├── data/                    # 입력 라벨링 데이터 경로
├── outputs/                 # 증강된 결과 JSON 저장 경로
├── requirements.txt         # 의존성 목록
└── README.md
```

---

## 🚀 실행 방법

1. `data/` 폴더에 증강할 라벨링 JSON을 넣습니다.
2. `templates/prompt_template.txt`에서 프롬프트 포맷을 설정합니다.
3. `config.py`에서 서버 URL 및 최대 턴 수 등 세부 설정을 확인합니다.
4. `main.py`를 실행하면 자동으로 대화가 생성됩니다.

```bash
python main.py
```

생성된 대화는 `outputs/` 폴더에 JSON 형태로 저장됩니다.

---

## ⚙️ 설정 예시 (`config.py`)

```python
TARGET_TURNS = 60              # 생성 목표 턴 수 (1턴 = 왕복 2화자 발화)
MAX_NEW_TOKENS = 256           # LLM 응답 최대 토큰 길이
TEMPERATURE = 0.7              # 생성 다양성 조절

LLAMA_SERVER_URL = "http://localhost:8001/v1/completions"  # LLaMA 서버 주소
```

---

## 📄 프롬프트 템플릿 예시 (`templates/prompt_template.txt`)

```
당신은 두 사람의 대화를 이어서 자연스럽게 작성합니다.
주제는 "{{topic}}"이며,

- speaker1 페르소나: {{persona1}}
- speaker2 페르소나: {{persona2}}

지금까지의 대화:
{{dialog_history}}

다음 발화를 이어서 1~2턴 생성해주세요 (형식: 화자명: 발화).
```

---

## 💡 주요 특징

* ✅ 멀티세션 대화 자동 합성
* ✅ 페르소나 기반 컨텍스트 유지
* ✅ 대화형 프롬프트 기반 생성
* ✅ LLaMA 서버 연동 구성

---

## 📦 requirements.txt

```
requests>=2.28.0
```

설치:

```bash
pip install -r requirements.txt
```

---

## 🔌 LLaMA API 연동 예시

```python
import requests

headers = {"Content-Type": "application/json"}
payload = {
  "model": "llama3",
  "prompt": "<|begin_of_text|>...",
  "max_tokens": 256
}

requests.post("http://localhost:8001/v1/completions", json=payload, headers=headers)
```

---

## 📬 문의

* 작성자: \[세원님 프로젝트 기반 - 2025 R\&D 긴 대화 이해 및 요약 과제]
* GitHub: [andimsewon/long-dialog-augment](https://github.com/andimsewon/long-dialog-augment)
* 이메일: [carrotsw@naver.com](mailto:your-email@example.com)
