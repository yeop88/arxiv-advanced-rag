# FreeToken 사용 가이드 (이 프로젝트 기준)

- **대상 버전**: FreeToken `0.2.2+g9866a699f` (`ft --version`으로 확인)
- **작성일**: 2026-11-10
- **근거**: 공식 웹 문서를 찾지 못해, 이 PC에 설치된 `ft` CLI 도움말(`ft --help`, `ft serve --help`,
  `ft ctl --help`, `ft daemon --help`)과 실행 중인 서버 조회값(`ft ctl stats`, `ft ctl cache status`)을 근거로 정리했다.
  **버전이 바뀌면 옵션 이름·기본값이 달라질 수 있으니 `ft <명령> --help`로 다시 확인할 것.**

---

## 1. FreeToken은 무엇이고, 이 프로젝트에서 어떻게 쓰나

FreeToken은 로컬 PC에서 LLM을 띄워 **OpenAI 호환 API**로 제공하는 추론 서버다.
이 프로젝트는 `LLM_PROVIDER=local`로 설정하면 이 서버를 생성·검증·질의변환 LLM으로 쓴다.

```
FreeToken 앱(GUI)
  └─ ft daemon  (감독자, 포트 19000)
       └─ ft serve (추론 서버, 포트 1919)  ← 우리 코드가 여기에 HTTP 요청
```

- **우리 코드는 FreeToken을 켜고 끄거나 자원 설정을 바꾸지 않는다.** 앱에서 띄운 서버에 요청만 보낸다.
- 연결은 `rag/llm.py`의 `LLM_PROVIDER=local` 경로(`ChatOpenAI` + `base_url`) 한 곳뿐이다.
- 준비 상태 확인은 인증이 필요 없는 `/health` 경로만 쓴다(`rag/warmup.py`의 `llm_status`).

### 현재 서버 실행 설정 (2026-11-10 프로세스 확인)
```
ft serve --model <...>\.freetoken\models\Qwen3.6-35B-A3B-FP8 --port 1919
         --moe-backend auto --max-running-requests 4 --memory-ratio 0.85 --host 127.0.0.1
```
- 모델: `Qwen3.6-35B-A3B-FP8` (MoE, 활성 파라미터 약 3B, 문맥 262k, 텍스트·이미지 입력)
- `--moe-backend`는 deprecated 별칭이며 `--moe-strategy`와 같다. `auto`는 MoE 모델에서 `offload`로 동작
  (전문가 가중치를 RAM에 두고 필요할 때 GPU로 가져옴).

---

## 2. 프로젝트 연결 설정 (`.env`)

```
LLM_PROVIDER=local
LOCAL_BASE_URL=http://127.0.0.1:1919/v1     # FreeToken 표준 포트 1919, /v1 까지
LOCAL_API_KEY=<FreeToken API 키>             # 서버가 --api-key(또는 FREETOKEN_API_KEY)로 켜졌을 때 필요
LLM_MODEL=Qwen3.6-35B-A3B-FP8                # /v1/models 가 돌려주는 id 그대로(기본값 = --model 폴더 이름)
LLM_MODEL_LIGHT=Qwen3.6-35B-A3B-FP8
LLM_TEMPERATURE=0.6                          # 서버 기본은 1.0. Qwen 권장: thinking 0.6 / non-thinking 0.7
LLM_MAX_TOKENS=8192                          # thinking 은 추론 토큰도 포함되므로 여유 있게
LOCAL_THINKING=on|off                        # 생성·검증 thinking (챗봇 사이드바 토글로도 변경)
LOCAL_THINKING_LIGHT=on|off                  # 질의변환 thinking (보통 off)
```

- **모델을 바꾸면 `LLM_MODEL`도 바꿔야 한다.** 서버의 모델 id는 모델 폴더 이름이라 BF16→FP8 처럼 바꾸면 id도 바뀐다.
- thinking 제어는 요청의 `chat_template_kwargs.enable_thinking`으로 보낸다. **서버 기본값은 thinking on**이므로
  값을 비워 두면 생각 모드로 동작한다(느림).

---

## 3. 주요 명령

### 3.1 상태 확인 — `ft ctl` (기본 주소 `http://127.0.0.1:1919`)
API 키가 걸린 서버는 `FREETOKEN_API_KEY` 환경변수나 `--api-key`로 키를 넘긴다(`health`는 키 불필요).

| 명령 | 내용 |
|---|---|
| `ft ctl health` | `status=ok/loading`, 모델, 가동 시간 |
| `ft ctl --json stats` | GPU, KV/상태 슬롯 사용량, 완료 요청 수, 처리량 |
| `ft ctl --json cache status` | 캐시 예산, MoE 전문가 GPU 슬롯, KV 토큰 수, thinking 기본값 |
| `ft ctl requests` | 진행 중인 요청 |
| `ft ctl cache rebuild --moe N --kv N --mamba N` | **재시작 없이** 캐시 크기 재조정 |

`--json`은 하위 명령 **앞**에 쓴다(`ft ctl --json stats`).

### 3.2 서버 실행 옵션 — `ft serve` (자원·속도 관련만)

| 옵션 | 역할 |
|---|---|
| `--memory-ratio` | **켜질 때 남아 있던 GPU 메모리** 중 엔진이 쓸 비율(가중치 + MoE 캐시 + KV 캐시) |
| `--moe-strategy {auto,fused,offload,cpu,hybrid}` | 전문가 실행 위치. auto = offload |
| `--moe-cache-size / -rate / -mem / -auto` | GPU에 올려 둘 전문가 수(개수·비율·메모리 크기·자동) |
| `--kv-reserve-tokens` | `--moe-cache-auto` 사용 시 KV 캐시 최소 확보량 |
| `--moe-cpu-threads`, `--moe-cpu-layers` | 전문가를 CPU에서 계산할 때의 스레드·대상 층 |
| `--max-running-requests` | 동시 처리 요청 수 |
| `--max-output-tokens` | 요청에 max_tokens 가 없을 때 기본값(32k) |
| `--max-seq-len-override` | 최대 문맥 길이 |
| `--cache-type {naive,radix}` | radix = 요청 간 앞부분(prefix) 재사용 |
| `--speculative-algorithm mtp` | 추측 디코딩(모델의 MTP 층 사용). 기본 꺼짐 |
| `--reasoning-parser auto` | 추론을 `reasoning_content`로 분리(Qwen3 → `<think>`) |
| `--sampling-defaults model` | temperature 등 미지정 시 모델 generation_config 값 사용 |
| `--enable-timings-report` / `--enable-cache-report` | 응답에 요청별 시간 / prefix 캐시 적중 토큰 수 포함 |
| `--api-key` | 인증 키(기본 `$FREETOKEN_API_KEY`, 없으면 인증 없음) |

### 3.3 수명 관리 — `ft daemon`
- 감독자 서버. 클라이언트 명령: `status health metrics stats start stop switch logs`
  (예: `ft daemon start MODEL --port 1919`, `ft daemon logs`)
- `--auto-restart`(비정상 종료 시 재시작, 기본 꺼짐), `--stop-serve-on-exit`(기본은 데몬이 꺼져도 엔진 유지)
- 기본 제어 포트 1900. FreeToken 앱은 19000으로 띄운다.

### 3.4 기타
- `ft bench bw`: CPU와 PCIe 대역폭 비교 — `--moe-strategy cpu/hybrid`가 나은지 판단용(시스템 부하 있음)
- `ft shell`: 터미널에서 대화, `ft checkpoint`: HF safetensors → FTW 변환

---

## 4. 자원 관리 — 알아야 할 것

- **한 번 올리면 계속 상주한다.** 요청마다 다시 로드하지 않으며, 유휴 시 자동으로 내리는 옵션은 도움말에 없다
  (Ollama의 keep_alive 같은 기능 없음). 끄려면 앱이나 `ft daemon stop`.
- **자원 배분은 대부분 시작 시점에 정해진다.** `--memory-ratio`는 "켜질 때의 여유 GPU 메모리" 기준이다.
- **켜는 순서가 중요하다.** 이 프로젝트의 로컬 모델(bge-m3 임베딩 + bge-reranker, 모두 GPU fp16, 약 2.3GB)이
  먼저 GPU를 잡고 있으면 FreeToken의 MoE 캐시가 줄어든다.
  - 실제 사례(2026-11-10): 재시작 전후 캐시 예산 9.66GB → 8.21GB, 전문가 GPU 슬롯 2,506 → 2,219개.
    재시작 시점에 다른 GPU 프로세스가 메모리를 쓰고 있었던 것으로 추정.
  - **권장: FreeToken을 먼저 켜고(로딩 완료 확인) → 챗봇 실행.**
- **현재 배분** (FP8, 2026-11-10 `ft ctl cache status`):

| 항목 | 값 |
|---|---|
| 캐시 예산 | 8.21GB |
| MoE 전문가 GPU 슬롯 | 2,219 / 10,240 (22%) = 6.50GB, 전문가 1개 3.0MB |
| KV 캐시 | 8,303 토큰 (0.17GB) — 조정 가능 최대 약 39만 토큰 |
| GDN 상태 슬롯 | 24개 (1.44GB) |
| GPU 전체 사용 | 약 12.8 / 16GB (RTX 4080 SUPER) |

- **KV 캐시가 작다(약 8.3k 토큰).** 한 번에 한 질문(입력 약 1.5k 토큰)은 충분하지만, thinking on으로 출력이
  수천 토큰이거나 평가·챗봇을 동시에 돌리면 부족할 수 있다. 필요 시 `ft ctl cache rebuild --kv 32k`
  (전문가 약 200개 = 0.6GB가 GPU에서 빠짐).

---

## 5. 측정한 성능 (RTX 4080 SUPER 16GB, i9-14900KF, RAM 128GB)

| 항목 | BF16 | FP8 |
|---|---|---|
| 생성 속도 | 21~28 tok/s | 28~38 tok/s |
| 긴 입력 처리 | 약 190~260 tok/s | 약 180~210 tok/s |
| 짧은 질문 첫 토큰 | 약 5초 | 약 5초 |

- **느린 근본 원인은 모델 구조와 VRAM 크기**: 35B MoE의 전문가 대부분이 RAM에 있고 PCIe로 가져오기 때문에,
  짧은 질문에도 요청당 약 5초가 고정으로 든다. FP8은 전문가 크기를 줄일 뿐 이 구조는 같다.
- 파이프라인(thinking off) 질문 1개 약 1분: 질의변환 약 6초 · 검색 약 3초 · 생성 약 15~25초 · 검증 약 20초.
  thinking on은 생성·검증이 각각 1~2분 늘어 질문당 3~4분.
- 속도 개선 후보(미시도): `--speculative-algorithm mtp`, `ft bench bw` 결과에 따라 `--moe-strategy hybrid/cpu`.

---

## 6. 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `503 model is still loading` | 서버가 모델을 올리는 중(재시작·모델 교체 직후) | `ft ctl health`가 `status=ok`가 될 때까지 대기. 챗봇은 자동 대기, `eval/backend_eval.py`는 실패를 기록하지 않고 재시도 |
| `Connection error` | `LOCAL_BASE_URL` 포트 불일치(서버는 1919) | `.env`의 포트 확인 |
| `{"error":"Unauthorized"}` | API 키 누락·불일치 | `.env`의 `LOCAL_API_KEY` 확인 |
| 요청 실패(모델 없음) | 모델 교체 후 `LLM_MODEL`이 예전 id | `ft ctl health`의 model 값으로 `.env` 수정 |
| 답변이 매우 느림 | thinking on(서버 기본값) | `.env` `LOCAL_THINKING=off` 또는 챗봇 사이드바 토글 |
| 같은 말 반복 | temperature 0.0(탐욕적 디코딩) | `LLM_TEMPERATURE=0.6` |

---

## 7. 관련 파일

- `rag/llm.py` — `LLM_PROVIDER=local` 연결, thinking 플래그, `<think>` 제거
- `rag/warmup.py` — `/health` 기반 LLM 준비 상태(`llm_status`)
- `eval/backend_eval.py` — 서버 로딩 중(503) 실패를 기록하지 않고 대기·재시도
- `docs/PROGRESS_REPORT.md` §5-B — FreeToken Qwen3.6 평가 결과
