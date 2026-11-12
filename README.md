# 📚 arXiv Advanced RAG — 학술논문 어시스턴트

**질의 변환 → 리랭킹 → 생성 → 검증** 4단계 RAG 파이프라인을 각 단계가 코드에서
**눈에 보이도록** 분리해 구현한 학습용 + 제품급 품질의 **비영리** 프로젝트입니다.

- **도메인**: arXiv 학술논문 Q&A (영어 논문 기반, 한국어 질문·논문 지원)
- **스택**: LangChain · Streamlit · HuggingFace(임베딩/리랭커) · Claude/OpenAI API
- **특징**: LLM 백엔드 교체 가능, 단계별 하이퍼파라미터 실험, 검증(자기수정) 내장
- **성격**: 비영리 — 학습·연구·개인용. 상업적 수익 모델은 다루지 않습니다.
- **필요성과 확장 방향**: [PROJECT_VISION.md](PROJECT_VISION.md)
- **목표 달성률**: [docs/ACHIEVEMENT.md](docs/ACHIEVEMENT.md) · 발표 자료: [docs/presentation/SLIDES.md](docs/presentation/SLIDES.md)

---

## 🧭 왜 이 도메인인가

| 기준 | 이유 |
|---|---|
| 데이터 확보 | arXiv API 로 논문을 **무료·합법**으로 수집 |
| RAG 필요성 | 논문 내용은 fine-tuning 불가 → **검색 증강이 유일 해법**, 인용 출처가 생명 |
| 검증 단계 가치 | "그럴듯하지만 틀린 답"이 치명적인 영역 → **4단계 검증**이 실전처럼 빛남 |
| 학습 가치 | Advanced RAG 전 기법(질의변환·하이브리드·리랭킹·검증·멀티턴)을 한 곳에서 실습 |

---

## 🏗️ 4단계 파이프라인

```
질문 (+ 대화 히스토리)
 │
 ▼ [0단계] 멀티턴 문맥화   rag/contextualize.py
 │   후속질문을 이전 맥락으로 '독립형 질문'으로 재작성 (히스토리 없으면 통과)
 ▼
독립형 질문
 │
 ▼ [1단계] 질의 변환      rag/stage1_query_transform.py
 │   multi_query : 여러 각도로 재작성 (기본값, 근거: docs/ADR-001 재검토)
 │   hyde        : 가상 답변 생성 후 그것으로 검색
 │   decompose   : 복합 질문을 하위 질문으로 분해
 ▼
여러 검색 질의
 │
 ▼ [2단계] 검색 + 리랭킹   rag/stage2_rerank.py
 │   벡터검색(bi-encoder)으로 넓게 → 크로스인코더로 정밀 재정렬
 ▼
엄선된 근거 top-k
 │
 ▼ [3단계] 생성(인용)      rag/stage3_generate.py
 │   근거만 사용, 문장마다 [번호] 인용 → 환각 억제 + 출처 추적
 ▼
초안 답변
 │
 ▼ [4단계] 검증            rag/stage4_verify.py
 │   LLM-as-judge 로 groundedness 채점
 │   실패 시 → 3단계로 돌아가 자기수정 재생성
 ▼
최종 답변 + 검증 리포트
```

각 단계는 독립 모듈이라, 하나씩 열어 보며 학습하거나 교체 실험할 수 있습니다.
프롬프트는 전부 [rag/prompts.py](rag/prompts.py) 한 곳에 모아 두었습니다 (품질의 8할).

### 단계별 사용 모델과 실행 위치

| 단계 | 하는 일 | 사용 모델 | 실행 위치 |
|---|---|---|---|
| 0·1단계 | 문맥화, 질의 변환 | 경량 LLM | `LLM_PROVIDER`에 따름 |
| 2단계 | 질문을 벡터로 변환 | bge-m3 임베딩 | 로컬 GPU |
| 2단계 | 비슷한 청크 찾기 | 벡터 검색(모델 없음) | Pinecone 원격 또는 Chroma 로컬 |
| 2단계 | 키워드 검색(BM25) | 모델 없음(단어 통계 공식) | 로컬 CPU |
| 2단계 | 후보 청크 채점 | bge-reranker-v2-m3 리랭커 | 로컬 GPU |
| 2단계 | 다양성 선택(MMR) | 저장된 벡터 재사용, 새 임베딩 없음 | 로컬 |
| 3·4단계 | 답변 생성, 검증 | 메인 LLM | `LLM_PROVIDER`에 따름 |

- **LLM**: `local`이면 FreeToken 서버, `anthropic`·`openai`면 각 API를 호출합니다.
- **임베딩은 질문에만** 돌립니다. 논문 청크의 벡터는 수집할 때 만들어 벡터 DB에 저장해 두고 다시 계산하지 않습니다.
- **BM25는 학습된 모델이 아닙니다.** 단어 빈도, 단어가 드문 정도, 문서 길이로 점수를 계산하며, 챗봇 시작 시 전체 원문으로 메모리에 색인을 만듭니다.
- **GPU**: 임베딩·리랭커는 기본값 `auto`로 GPU가 있으면 GPU(fp16), 없으면 CPU에서 돕니다(`EMBED_DEVICE`, `RERANK_DEVICE`).
- **LLM 없이 평가**: 질의 변환을 끄고 검색 지표만 보면(`compare.py --retrieval-only --only baseline`) 임베딩·리랭커만 돌고 LLM은 호출하지 않습니다.

### 2단계 검색 자세히 보기

2단계는 넓게 모은 뒤 좁혀 가는 구조입니다. 질의 변환이 만든 검색어마다 벡터 검색과 BM25를 돌리고, 결과를 합쳐
후보 20개(`top_k_retrieve`)를 만든 다음, 리랭커와 MMR로 최종 근거 5개(`top_k_rerank`)를 고릅니다.

#### 임베딩과 벡터 검색: 뜻으로 찾기

bge-m3는 글을 1024차원 벡터로 바꾸는 다국어 모델입니다. 뜻이 비슷한 글은 벡터도 가까워지므로,
**한국어 질문으로 영어 논문을 찾을 수 있습니다.** 벡터 검색 자체는 모델이 아니라, 저장된 벡터 가운데
질문 벡터와 가까운 것을 찾는 계산입니다.

- 논문 청크는 수집할 때 한 번만 임베딩해 저장합니다. 질문할 때는 질문 문장만 임베딩합니다(GPU에서 약 22ms).
- Pinecone을 써도 임베딩은 로컬에서 만듭니다. Pinecone은 벡터를 저장하고 찾아 주는 역할이며, 원격 호출이라 검색 1회에 약 0.4초가 더 걸립니다.

#### BM25: 단어로 찾기

BM25는 학습된 신경망이 아니라 **단어 수를 세는 통계 공식**이라, 학습할 가중치도 GPU도 필요 없습니다.
질문의 단어마다 아래 세 가지를 보고 점수를 더합니다.

- **문서에 그 단어가 몇 번 나오는가**: 많을수록 높지만, 일정 수준을 넘으면 점수가 거의 늘지 않습니다.
- **그 단어가 얼마나 드문가(IDF)**: "model" 같은 흔한 단어는 낮고, "GraphCast" 같은 드문 단어는 높습니다.
- **문서 길이**: 긴 문서는 단어가 우연히 많이 나오기 쉬워서 보정합니다.

필요한 건 코퍼스의 단어 통계뿐이라, 챗봇이 시작할 때 전체 청크를 읽어 메모리에 색인을 만듭니다.
Pinecone을 쓰면서도 원문을 로컬 docstore에 두는 이유가 이것입니다.

> **한국어 질문에서의 한계**: 지금 BM25는 `BM25Retriever` 기본 설정이라 **공백으로만 단어를 나눕니다.**
> "KAN은"과 "KAN"을 다른 단어로 보는 식입니다. 논문 본문은 영어이고 질문은 한국어인 경우가 많아,
> 한국어 질문에서는 BM25가 영문 약어 정도만 맞추고 나머지는 임베딩에 의존하는 셈입니다.
> 챗봇에서는 질의 변환이 만드는 검색어가 이 약점을 일부 메워 줄 수 있지만, 그 효과를 따로 측정하지는 않았습니다.

#### 하이브리드: 두 결과 합치기

| | BM25 (키워드) | 임베딩 (의미) |
|---|---|---|
| 강한 경우 | 논문 이름, 약어, 고유명사가 정확히 일치할 때 | 표현은 달라도 뜻이 같을 때 |
| 약한 경우 | 동의어, 다른 언어 | 드문 고유명사, 숫자 |

둘의 약점이 서로 달라서 양쪽 결과를 RRF(Reciprocal Rank Fusion)로 합칩니다. RRF는 점수가 아니라 **순위**만 쓰기 때문에
척도가 다른 두 점수를 공정하게 합칠 수 있습니다(문서 점수 = 각 순위에서 `1/(60 + 순위)`의 합).
111문항 비교에서 하이브리드는 벡터만 쓸 때보다 MRR이 0.944에서 0.950으로 조금 높았습니다(recall은 0.968로 같음).

BM25 자리에 학습된 모델을 쓰는 방식도 있습니다. SPLADE 같은 희소 임베딩이나 bge-m3의 sparse 출력인데, 이 프로젝트에는 적용하지 않았습니다.

#### 리랭커: 후보를 정밀하게 다시 채점

임베딩은 질문과 문서를 **따로** 벡터로 만들어 비교하므로 빠르지만 세밀함이 떨어집니다. 리랭커(크로스인코더)는
질문과 청크를 **함께** 입력받아 관련도를 매기므로 더 정확하지만, 쌍마다 모델을 돌려야 해서 느립니다.
그래서 전체 코퍼스가 아니라 후보 20개에만 씁니다. 여기서 나온 최고 점수는 논문 자동 수집 여부를 판단하는 데도 쓰입니다.

#### MMR: 비슷한 근거 걸러내기

점수순으로만 고르면 한 논문의 청크가 상위를 독점해, 여러 논문을 비교하는 질문에서 근거가 한쪽으로 쏠립니다.
MMR은 관련도(`λ=0.6`)와 이미 고른 청크와의 유사도를 함께 고려해 고르고, 한 논문은 최대 2개까지만 뽑습니다.
다중 홉 질문에서 recall@5가 0.73에서 0.80으로 올랐습니다.

유사도 계산에는 벡터 DB에 저장된 벡터를 꺼내 쓰므로 새로 임베딩하지 않습니다. 이 변경으로 리랭크와 MMR 구간이 평균 4.74초에서 0.06초로 줄었습니다.

---

## 🚀 빠른 시작

```bash
# 1) 가상환경 + 의존성
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt

# 2) 환경변수 설정
copy .env.example .env          # 기본은 FreeToken Local(local): LOCAL_BASE_URL·LLM_MODEL 입력
                                #   Claude 를 쓰려면 LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY

# 3) 논문 수집 → 벡터DB 구축
#    (a) 네트워크 가능 환경: arXiv 실수집 (검색어 여러 개 가능, 이미 받은 논문은 자동 중복 제외)
python scripts/ingest_arxiv.py --max 10 --query "retrieval augmented generation" "passage reranking"
#        └ 전문 파싱 + 섹션 청킹: --fulltext (PDF 받아 Abstract/Method/Results 로 분리)
#        └ 초록→전문 업그레이드: --ids 2310.11511 2004.04906 --fulltext --replace (기존 청크 교체)
#        └ 사내 프록시(SSL) 환경이면: --ca-bundle <회사CA.pem> (권장) 또는 --insecure
#          (Windows는 OS 인증서 저장소를 PEM으로 내보내 --ca-bundle 로 주면 검증 유지한 채 통과)
#    (b) 네트워크 불가/빠른 데모: 번들 샘플 57편 (RAG/IR 심화 + 주요 AI/LLM 논문, 재실행 시 중복 제외)
python scripts/ingest_sample.py
#    (c) 한국어 논문 등 로컬 PDF/텍스트 폴더 (RISS/DBpia에서 받은 PDF 투입)
python scripts/ingest_local.py --dir ./my_papers --reset

# 4) 챗봇 실행
streamlit run app.py
```

> 💬 **챗봇형 UI**: 대화창에 질문하면 근거 기반 답변 + 검증 배지 + **📑 근거 출처 패널**
> (각 `[n]` 인용의 제목·섹션·arXiv 링크·원문) + "단계별 내부 동작"을 펼쳐 볼 수 있습니다.
> **멀티턴 지원** — "그럼 둘 중 뭐가 더 느려?" 같은 후속질문도 이전 맥락을 반영해 답합니다
> (0단계 질의 문맥화). 사내 서버/Streamlit Cloud 에 그대로 배포 가능
> **⏱️ 실시간 진행 표시** — 질문하면 4단계가 각각 무엇을 하는지(만든 검색 질의, 고른 논문, 검증 결과)와 경과 시간이
> 계속 갱신되고, 답변은 써지는 대로 보입니다. 로컬 모델처럼 오래 걸려도 어디까지 왔는지 알 수 있습니다.
> **📥 논문 자동 수집** — DB 에 맞는 근거가 없으면 arXiv 에서 관련 논문을 찾아 **전문**을 DB 에 추가하고 다시 답합니다
> (관련도 기준을 넘는 논문만, 인용이 많은 원 논문 우선, 질문당 최대 3편, 참고문헌·부록은 제외, 사이드바에서 끄기 가능). 추가된 논문은 `python scripts/manage_auto_papers.py`로 보고 지울 수 있습니다.
> 평가는 지우지 않고도 자동 수집분을 빼고(기본) 또는 넣고 할 수 있습니다(`--corpus`, 아래 학습·평가 도구).
> 코퍼스가 커지면 검색 품질이 조금씩 내려가므로, 자동 수집은 **2,000청크까지만** 유지하고 넘으면 오래 안 쓰인 논문부터 지웁니다(`AUTO_FETCH_MAX_CHUNKS`). `python eval/corpus_health.py`로 상태를 점검할 수 있습니다.
> **🧠 Thinking 토글**(로컬 모델 전용) — 사이드바에서 생성·검증 / 질의변환의 thinking 을 켜고 끌 수 있습니다.
> 켜면 다중 홉 비교에 유리하지만 느립니다(측정: 질문당 off 약 1분, on 약 3~4분).
> (API 키는 secrets 로 주입, 벡터DB는 사전 구축).

> 💡 첫 실행 시 HuggingFace 임베딩(bge-m3)·리랭커 모델을 다운로드하므로 몇 분 걸립니다.
> GPU 가 있으면 자동으로 사용합니다(`EMBED_DEVICE`·`RERANK_DEVICE`, 기본 `auto`).

### 평가 실행

```bash
python eval/evaluate.py         # RAGAS 스타일 3지표 측정
python eval/compare.py --n 3    # 설정 조합별 A/B 비교표 (하이브리드 ON/OFF 등)
```

---

## 📂 구조

```
exam03_Advanced_RAG/
├── config.py                  # 전역 설정 (모델·청크·top_k 등)
├── rag/
│   ├── prompts.py             # ⭐ 4단계 프롬프트 (학습 핵심)
│   ├── schemas.py             # 단계별 데이터 구조
│   ├── llm.py                 # LLM/임베딩/리랭커 팩토리 (교체 가능)
│   ├── vectorstore.py         # 벡터 DB 로드/구축 (Chroma | Pinecone 어댑터)
│   ├── docstore.py            # 원문 보관소(JSONL) — Pinecone 사용 시 BM25·본문 기준
│   ├── stage1_query_transform.py
│   ├── stage2_rerank.py
│   ├── stage3_generate.py
│   ├── stage4_verify.py
│   └── pipeline.py            # 4단계 오케스트레이터
├── rag/chunking.py            # 섹션 인식 청킹 (영문/국문 섹션 감지)
├── scripts/
│   ├── ingest_arxiv.py        # arXiv 수집 (--fulltext/--ca-bundle/--insecure)
│   ├── ingest_sample.py       # 번들 샘플 수집 (네트워크 불필요)
│   ├── ingest_local.py        # 로컬 PDF/텍스트 폴더 수집 (한국어 논문)
│   ├── migrate_to_pinecone.py # Chroma → Pinecone 이전(벡터 그대로 복사, --dry-run)
│   └── manage_auto_papers.py  # 자동 수집 논문(origin=auto) 목록·삭제
├── eval/
│   ├── eval_set.jsonl         # 평가셋 단일 소스(질문+정답논문+난이도)
│   ├── dataset.py             # 평가셋 로더(난이도/카테고리 필터)
│   ├── retrieval_metrics.py   # 무료 검색 지표 recall@k·MRR·hit@k
│   ├── gen_questions.py       # 평가셋 반자동 확대(논문별 질문 생성→검수)
│   ├── ragas_lite.py          # RAGAS 스타일 지표(LLM 심판)
│   ├── evaluate.py            # 평가 하네스
│   ├── compare.py             # 설정 조합별 A/B 비교(--retrieval-only 등)
│   ├── backend_eval.py        # 현재 LLM 백엔드로 질문별 지표·지연 기록 + 기록 간 비교
│   ├── corpus.py              # 평가 코퍼스 선택(--corpus base|all|both: 자동 수집 논문 제외/포함)
│   ├── auto_fetch_effect.py   # 자동 수집 전/후 검색 비교(수집을 일으킨 질문들)
│   ├── auto_fetch_tuning.py   # 자동 수집 설정별(기준·편수·가중치) 부작용 비교
│   ├── auto_fetch_e2e.py      # 자동 수집 실전 시험(시험용 DB 복사본)
│   ├── auto_fetch_scenarios.py  # 자동 수집 시나리오 시험(분야·멀티턴·장애·예산 등)
│   ├── corpus_health.py       # 코퍼스 증가에 따른 검색 품질 점검
│   └── vector_parity.py       # Chroma vs Pinecone 검색 결과 동일성 검증
├── tests/                     # 단위·오프라인 테스트 (fake_pinecone.py: Pinecone 대역)
├── notebooks/
│   └── advanced_rag_walkthrough.ipynb  # 4단계 셀 단위 학습 노트북
├── docs/
│   ├── PROGRESS_REPORT.md     # 진행 보고서(A/B 결과·이슈·로드맵)
│   ├── ADR-001-query-transform-default.md  # 기본 질의변환 모드 결정 기록
│   └── FREETOKEN.md           # FreeToken(로컬 LLM 서버) 사용법·자원 관리·문제 해결 (v0.2.2)
├── ui_progress.py             # 챗봇 실시간 진행 표시(백그라운드 실행 + 0.5초 갱신)
└── app.py                     # Streamlit 챗봇 UI
```

---

## 🗄️ 벡터 DB 전환 (Chroma → Pinecone)

성능 저하 없이 옮기도록 설계했습니다.

- **벡터를 다시 만들지 않고 Chroma 의 벡터를 그대로 복사**(ID 유지) → 임베딩이 완전히 동일.
- 인덱스는 **cosine**: 벡터가 길이 1로 정규화돼 있어 Chroma(L2)와 순위가 같습니다.
- **원문은 로컬 docstore(JSONL)에도 저장** → 하이브리드 검색의 BM25 가 Chroma 때와 같은 원문을 사용.

```bash
python scripts/migrate_to_pinecone.py --dry-run   # 키 없이 점검
python scripts/migrate_to_pinecone.py             # 이전 (PINECONE_API_KEY 필요)
python eval/vector_parity.py                      # 평가셋 전체로 Chroma 와 결과 비교
# 통과하면 .env 에 VECTOR_BACKEND=pinecone
```
키가 없을 때는 `python eval/vector_parity.py --fake` 로 메모리 내 대역에 같은 벡터를 넣어 어댑터를 검증할 수 있습니다.
회사망 프록시 때문에 `CERTIFICATE_VERIFY_FAILED` 가 나면 `python scripts/export_ca_bundle.py` 를 한 번 실행하세요(인증서 검증은 유지).

---

## 🔧 실험 포인트 (학습용)

사이드바 또는 [config.py](config.py) 에서 바꿔보며 품질 변화를 관찰하세요.

- `query_transform_mode`: 기본 `multi_query`(111문항 검색 A/B 재검토 근거, [docs/ADR-001](docs/ADR-001-query-transform-default.md)) ↔ `hyde` ↔ `decompose` 비교
- `top_k_retrieve` / `top_k_rerank`: 리랭킹이 실제로 품질을 올리는지
- `rerank_diversity` / `rerank_max_per_source`: 다양성 리랭킹(MMR) on/off — 다중 홉 질문에서 한 논문 쏠림 방지(recall@5 0.73→0.80)
- `enable_verification`: 검증 on/off 로 환각률 비교
- `LLM_MODEL`: Claude ↔ OpenAI, 큰 모델 ↔ 작은 모델
- **Effort (provider별·모델별 분리)** — 높을수록 깊게 추론하지만 토큰·지연↑. 비우면 기본값:

  | provider | 메인 / 경량 env | 전달 방식 | 값 |
  |---|---|---|---|
  | `anthropic` | `ANTHROPIC_EFFORT` / `_LIGHT` | `output_config.effort` | low·medium·high·xhigh·max |
  | `openai` | `OPENAI_REASONING_EFFORT` / `_LIGHT` | `reasoning_effort` | none·minimal·low·medium·high·xhigh |
  | `local` (Qwen3 등) | `LOCAL_THINKING` / `_LIGHT` | `chat_template_kwargs.enable_thinking` | on·off |

- **로컬 모델(OpenAI 호환 서버)**: `LLM_PROVIDER=local`, `LOCAL_BASE_URL`, `LLM_MODEL`(서버의 모델 id)을 설정.
  응답에 섞인 `<think>…</think>` 추론은 자동 제거됩니다. 자세한 예시는 `.env.example` 참고.

**학습·평가 도구:**
```bash
jupyter notebook notebooks/advanced_rag_walkthrough.ipynb   # 4단계 셀 단위로 따라가기
python eval/compare.py                    # 설정별 A/B (검색+생성 지표)
python eval/compare.py --retrieval-only   # LLM 없이 검색 품질만 저비용 스크리닝
python eval/compare.py --difficulty hard  # 난이도 필터
python eval/compare.py --retrieval-only --only baseline --corpus both   # LLM 0회, 두 코퍼스 나란히
python eval/auto_fetch_effect.py          # 자동 수집 전/후 비교(수집을 일으킨 질문들, LLM 0회)
```
**평가 코퍼스**: 챗봇이 자동 수집한 논문은 지우지 않고, 평가할 때만 골라 씁니다.
기본값 `--corpus base`는 자동 수집 논문을 숨긴 기준 코퍼스라 이전 수치와 그대로 비교됩니다.
`--corpus all`은 자동 수집분까지 포함한 실제 챗봇 상태, `both`는 둘 다 측정합니다(`compare.py`, `backend_eval.py`).
평가셋은 `eval/eval_set.jsonl` 한 줄 = 질문 하나. 질문·정답논문(`expected_sources`)을
추가하면 코드 수정 없이 평가셋이 커집니다. 대량으로 늘릴 땐:
```bash
python eval/gen_questions.py --n 20   # 코퍼스에서 질문 후보 생성(사람 검수 후 eval_set.jsonl 로 승격)
```

---

## 🧩 확장 로드맵 (비영리)

1. **데이터 확대**: 초록 → PDF 전문(`--fulltext`, 섹션 청킹) ✅ / 로컬 PDF 폴더 수집 ✅
2. **한국어 강화**: 국내 논문 PDF 로컬 수집(RISS·DBpia에서 본인 권한으로 내려받아 폴더 투입)
3. **품질**: 하이브리드 검색 ✅ / 멀티턴 ✅ / 인용 출처 패널 ✅ / RAGAS 평가 ✅ / 응답 캐싱 ✅
4. **학습·공개**: 노트북형 설명 ✅ / 평가셋 외부화·무료 검색지표 ✅ / 평가셋 대량 확대(진행) / 공개 데모(배포)

---

## 📋 진행 보고서 · 변경 이력

- 의사결정·구현·검증·이슈·로드맵: [docs/PROGRESS_REPORT.md](docs/PROGRESS_REPORT.md)
- 로컬 LLM 서버(FreeToken) 사용법: [docs/FREETOKEN.md](docs/FREETOKEN.md)
- 기능 추가/수정 이력: [CHANGELOG.md](CHANGELOG.md)

## 📝 참고

- LLM 백엔드는 [rag/llm.py](rag/llm.py) 한 파일에서 Claude ↔ OpenAI ↔ (추가 시) Ollama 로 전환됩니다.
- 비용 절감을 위해 질의변환 같은 쉬운 작업은 가벼운 모델(`LLM_MODEL_LIGHT`)을 씁니다.
