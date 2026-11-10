# 📚 arXiv Advanced RAG — 학술논문 어시스턴트

**질의 변환 → 리랭킹 → 생성 → 검증** 4단계 RAG 파이프라인을 각 단계가 코드에서
**눈에 보이도록** 분리해 구현한 학습용 + 제품급 품질의 **비영리** 프로젝트입니다.

- **도메인**: arXiv 학술논문 Q&A (영어 논문 기반, 한국어 질문·논문 지원)
- **스택**: LangChain · Streamlit · HuggingFace(임베딩/리랭커) · Claude/OpenAI API
- **특징**: LLM 백엔드 교체 가능, 단계별 하이퍼파라미터 실험, 검증(자기수정) 내장
- **성격**: 비영리 — 학습·연구·개인용. 상업적 수익 모델은 다루지 않습니다.

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
> (API 키는 secrets 로 주입, 벡터DB는 사전 구축).

> 💡 첫 실행 시 HuggingFace 임베딩(bge-m3)·리랭커 모델을 다운로드하므로 몇 분 걸립니다.
> GPU 가 있으면 [rag/llm.py](rag/llm.py) 에서 `device="cuda"` 로 바꾸세요.

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
│   ├── vectorstore.py         # Chroma 로드/구축
│   ├── stage1_query_transform.py
│   ├── stage2_rerank.py
│   ├── stage3_generate.py
│   ├── stage4_verify.py
│   └── pipeline.py            # 4단계 오케스트레이터
├── rag/chunking.py            # 섹션 인식 청킹 (영문/국문 섹션 감지)
├── scripts/
│   ├── ingest_arxiv.py        # arXiv 수집 (--fulltext/--ca-bundle/--insecure)
│   ├── ingest_sample.py       # 번들 샘플 수집 (네트워크 불필요)
│   └── ingest_local.py        # 로컬 PDF/텍스트 폴더 수집 (한국어 논문)
├── eval/
│   ├── eval_set.jsonl         # 평가셋 단일 소스(질문+정답논문+난이도)
│   ├── dataset.py             # 평가셋 로더(난이도/카테고리 필터)
│   ├── retrieval_metrics.py   # 무료 검색 지표 recall@k·MRR·hit@k
│   ├── gen_questions.py       # 평가셋 반자동 확대(논문별 질문 생성→검수)
│   ├── ragas_lite.py          # RAGAS 스타일 지표(LLM 심판)
│   ├── evaluate.py            # 평가 하네스
│   └── compare.py             # 설정 조합별 A/B 비교(--retrieval-only 등)
├── notebooks/
│   └── advanced_rag_walkthrough.ipynb  # 4단계 셀 단위 학습 노트북
├── docs/
│   ├── PROGRESS_REPORT.md     # 진행 보고서(A/B 결과·이슈·로드맵)
│   └── ADR-001-query-transform-default.md  # 기본 질의변환 모드 결정 기록
└── app.py                     # Streamlit 챗봇 UI
```

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
```
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
- 기능 추가/수정 이력: [CHANGELOG.md](CHANGELOG.md)

## 📝 참고

- LLM 백엔드는 [rag/llm.py](rag/llm.py) 한 파일에서 Claude ↔ OpenAI ↔ (추가 시) Ollama 로 전환됩니다.
- 비용 절감을 위해 질의변환 같은 쉬운 작업은 가벼운 모델(`LLM_MODEL_LIGHT`)을 씁니다.
