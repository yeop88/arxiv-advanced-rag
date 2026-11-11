# Changelog

이 프로젝트의 주요 변경사항을 기록합니다. 형식은 [Keep a Changelog](https://keepachangelog.com/ko/)를 따릅니다.

## [Unreleased]

### Added
- 자동 수집 시나리오 시험 `eval/auto_fetch_scenarios.py`: 다른 학문 분야, 영어, 멀티턴, 이상한 입력, 장애, 예산 초과, 토글, 동시 질문
- 자동 수집 예산 `AUTO_FETCH_MAX_CHUNKS`(기본 2,000청크)와 사용 기록 기반 자동 정리: 넘으면 오래 안 쓰인 논문부터 삭제 (`rag/auto_fetch.py`, `storage/auto_usage.json`)
- 코퍼스 상태 점검 `eval/corpus_health.py`: 자동 수집 제외·포함 검색 지표를 비교해 정상/주의/초과 판정, 추세 기록(LLM 없음, 로컬 백엔드 전용)
- 프로젝트 필요성과 확장 방향 문서 `PROJECT_VISION.md`
- 자동 수집 실전 시험 `eval/auto_fetch_e2e.py`(시험용 DB 복사본에서 코퍼스 밖·무관한 질문·재질문·검색 지표 측정), `backend_eval.py` 기록에 최고 리랭크 점수 추가
- `manage_auto_papers.py --strip-back-matter`: 이미 수집한 논문의 참고문헌·부록 청크만 삭제
- 자동 수집 부작용 해결 방법 비교 `eval/auto_fetch_tuning.py`(관련도 기준·편수·가중치를 294문항과 코퍼스 밖 질문으로 측정, LLM 0회)
- 자동 수집 논문 리랭크 가중치 `AUTO_RERANK_WEIGHT`(기본 0.7), 참고문헌·부록 제외 `AUTO_FETCH_DROP_BACK_MATTER`(기본 켜짐), 수집 점수 메타데이터 `fetch_score`, arXiv ID 직접 조회 `fetch_arxiv_by_ids`
- 평가 코퍼스 선택 `--corpus base|all|both`: 자동 수집 논문(`origin=auto`)을 지우지 않고 평가에서만 숨김(기본 base → 이전 수치와 비교 가능), all 은 실제 챗봇 상태. 숨긴 보기 `ExcludeOriginStore`는 벡터 검색·BM25 원문 목록 양쪽에서 빼 '처음부터 없던 코퍼스'와 같은 결과 (`rag/vectorstore.py`, `eval/corpus.py`, `compare.py`, `backend_eval.py`, `EXCLUDE_AUTO_PAPERS`)
- 자동 수집 효과 측정 `eval/auto_fetch_effect.py`: 수집을 일으킨 질문들을 수집 전/후 코퍼스로 검색해 최고 점수·트리거·수집 논문 recall 비교(LLM 0회)
- `compare.py --only`: 이름으로 설정 일부만 실행(예: `--only baseline` → LLM 0회)
- 논문 자동 수집(자기 확장 RAG): 근거가 부족하면 arXiv 에서 관련 논문을 찾아 PDF 전문을 DB 에 추가하고 다시 검색·답변. 신호 2개 — ① 리랭크 최고 점수 < 0.2(평가셋 294문항 보정: 코퍼스 안 질문 4.1%만 오작동, 코퍼스 밖 8개 중 6개 감지) ② 답변이 '근거에서 찾을 수 없다'고 밝힘. 후보는 (질문, 제목+초록) 리랭크 0.3 이상만 최대 3편. 추가 논문은 `origin=auto` 표시 (`rag/auto_fetch.py`, `rag/pipeline.py`, `config.py AUTO_FETCH*`). 챗봇 사이드바 토글·진행 표시, 평가·테스트에서는 꺼짐
- 논문 검색 출처 이중화: arXiv 검색 API 가 막히면(회사망 공용 IP 429) 30분간 OpenAlex 로 검색, PDF 는 arxiv.org 에서. arXiv 검색은 직접 호출·8초 제한(라이브러리 경유 시 실패까지 15~60초) (`rag/arxiv_source.py`)
- 자동 수집 논문 관리 `scripts/manage_auto_papers.py`(목록·개별/전체 삭제, 수동 수집 논문 보호)
- FreeToken 사용 가이드 `docs/FREETOKEN.md`(v0.2.2 기준): 구조·연결 설정·주요 명령(`ft ctl`/`serve`/`daemon`)·자원 관리(켜는 순서, MoE/KV 캐시)·측정 성능·문제 해결
- 챗봇 시작 시 백그라운드 로딩 + 상태 관리: 임베딩·리랭커·벡터DB·BM25 를 미리 올리고, LLM 서버 `/health` 확인. 준비 전 질문은 진행 상황을 보여주며 대기 후 자동 진행, 사이드바 상태 2초 갱신 (`rag/warmup.py`, `ui_progress.py`, `app.py`)
- 임베딩 모델도 GPU fp16 실행(`EMBED_DEVICE`/`RERANK_DEVICE`=auto): 질의 임베딩 148→22ms, 문서 20개 14.2→0.3s, 검색 결과 상위20 겹침 99.9% (`config.py`, `rag/llm.py`)
- 챗봇 실시간 진행 표시: 질문 입력 즉시 표시, 4단계 진행 로그(만든 검색 질의·고른 논문·검증 결과·재작성 여부)와 경과 시간을 0.5초마다 갱신, 답변은 생성되는 대로 스트리밍. 완료 시 단계별 소요 시간 요약 (`ui_progress.py`, `app.py`)
- 파이프라인 진행 알림 `RAGPipeline.run(progress=...)`과 답변 토큰 스트리밍(`generate(on_token=...)`). 로컬 모델은 스트리밍 시 `<think>` 구간을 숨김 (`rag/pipeline.py`, `rag/stage3_generate.py`, `rag/llm.py get_llm(raw=)`)
- 사내 프록시 CA 묶음 생성 `scripts/export_ca_bundle.py`(certifi + Windows 인증서 저장소 → `storage/ca_bundle.pem`)와 `CA_BUNDLE` 설정 — Pinecone 연결이 인증서 검증을 유지한 채 프록시를 통과
- `.env.example` Pinecone 입력란(API 키·인덱스·네임스페이스·위치·docstore·CA_BUNDLE)
- 챗봇 사이드바 Thinking 토글: 로컬 모델(Qwen 등)의 생성·검증 / 질의변환 thinking 을 실행 중에 켜고 끔. 답변마다 사용한 모드 표시 (`app.py`)
- 평가 실패 방지: 생성·검증·질의변환·채점 실패(예: 서버 `503 model is still loading`)를 감지하면 기록하지 않고 서버 준비를 기다린 뒤 재시도, 계속 실패하면 중단 (`eval/backend_eval.py --retries/--max-wait`)
- Pinecone 벡터 DB 지원 `VECTOR_BACKEND=pinecone`: 공식 SDK 위 어댑터(`PineconeStore`)가 Chroma 와 같은 메서드 모양을 제공해 파이프라인 코드 변경 없이 전환. 원문은 로컬 docstore(JSONL)에 보관해 BM25 하이브리드 검색 유지 (`rag/vectorstore.py`, `rag/docstore.py`, `config.py`)
- Chroma → Pinecone 이전 스크립트: Chroma 벡터를 재임베딩 없이 그대로 복사(ID 유지), `--dry-run` 점검·`--reset` (`scripts/migrate_to_pinecone.py`)
- 검색 동일성 검증 `eval/vector_parity.py`(평가셋 전체로 벡터 검색 겹침률·최종 근거 일치율·recall/MRR 비교, `--fake` 오프라인 모드) + Pinecone 대역 `tests/fake_pinecone.py` + 어댑터 단위테스트
- 백엔드 단위 평가 `eval/backend_eval.py`: 질문별 지표·단계별 지연을 JSONL 로 누적 기록(중단 후 이어서 실행), `--compare` 로 기록 간 비교. 채점 thinking 고정 옵션
- 로컬 LLM 백엔드 `LLM_PROVIDER=local`: OpenAI 호환 서버(예: FreeToken Local의 Qwen3.6-35B-A3B)에 `LOCAL_BASE_URL`로 연결. 응답의 `<think>…</think>` 추론 자동 제거 (`rag/llm.py`, `config.py`)
- provider별·모델별 effort 설정: 이름·값 체계가 달라 env 분리 — `ANTHROPIC_EFFORT[_LIGHT]`(output_config.effort), `OPENAI_REASONING_EFFORT[_LIGHT]`(reasoning_effort), `LOCAL_THINKING[_LIGHT]`(enable_thinking on/off). 잘못된 값은 즉시 오류. `LLM_TEMPERATURE`·`LLM_MAX_TOKENS` env 추가 (`config.py`, `rag/llm.py`, `.env.example`)
- 다분야 논문 대량 수집: 라이브 arXiv 수집으로 코퍼스 59→**490편**. CV·RL·GNN·speech·NLP·diffusion·추천·시계열·self-supervised·**멀티모달(162편)**·**양자화(44편)** 등 다분야 커버리지(초록 기반, Windows CA 번들로 프록시 SSL 통과)
- 평가셋 대량 확대 12→**294문항**(단일홉 227·다중홉 67, easy81/medium144/hard69): `eval/gen_questions.py` 로 단일홉(논문별)·다중홉(`--multi-hop`, 임베딩 최근접 쌍) 질문을 경량 모델(`--light`)로 대량 생성 → 무료 검색지표로 자동 스크리닝(단일홉 recall=1.0·다중홉 recall≥0.5 통과분만) 승격
- 평가셋 반자동 확대 도구: 코퍼스 논문별로 '그 논문만으로 답할 수 있는' 질문을 LLM으로 생성→검수/스크리닝→승격 (`eval/gen_questions.py`, `--dry-run`/`--light`/`--multi-hop` 지원)
- 다양성 리랭킹(MMR) + 논문당 청크 상한: 전문 코퍼스에서 한 논문이 상위를 독점하는 쏠림을 막아 다중 홉 커버리지 개선 (`rag/stage2_rerank.py`, `config.rerank_diversity`/`mmr_lambda`/`same_source_penalty`/`rerank_max_per_source`). 2-arm A/B(hyde, 10문항): recall@5 0.653→0.750·answer_rel 0.754→0.771·faithfulness 유지, **tradeoff context_rel 0.625→0.546**(§5.5). 단위테스트 추가
- 학습용 Jupyter 노트북: 4단계(+0단계)를 셀 단위로 따라가며 각 단계 입출력을 확인 (`notebooks/advanced_rag_walkthrough.ipynb`)
- 평가셋 외부화: 질문을 코드에서 분리해 단일 JSONL로 관리(`eval/eval_set.jsonl` + 로더 `eval/dataset.py`) — 난이도/카테고리 필터, 코드 수정 없이 확장. `compare.py`·`evaluate.py`가 이를 로드
- 무료 검색 지표 recall@k·MRR·hit@k: `expected_sources`와 검색 결과만으로 계산(LLM 불필요) (`eval/retrieval_metrics.py`) — 대량 평가셋 저비용 스크리닝의 기반
- `compare.py` 옵션: `--retrieval-only`(생성·검증·심판 생략, 저비용) / `--difficulty` / `--category` / `--n`
- 코퍼스 대폭 확장: 샘플 논문 26→**57편** — RAG/IR·검색·임베딩 심화 10편(ColBERT·ANCE·BEIR·E5·BGE-M3·IRCoT·Self-Ask·DrQA·MS MARCO·GTR) + **주요 AI/LLM 논문 21편**(Transformer·BERT·GPT-3·InstructGPT·CoT·T5·RoBERTa·LLaMA·Llama2·Chinchilla·PaLM·GPT-4·Scaling Laws·Flan·Emergent·ReAct·ToT·LoRA·attention·word2vec·LLM Survey) (`data/sample_papers.json`)
- 전문(full-text) 수집: 다중 홉 평가 질문이 참조하는 핵심 논문 15편을 PDF 전문으로 교체 수집 → 섹션 청킹, 코퍼스 37→1,319청크. 재측정에서 설정 간 격차가 드러남(§5.3)
- arXiv 수집 옵션 `--ids`(논문 ID 직접 지정) / `--replace`(초록→전문 교체, 기존 청크 삭제 후 재적재) + `delete_sources()` (`scripts/ingest_arxiv.py`, `rag/vectorstore.py`)
- 수집 중복 제거: 이미 벡터DB에 있는 논문(source)은 재수집 시 자동 건너뜀 (`scripts/ingest_arxiv.py`, `scripts/ingest_sample.py`) — 반복 실행·누적 수집 안전
- arXiv 수집 다중 검색어: `--query` 에 여러 주제를 한 번에 지정, rate-limit 지연·재시도 내장 (`scripts/ingest_arxiv.py`)
- 로컬 문서 수집: PDF/txt/md 폴더를 섹션 청킹으로 수집 (`scripts/ingest_local.py`) — 한국어 논문(RISS/DBpia에서 받은 PDF) 경로
- 섹션 감지 한국어 확장: 초록/서론/방법/결과/결론 등 국문 섹션 제목 인식 (`rag/chunking.py`)
- LLM 응답 캐싱(SQLite): 동일 프롬프트 재호출 비용·지연 제거 (`rag/llm.py`, `config.enable_llm_cache`) — 측정상 4.6s→0.002s
- 단위 테스트: RRF/파싱/문맥화 순수함수 (`tests/test_units.py`)
- 인용 출처 패널: 챗봇 답변의 `[n]` 인용에 대응하는 제목·섹션·arXiv 링크·원문 스니펫 표시 (`app.py` render_sources, `RetrievedChunk.link`/`url`)
- 섹션 인식 청킹: 논문 전문을 섹션(Abstract/Method/Results 등)으로 나눠 청킹하고 `section` 메타 부여 (`rag/chunking.py`). 인용 헤더에 섹션 표시, 테스트 `tests/test_chunking.py`
- 멀티턴 대화: 0단계 질의 문맥화 — 후속질문을 히스토리로 독립형 질문으로 재작성 (`rag/contextualize.py`, `pipeline.run(chat_history=...)`)
- 4단계 Advanced RAG 파이프라인 (질의변환 → 리랭킹 → 생성 → 검증)
- 질의변환 3전략: multi_query / HyDE / decompose (`rag/stage1_query_transform.py`)
- 하이브리드 검색: BM25 + 벡터를 RRF로 융합 (`rag/stage2_rerank.py`, `config.use_hybrid`)
- 크로스인코더 리랭킹 (bge-reranker-v2-m3)
- 인용 기반 생성 + LLM-as-judge 검증 + 자기수정 재생성
- arXiv 수집 스크립트 (`scripts/ingest_arxiv.py`) + 오프라인 샘플 수집 (`scripts/ingest_sample.py`)
- 샘플 코퍼스 (`data/sample_papers.json`) — 초기 26편(위 항목에서 57편으로 확장)
- RAGAS 스타일 평가 (`eval/ragas_lite.py`, `eval/evaluate.py`)
- A/B 평가 자동화 — 설정 조합별 비교표 (`eval/compare.py`)
- Streamlit 챗봇형 테스트 배포 UI (`app.py`)
- 오프라인 전 파이프라인 테스트 (`tests/test_pipeline_offline.py`)
- 결정 기록(ADR): 기본 질의변환 모드 선택 근거 (`docs/ADR-001-query-transform-default.md`)
- 진행 보고서 (`docs/PROGRESS_REPORT.md`), 작업 지침 (`CLAUDE.md`)

### Changed
- 자동 수집 답변 신호 강화: 답변이 근거 부족을 밝히고 질문의 고유 명칭이 근거에 없을 때만 수집(코퍼스 안 질문의 신호 47%→23%) (`rag/auto_fetch.py answer_signal`)
- 자동 수집 후보에 OpenAlex 인용 수 순 검색 추가, 관련도 0.5 이상 중 최다 인용 논문 우선 선정(`AUTO_FETCH_CITATION_SEARCH`, `AUTO_FETCH_ANCHOR_SCORE`): 원 논문 수집 2/12→6/12
- 자동 수집 검색어당 후보 8→20개(`AUTO_FETCH_CANDIDATES`): 후보에서 잘리던 원 논문 포함. 후보 20·참고문헌 제외·가중치 0.7 조합으로 평가셋 recall 하락 −0.0085→−0.0017, 원 논문 3/7→5/7
- README 에 단계별 사용 모델·실행 위치 표와 2단계 검색 구성 요소(임베딩·BM25·RRF·리랭커·MMR) 설명 추가, GPU 안내를 현재 동작(`auto`)에 맞게 수정
- `vector_parity.py`는 자동 수집 논문을 숨기고 비교(자동 수집분은 Pinecone 에만 있어 그대로면 두 백엔드 코퍼스가 달라짐). 자동 수집 중복 판단·`manage_auto_papers.py`는 숨김 설정과 무관하게 전체를 봄
- arXiv 수집 공용 코드를 `rag/arxiv_source.py`로 분리(클라이언트·PDF 전문·청킹), `scripts/ingest_arxiv.py`는 이를 사용(동작 동일)
- 2단계(검색·리랭크) 속도 개선: 리랭크+MMR 4.74s→0.06s(294문항 평균). MMR 이 후보를 CPU 로 다시 임베딩하던 것을 벡터 스토어에 저장된 벡터를 ID 로 꺼내 쓰도록 변경(Chroma `get`·Pinecone `fetch`, 결과 동일), 리랭커 GPU fp16(294문항 중 1문항 근소한 점수 뒤집힘 → recall@5 0.9711→0.9694, `RERANK_FP16=0`으로 끌 수 있음). 검색 결과에 청크 ID 를 싣도록 함 (`rag/llm.py`, `rag/stage2_rerank.py`, `rag/vectorstore.py`, `rag/schemas.py`)
- `.env.example`을 FreeToken Local(local) 기준으로 재구성: local 설정을 기본 활성화(Qwen3.6-35B-A3B, thinking 메인 on/경량 off, temperature 0.6, max_tokens 8192), Claude·OpenAI는 주석 처리된 대안 블록으로, 백엔드 선택지별 필수 값 안내 추가
- 기본 질의변환 모드를 **`hyde` → `multi_query`로 번복**: 확장 평가셋 검색 A/B(hyde 최하)에 더해 **생성지표 2-arm 교차검증**(hard 31문항)에서도 multi_query가 모든 지표 우위/동률(faithfulness 0.953 vs 0.950)로 확정(ADR-001 재검토, §5.7). (`config.py`)
- 평가 하네스가 질문을 하드코딩 대신 `eval/eval_set.jsonl`에서 로드하도록 변경 (`eval/compare.py`, `eval/evaluate.py`)
- (이력) 기본 질의변환 모드를 `multi_query` → `hyde`로 변경했다가(10문항 소표본 근거) 위와 같이 번복 — 결정 기록 [docs/ADR-001](docs/ADR-001-query-transform-default.md)
- 평가셋을 다중 홉(교차 비교·종합) 질문으로 교체 — 난이도 변별력 확보, 설정 간 격차 검증용 (`eval/compare.py`, `eval/evaluate.py`)
- 평가셋 확대(10문항) + A/B 설정 그리드 확대(baseline/multi_query/hybrid/hyde 4종) (`eval/compare.py`)
- `answer_rate` 지표 개선: substring 매칭 → 길이+근거0건 판별(`is_answered`) — 정직한 단서가 달린 긴 답변의 오판 제거
- 프로젝트 성격을 **비영리**로 명시, 문서의 상업화/수익 서술을 공개·확장 관점으로 변경
- 파이프라인에 graceful 에러 핸들링 추가(질의변환·검증 실패 시 degrade, 생성 실패 시 안내 메시지)
- LLM 호출에 자동 재시도(`max_retries`) 적용
- arXiv 수집에 사내 프록시 SSL 대응 옵션 추가 (`--ca-bundle`, `--insecure`)
- `ingest_arxiv.py`: 전문 모드가 섹션 인식 청킹을 사용하도록 변경(청크 단위로 수집, 중복 청킹 제거)

### Fixed
- 검색어 앞 숫자가 지워지던 버그: "3D Gaussian"이 "D Gaussian", "4-bit"가 "bit"로 바뀌어 엉뚱한 논문을 검색 (`rag/stage1_query_transform.py`)
- PDF 깨진 문자로 임베딩이 실패하면 자동 수집 전체가 조용히 중단되던 문제: 문자 정리, 논문별 예외 처리 (`rag/arxiv_source.py`, `rag/auto_fetch.py`)
- arXiv 대량 수집 중 한 검색어의 HTTP 429가 예외로 전체 실행을 중단시켜 수집분이 전부 유실되던 문제 → 검색어별 격리(실패 검색어만 건너뛰고 나머지는 저장) (`scripts/ingest_arxiv.py`)
- arxiv 4.x에서 제거된 `Result.download_pdf` 사용으로 `--fulltext` 실패 → `pdf_url` 직접 다운로드(클라이언트 세션의 CA 설정 재사용)로 수정 (`scripts/ingest_arxiv.py`)
- Anthropic 최신 API 버전에서 `temperature` 파라미터 400 에러 → 제거 (`rag/llm.py`)
