"""Advanced RAG — arXiv 학술논문 어시스턴트.

  pipeline.py   4단계를 엮는 오케스트레이터(RAGPipeline)
  warmup.py     서버 시작 시 모델·DB 를 미리 올리는 로더 + LLM 서버 상태 확인
  core/         LLM·임베딩·리랭커 팩토리, 프롬프트, 데이터 구조
  stages/       0 문맥화 → 1 질의 변환 → 2 검색·리랭킹 → 3 생성 → 4 검증
  store/        벡터 DB(Chroma·Pinecone) 어댑터, 원문 저장소
  ingest/       논문 수집·청크화, 근거 부족 시 자동 수집, 표·그림·수식 이해
  features/     비교표, 문헌 조사 초안, 인용 관계, 관심 분야 알림
"""
