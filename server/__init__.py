"""FastAPI 서버 — React 화면(frontend/)에 RAG 파이프라인을 SSE 스트리밍 API 로 제공한다.

  app.py        앱 생성(라우터 연결, 빌드된 화면 제공, 시작 시 모델 예열)
  runtime.py    공유 상태: 모델 로더, LLM 서버 상태, LLM 작업 잠금 + 요청별 설정
  streaming.py  스레드 작업의 진행 알림 → SSE
  schemas.py    요청 형식, 결과 → JSON
  routes/       /api/status·config·chat·compare-table·survey·citations·watch
"""
