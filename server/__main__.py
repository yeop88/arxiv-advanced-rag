"""실행: python -m server [--host 127.0.0.1] [--port 8000] [--reload]"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="arXiv RAG 챗봇 서버")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true",
                    help="코드 변경 시 자동 재시작(개발용 — 재시작마다 모델을 다시 올림)")
    args = ap.parse_args()
    print(f"arXiv RAG 챗봇: http://{args.host}:{args.port}")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
