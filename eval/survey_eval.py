"""
eval/survey_eval.py — 문헌 조사 초안(C3) 품질 측정. DB 는 읽기만 한다.

주제마다 초안을 만들어 근거 논문 수, 인용된 근거 비율, 근거 번호가 붙은 문장 비율, 4단계 검증, 시간을 기록한다.
LLM: 로컬(질의변환 1회 + 작성 1회 + 검증 1회).
    python eval/survey_eval.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "survey_eval.jsonl"
TOPICS = [
    "검색 증강 생성(RAG)의 발전 흐름",
    "밀집 검색과 후기 상호작용 검색 모델",
    "대규모 언어 모델의 효율적 미세조정",
    "다중 홉 질의응답을 위한 검색 방법",
]


def sentence_coverage(text: str) -> tuple[int, int]:
    """제목 줄을 뺀 본문 문장 중 [n] 이 붙은 문장 수 / 전체 문장 수."""
    body = " ".join(l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    # 마침표 뒤에서만 자른다("~다 [1]." 의 근거 번호가 다음 문장으로 넘어가지 않게)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 10]
    return sum(bool(re.search(r"\[\d+\]", s)) for s in sents), len(sents)


def main() -> None:
    from rag.survey import build_survey
    from rag.vectorstore import load_vectorstore

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.local_thinking = "off"
    settings.local_thinking_light = "off"
    settings.exclude_auto_papers = False
    vs = load_vectorstore()
    OUT.unlink(missing_ok=True)
    rows = []
    for topic in TOPICS:
        t0 = time.perf_counter()
        sv = build_survey(topic, vs)
        v = sv.verification
        cov, n = sentence_coverage(sv.text)
        row = {"topic": topic, "papers": len({c.source for c in sv.chunks}), "chunks": len(sv.chunks),
               "cited": len(sv.cited & set(range(1, len(sv.chunks) + 1))),
               "bad_refs": sorted(sv.cited - set(range(1, len(sv.chunks) + 1))),
               "sent_cited": cov, "sentences": n,
               "grounded": v.grounded if v else None, "faithfulness": v.faithfulness if v else None,
               "unsupported": v.unsupported_claims if v else None,
               "sec": round(time.perf_counter() - t0, 1), "markdown": sv.to_markdown()}
        rows.append(row)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{topic[:22]:<24} 논문 {row['papers']} 인용된 근거 {row['cited']}/{row['chunks']} "
              f"문장 인용 {cov}/{n} 잘못된 번호 {row['bad_refs']} grounded={row['grounded']} "
              f"faith={row['faithfulness']} {row['sec']}s", flush=True)
    print(f"\n검증 통과 {sum(bool(r['grounded']) for r in rows)}/{len(rows)}, "
          f"문장 인용 {sum(r['sent_cited'] for r in rows)}/{sum(r['sentences'] for r in rows)}, "
          f"평균 {sum(r['sec'] for r in rows) / len(rows):.0f}초")


if __name__ == "__main__":
    main()
