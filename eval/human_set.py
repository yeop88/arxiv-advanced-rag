"""
eval/human_set.py — 사람 판정 기준 세트 만들기 (로컬 LLM 사용)

지금 답변 품질 점수는 로컬 모델이 자기 답을 채점한 값이다(4단계 검증·RAGAS 스타일). 그 점수와
문장 단위 인용 확인(4b)이 사람 판단과 얼마나 맞는지 재려면 사람이 직접 판정한 기준 세트가 필요하다.

  --build : 50문항(단일 논문 20 · 다중 홉 18 · 표·그림·수식 12, 고정 시드)을 실제 파이프라인으로 답하고
            답변·근거 원문·LLM 검증·문장별 인용 확인을 storage/human_set.jsonl 에 저장(이미 만든 문항은 건너뜀)
  판정    : 챗봇 화면 사이드바 '사람 판정' → 문장마다 근거 있음/없음/번호 틀림, 답변 전체 맞음/일부/틀림
            (storage/human_labels.json)
  일치율  : python eval/human_agreement.py

LLM: 문항당 질의 변환 1 + 생성 1 + 검증 1~3회(재작성 시). 자동 수집은 끔, thinking off.
근거 원문이 들어 있어 storage/ 에 둔다(공개 저장소에 올리지 않음).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

SET_PATH = Path(settings.human_set_path)
LABEL_PATH = Path(settings.human_labels_path)
N_SINGLE, N_MULTI, N_VISUAL = 20, 18, 12


def pick() -> list[dict]:
    """고정 시드로 고른 50문항."""
    from eval.dataset import load_eval_set
    from eval.visual_qa_eval import QUESTIONS

    rng = random.Random(15)
    rows = load_eval_set()
    single = [r for r in rows if not r.get("multi_hop")]
    multi = [r for r in rows if r.get("multi_hop")]
    out = [{"id": r["id"], "kind": "single", "question": r["question"], "expected_sources": r.get("expected_sources", [])}
           for r in rng.sample(single, N_SINGLE)]
    out += [{"id": r["id"], "kind": "multi_hop", "question": r["question"], "expected_sources": r.get("expected_sources", [])}
            for r in rng.sample(multi, N_MULTI)]
    vis = rng.sample(QUESTIONS, N_VISUAL)
    out += [{"id": f"vis_{i:02d}", "kind": kind, "question": q, "expected_sources": [src]}
            for i, (q, _exp, src, kind) in enumerate(vis, 1)]
    return out


def load_set() -> list[dict]:
    if not SET_PATH.exists():
        return []
    return [json.loads(x) for x in SET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


def build() -> None:
    from rag.pipeline import RAGPipeline
    from rag.stages import citation_check as cc

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.auto_fetch = False
    settings.local_thinking = settings.local_thinking_light = "off"
    settings.enable_verification = True
    settings.exclude_auto_papers = False
    done = {r["id"] for r in load_set()}
    todo = [q for q in pick() if q["id"] not in done]
    print(f"기준 세트: 이미 {len(done)}문항, 이번에 {len(todo)}문항", flush=True)
    pipe = RAGPipeline()
    SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(todo, 1):
        t = time.perf_counter()
        try:
            r = pipe.run(q["question"])
        except Exception as e:  # noqa: BLE001 — 한 문항 실패가 나머지를 막지 않게
            print(f"[{i}/{len(todo)}] {q['id']} 실패: {e}", flush=True)
            continue
        rep = cc.check(r.answer, r.reranked, question=r.standalone_question)
        v = r.verification
        row = {
            **q, "answer": r.answer, "regenerated": r.regenerated,
            "sources": [{"source": c.source, "title": c.title, "section": c.section, "kind": c.kind,
                         "content": c.content} for c in r.reranked],
            "llm_verification": None if v is None else {"grounded": v.grounded, "faithfulness": v.faithfulness,
                                                        "unsupported": v.unsupported_claims},
            "citation_check": {"threshold": rep.threshold, "summary": rep.summary(),
                               "sentences": [asdict(s) for s in rep.sentences]},
            "sec": round(time.perf_counter() - t, 1), "built_at": time.strftime("%Y-%m-%d %H:%M"),
        }
        with SET_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        s = rep.summary()
        print(f"[{i}/{len(todo)}] {q['id']} {row['sec']}s 문장 {s['claims']} 인용확인 {s['supported']}/{s['claims']}"
              f" 검증 {None if v is None else v.grounded}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--list", action="store_true", help="고를 50문항만 출력")
    args = ap.parse_args()
    if args.list:
        for q in pick():
            print(q["id"], q["kind"], q["question"][:60])
    elif args.build:
        build()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
