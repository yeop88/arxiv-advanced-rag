"""
eval/auto_fetch_e2e.py — 자동 수집을 켠 실제 파이프라인 시험 (시험용 DB 복사본에서)

실제 DB 를 건드리지 않도록 로컬 Chroma 복사본에서 돌린다(원격 전송량 한도를 쓰지 않음).
튜닝에 쓴 질문(auto_fetch_tuning.OOC)과 겹치지 않는 새 주제로 평가한다.

단계:
    --prepare     Chroma 복사본에 자동 수집 논문을 docstore 원문으로 임베딩해 추가(LLM 없음)
    --run         코퍼스 밖 질문·무관한 질문을 자동 수집을 켜고 실행(LLM 사용)
    --repeat      수집된 질문을 다시 물어 재수집하지 않는지 확인(LLM 사용, 검증 생략)
    --retrieval   평가셋 전체 검색 지표(LLM 없음). --label before|after
    --report      결과 요약
    --cleanup     시험용 Chroma 복사본 삭제

backend_eval.py 로 시험 DB 를 평가하려면 환경변수로 지정한다:
    VECTOR_BACKEND=chroma CHROMA_DIR=./storage/chroma_e2e
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

TEST_CHROMA = "./storage/chroma_e2e"
SOURCE_CHROMA, SOURCE_DOCSTORE = "./storage/chroma", settings.docstore_path
RESULTS = Path(__file__).resolve().parent / "results"
TAG = ""  # --tag 로 정하는 결과 파일 구분자


def out_path(name: str, ext: str = "jsonl") -> Path:
    return RESULTS / f"e2e{TAG}_{name}.{ext}"


OOC = [
    ("CLIP은 이미지와 텍스트를 어떻게 함께 학습하나요?", "2103.00020"),
    ("Segment Anything 모델은 프롬프트로 어떻게 이미지를 분할하나요?", "2304.02643"),
    ("MuZero는 환경 규칙을 모르고 어떻게 계획을 세우나요?", "1911.08265"),
    ("3D Gaussian Splatting은 어떻게 실시간 렌더링을 하나요?", "2308.04079"),
    ("PPO 알고리즘의 클리핑 목적함수는 어떤 역할을 하나요?", "1707.06347"),
    ("DDPM은 노이즈를 제거하며 이미지를 어떻게 생성하나요?", "2006.11239"),
    ("YOLO는 객체 탐지를 어떻게 하나의 회귀 문제로 푸나요?", "1506.02640"),
    ("ResNet의 잔차 연결은 깊은 신경망 학습을 어떻게 돕나요?", "1512.03385"),
    ("Neural ODE는 신경망 층을 어떻게 연속적으로 모델링하나요?", "1806.07366"),
    ("SimCLR는 대조 학습으로 이미지 표현을 어떻게 배우나요?", "2002.05709"),
    ("Decision Transformer는 강화학습을 어떻게 시퀀스 모델링으로 푸나요?", "2106.01345"),
    ("GCN은 그래프의 이웃 노드 정보를 어떻게 합치나요?", "1609.02907"),
]
# 개선 작업의 근거로 쓰지 않은 새 주제(개선 효과를 치우침 없이 보기 위한 질문)
OOC_NEW = [
    ("Latent Diffusion은 잠재 공간에서 이미지를 어떻게 생성하나요?", "2112.10752"),
    ("Vision Transformer는 이미지를 패치로 나눠 어떻게 분류하나요?", "2010.11929"),
    ("MAE는 이미지 패치를 가리고 어떻게 표현을 학습하나요?", "2111.06377"),
    ("World Models 논문은 환경 모델 안에서 정책을 어떻게 학습하나요?", "1803.10122"),
    ("DreamerV3는 다양한 도메인을 어떻게 하나의 설정으로 학습하나요?", "2301.04104"),
    ("RWKV는 RNN과 트랜스포머의 장점을 어떻게 결합하나요?", "2305.13048"),
    ("PINN은 물리 법칙을 신경망 학습에 어떻게 반영하나요?", "1711.10561"),
    ("GAN의 생성자와 판별자는 어떻게 경쟁하며 학습하나요?", "1406.2661"),
]
UNRELATED = [
    "강남역 근처 맛있는 파스타집 추천해 주세요.",
    "내일 서울 날씨는 어떤가요?",
    "김치찌개를 맛있게 끓이는 방법을 알려주세요.",
    "제주도 2박 3일 여행 코스를 짜 주세요.",
    "요즘 삼성전자 주식을 사도 될까요?",
]


def use_test_db() -> None:
    settings.vector_backend = "chroma"
    settings.chroma_dir = TEST_CHROMA


def prepare() -> None:
    """본 Chroma 를 시험용으로 그대로 복사한다."""
    import shutil

    from rag.vectorstore import load_vectorstore

    settings.exclude_auto_papers = False
    if Path(TEST_CHROMA).exists():
        shutil.rmtree(TEST_CHROMA)
    shutil.copytree(SOURCE_CHROMA, TEST_CHROMA)
    use_test_db()
    metas = load_vectorstore().get()["metadatas"]
    auto = [m for m in metas if (m or {}).get("origin") == "auto"]
    print(f"복사 완료: 시험 DB {len(metas)}청크(자동 수집 {len(auto)}청크, {len({m['source'] for m in auto})}편)")


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ask(pipe, question: str) -> tuple:
    events: list[tuple[str, dict]] = []
    t0 = time.perf_counter()
    res = pipe.run(question, progress=lambda ev, **kw: events.append((ev, kw)) if ev != "token" else None)
    return res, events, time.perf_counter() - t0


def _row(kind: str, question: str, key: str | None, res, events, sec: float) -> dict:
    from rag.auto_fetch import answer_lacks_evidence

    reasons = [kw.get("reason") for ev, kw in events if ev == "fetch_start"]
    sources = sorted({s for ev, kw in events if ev == "fetch_candidates" for s in kw.get("sources", [])})
    added = [a["source"] for a in res.auto_fetched]
    return {
        "kind": kind, "question": question, "key": key, "triggered": bool(reasons), "reasons": reasons,
        "search_sources": sources, "warnings": [kw.get("message") for ev, kw in events if ev in ("fetch_warn", "error")],
        "added": added, "added_titles": [a["title"] for a in res.auto_fetched],
        "key_added": key in added if key else None,
        "key_in_topk": key in [c.source for c in res.reranked] if key else None,
        "top_score": max((c.rerank_score or 0.0 for c in res.reranked), default=0.0),
        "lacks_evidence": answer_lacks_evidence(res.answer),
        "error": res.answer.startswith("답변 생성 중 오류"),
        "grounded": res.verification.grounded if res.verification else None,
        "faithfulness": res.verification.faithfulness if res.verification else None,
        "regenerated": res.regenerated, "sec": round(sec, 1), "answer": res.answer,
    }


def run() -> None:
    from rag.pipeline import RAGPipeline

    settings.auto_fetch = True
    pipe = RAGPipeline()
    run_out = out_path("run")
    run_out.unlink(missing_ok=True)
    for kind, items in (("ooc", OOC), ("ooc_new", OOC_NEW), ("unrelated", [(q, None) for q in UNRELATED])):
        for q, key in items:
            res, events, sec = _ask(pipe, q)
            row = _row(kind, q, key, res, events, sec)
            _append(run_out, row)
            print(f"[{kind}] {sec:.0f}초 수집={row['triggered']} 추가={row['added']} 원논문={row['key_added']} "
                  f"top5={row['key_in_topk']} 근거부족={row['lacks_evidence']} | {q[:30]}", flush=True)


def repeat() -> None:
    from rag.pipeline import RAGPipeline

    settings.auto_fetch = True
    settings.enable_verification = False
    pipe = RAGPipeline()
    repeat_out = out_path("repeat")
    repeat_out.unlink(missing_ok=True)
    rows = _read(out_path("run"))
    for r in [x for x in rows if x["added"]]:
        res, events, sec = _ask(pipe, r["question"])
        row = _row(r["kind"], r["question"], r["key"], res, events, sec)
        _append(repeat_out, row)
        print(f"[재질문] {sec:.0f}초 수집={row['triggered']} 추가={row['added']} | {r['question'][:30]}", flush=True)


def retrieval(label: str) -> None:
    from eval.dataset import load_eval_set
    from eval.retrieval_metrics import recall_at_k, reciprocal_rank
    from rag.stage2_rerank import retrieve_and_rerank
    from rag.vectorstore import load_vectorstore

    settings.auto_fetch = False
    settings.query_transform_mode = "none"
    settings.use_hybrid = True
    vs = load_vectorstore()
    recalls, mrrs = [], []
    for r in load_eval_set():
        _, rr = retrieve_and_rerank(vs, r["question"], [r["question"]])
        exp = r.get("expected_sources") or []
        if exp:
            recalls.append(recall_at_k(rr, exp, settings.top_k_rerank))
            mrrs.append(reciprocal_rank(rr, exp))
    metas = [m or {} for m in vs.get()["metadatas"]]
    auto = [m for m in metas if m.get("origin") == "auto"]
    out = {"recall": sum(recalls) / len(recalls), "mrr": sum(mrrs) / len(mrrs), "chunks": len(metas),
           "auto_chunks": len(auto), "auto_papers": len({m["source"] for m in auto})}
    out_path(f"retrieval_{label}", "json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(out)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def report() -> None:
    rows, rep = _read(out_path("run")), _read(out_path("repeat"))
    bad = [r for r in rows if r["kind"] == "unrelated"]

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    for kind in ("ooc", "ooc_new"):
        ooc = [r for r in rows if r["kind"] == kind]
        print(f"{kind} {len(ooc)}개: 수집 시도 {sum(r['triggered'] for r in ooc)}, 논문 추가 {sum(bool(r['added']) for r in ooc)}, "
              f"원 논문 수집 {sum(bool(r['key_added']) for r in ooc)}, 원 논문 top-5 {sum(bool(r['key_in_topk']) for r in ooc)}, "
              f"근거 부족 답변 {sum(r['lacks_evidence'] for r in ooc)}, grounded {sum(bool(r['grounded']) for r in ooc)}, "
              f"faithfulness {avg([r['faithfulness'] for r in ooc]):.3f}, 평균 {avg([r['sec'] for r in ooc]):.0f}초")
    print(f"무관한 질문 {len(bad)}개: 수집 시도 {sum(r['triggered'] for r in bad)}, 추가 논문 {sum(len(r['added']) for r in bad)}편")
    for r in bad:
        if r["added"]:
            print(f"  {r['question'][:30]} → {r['added_titles']}")
    print(f"재질문 {len(rep)}개: 재수집 {sum(bool(r['added']) for r in rep)}, 수집 시도 {sum(r['triggered'] for r in rep)}, "
          f"평균 {avg([r['sec'] for r in rep]):.0f}초")
    print(f"검색 출처: {sorted({s for r in rows for s in r['search_sources']})}, 오류 {sum(r['error'] for r in rows)}, "
          f"경고 {[w[:90] for r in rows for w in r['warnings']][:8]}")
    for label in ("before", "after"):
        path = out_path(f"retrieval_{label}", "json")
        print(f"검색 지표 {label}:", json.loads(path.read_text(encoding="utf-8")) if path.exists() else "없음")
    for name in ("before", "after"):
        b = _read(out_path(name))
        if b:
            print(f"{name} n={len(b)} recall {avg([r['recall'] for r in b]):.4f} mrr {avg([r['mrr'] for r in b]):.4f} "
                  f"faith {avg([r['faithfulness'] for r in b]):.3f} ans_rel {avg([r['answer_relevance'] for r in b]):.3f} "
                  f"ctx_rel {avg([r['context_relevance'] for r in b]):.3f} 답변 {avg([float(r['answered']) for r in b]):.0%} "
                  f"수집 신호 {sum(bool(r.get('would_fetch')) for r in b)} 평균 {avg([r['total_sec'] for r in b]):.0f}초")


def cleanup() -> None:
    import shutil

    shutil.rmtree(TEST_CHROMA, ignore_errors=True)
    print("시험용 Chroma 복사본 삭제 완료")


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("prepare", "run", "repeat", "retrieval", "report", "cleanup"):
        ap.add_argument(f"--{name}", action="store_true")
    ap.add_argument("--label", default="after")
    ap.add_argument("--tag", default="", help="결과 파일 구분자(예: _v2)")
    args = ap.parse_args()
    global TAG
    TAG = args.tag
    if args.prepare:
        return prepare()
    if args.cleanup:
        return cleanup()
    if args.report:
        return report()
    use_test_db()
    settings.exclude_auto_papers = False
    if args.run:
        run()
    elif args.repeat:
        repeat()
    elif args.retrieval:
        retrieval(args.label)


if __name__ == "__main__":
    main()
