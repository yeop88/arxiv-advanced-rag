"""
eval/backend_eval.py — 현재 LLM 백엔드로 전체 파이프라인을 돌려 질문별 지표·지연을 기록

compare.py 가 "설정 조합 A/B"라면, 이 스크립트는 "백엔드/모드 하나를 깊게" 측정한다.
질문마다 결과를 JSONL 에 바로 추가하므로 중간에 멈춰도 그때까지의 기록이 남고,
같은 질문셋으로 만든 두 기록(예: thinking on/off)을 --compare 로 비교한다.

사용:
    # 로컬 Qwen, thinking off 로 hard 질문 측정
    python eval/backend_eval.py --difficulty hard --label qwen_think_off --thinking off
    # 두 기록 비교
    python eval/backend_eval.py --compare eval/results/qwen_think_off.jsonl eval/results/qwen_think_on.jsonl

공정성: 채점(answer_relevance·context_relevance)은 --judge-thinking 으로 고정(기본 off)해,
비교 대상 간에 "답변을 만드는 쪽"만 달라지게 한다. faithfulness 는 파이프라인 4단계
검증 결과를 그대로 쓴다(검증은 시스템의 일부이므로 측정 대상 설정을 따른다).
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

settings.auto_fetch = False  # 평가는 코퍼스를 고정한다(근거 부족 시 논문 자동 수집 끔)
from eval.dataset import load_eval_set  # noqa: E402
from eval.ragas_lite import answer_relevance, context_relevance, is_answered  # noqa: E402
from eval.retrieval_metrics import recall_at_k, reciprocal_rank  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _set_thinking(value: str | None) -> None:
    """로컬 백엔드의 메인 모델 thinking 을 런타임에 바꾼다(None 이면 유지)."""
    if value is not None:
        settings.local_thinking = value


def _failure(res) -> str | None:
    """파이프라인은 오류를 삼키고 안내 문구를 답변처럼 돌려준다(사용자용 graceful degrade).
    평가에서는 그런 결과를 정상 기록으로 남기면 지표가 오염되므로 실패로 판정한다."""
    from eval.ragas_lite import _NO_EVIDENCE

    if res.answer.startswith("답변 생성 중 오류"):
        return res.answer[:120]
    if _NO_EVIDENCE in res.answer:
        return None  # 근거 0건은 정상 결과
    if settings.enable_verification and res.verification is None:
        return "검증 단계 실패(verification 없음)"
    if settings.query_transform_mode != "none" and len(res.transformed_queries) <= 1:
        return "질의변환 실패(원 질문으로 대체됨 → 검색 입력이 달라짐)"
    return None


def _wait_ready(max_wait: int) -> None:
    """로컬 서버가 모델을 다시 올리는 중(503 model is still loading)이면 준비될 때까지 기다린다.
    LLM 캐시를 거치지 않도록 OpenAI 클라이언트로 직접 1토큰 요청."""
    if settings.llm_provider != "local":
        return
    from openai import OpenAI

    client = OpenAI(base_url=settings.local_base_url, api_key=settings.local_api_key, timeout=60, max_retries=0)
    t0 = time.time()
    while True:
        try:
            client.chat.completions.create(
                model=settings.llm_model, max_tokens=1, messages=[{"role": "user", "content": "ping"}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return
        except Exception as e:  # noqa: BLE001
            if time.time() - t0 > max_wait:
                raise RuntimeError(f"서버가 {max_wait}초 동안 준비되지 않음: {e}") from e
            print(f"  서버 대기 중({int(time.time() - t0)}s): {str(e)[:80]}")
            time.sleep(30)


def run(args) -> None:
    from rag.pipeline import RAGPipeline

    records = load_eval_set(difficulty=args.difficulty, category=args.category, limit=args.n)
    out = RESULTS_DIR / f"{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not args.fresh:  # 이어서 실행: 이미 기록된 질문은 건너뜀
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    elif out.exists():
        out.unlink()

    arm_thinking = args.thinking
    _set_thinking(arm_thinking)
    k = settings.top_k_rerank
    print(f"[{args.label}] provider={settings.llm_provider} model={settings.llm_model} "
          f"thinking(main)={settings.local_thinking or '-'} judge_thinking={args.judge_thinking} "
          f"질문 {len(records)}개 (기록됨 {len(done)}개 건너뜀)")
    _wait_ready(args.max_wait)
    pipe = RAGPipeline()

    for i, rec in enumerate(records, 1):
        if rec["id"] in done:
            continue
        q = rec["question"]
        for attempt in range(1, args.retries + 2):
            _set_thinking(arm_thinking)  # 측정 대상 설정으로 파이프라인 실행
            t0 = time.perf_counter()
            res = pipe.run(q)
            total = time.perf_counter() - t0
            reason = _failure(res)
            if reason is None:
                _set_thinking(args.judge_thinking)  # 채점은 고정 설정
                try:
                    ans_rel = answer_relevance(q, res.answer)
                    ctx_rel = context_relevance(q, res.reranked)
                except Exception as e:  # noqa: BLE001
                    reason = f"채점 실패: {str(e)[:100]}"
            if reason is None:
                break
            print(f"  [{rec['id']}] 시도 {attempt} 실패 → 기록하지 않고 재시도: {reason}")
            _wait_ready(args.max_wait)
        else:
            raise SystemExit(f"[{rec['id']}] {args.retries + 1}회 실패 — 오염된 기록을 남기지 않고 중단합니다. "
                             f"서버 상태 확인 후 같은 명령으로 이어서 실행하세요.")
        exp = rec.get("expected_sources") or []
        row = {
            "id": rec["id"], "difficulty": rec.get("difficulty"), "multi_hop": rec.get("multi_hop"),
            "recall": recall_at_k(res.reranked, exp, k) if exp else None,
            "mrr": reciprocal_rank(res.reranked, exp) if exp else None,
            "faithfulness": res.verification.faithfulness if res.verification else None,
            "answer_relevance": ans_rel, "context_relevance": ctx_rel,
            "answered": is_answered(res.answer), "regenerated": res.regenerated,
            "total_sec": round(total, 2),
            "timings": {kk: round(v, 2) for kk, v in (res.timings or {}).items()},
            "answer_chars": len(res.answer),
            "answer": res.answer,  # 점수가 낮은 문항의 원인을 나중에 직접 읽어 보기 위해 보관
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [{i}/{len(records)}] {rec['id']} {total:.0f}s recall={row['recall']} "
              f"faith={row['faithfulness']} ans_rel={ans_rel} ctx_rel={ctx_rel}")
    print(f"완료 → {out}")


def _summary(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def avg(key, sub=None):
        vals = [(r[key] if sub is None else r[key].get(sub)) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": len(rows), "ids": {r["id"] for r in rows},
        "recall": avg("recall"), "mrr": avg("mrr"), "faithfulness": avg("faithfulness"),
        "answer_relevance": avg("answer_relevance"), "context_relevance": avg("context_relevance"),
        "answer_rate": sum(bool(r["answered"]) for r in rows) / len(rows) if rows else None,
        "total_sec": avg("total_sec"),
        "gen_sec": avg("timings", "3_generate"), "verify_sec": avg("timings", "4_verify"),
    }


def compare(paths: list[str]) -> None:
    sums = {Path(p).stem: _summary(Path(p)) for p in paths}
    common = set.intersection(*(s["ids"] for s in sums.values()))
    print(f"공통 질문 {len(common)}개 (각 기록: " + ", ".join(f"{n}={s['n']}" for n, s in sums.items()) + ")")
    if any(s["n"] != len(common) for s in sums.values()):
        print("※ 기록된 질문이 서로 달라 공통 질문만 보려면 같은 질문셋으로 다시 측정하세요.")
    cols = ["recall", "mrr", "faithfulness", "answer_relevance", "context_relevance",
            "answer_rate", "total_sec", "gen_sec", "verify_sec"]
    print("| 기록 | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, s in sums.items():
        cells = [("-" if s[c] is None else (f"{s[c]:.0%}" if c == "answer_rate"
                  else f"{s[c]:.1f}" if c.endswith("sec") else f"{s[c]:.3f}")) for c in cols]
        print(f"| {name} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="결과 파일 이름(eval/results/<label>.jsonl)")
    ap.add_argument("--difficulty", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--thinking", choices=["on", "off"], default=None,
                    help="로컬 백엔드 메인 모델 thinking(미지정 시 .env 값)")
    ap.add_argument("--judge-thinking", choices=["on", "off"], default="off",
                    help="채점 시 thinking(비교 공정성을 위해 고정, 기본 off)")
    ap.add_argument("--fresh", action="store_true", help="기존 기록을 지우고 처음부터")
    ap.add_argument("--retries", type=int, default=3, help="문항 실패 시 재시도 횟수(실패 결과는 기록하지 않음)")
    ap.add_argument("--max-wait", type=int, default=1800, help="서버 준비 대기 최대 초(503 model loading 등)")
    ap.add_argument("--compare", nargs="+", help="비교할 결과 JSONL 경로들")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    elif args.label:
        run(args)
    else:
        ap.error("--label 또는 --compare 가 필요합니다.")


if __name__ == "__main__":
    main()
