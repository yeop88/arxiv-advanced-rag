"""
eval/fetch_signal_set.py — 수집 신호 평가 세트 (B1: 현실 질문)

수집이 필요한 질문(원 논문이 DB에 없음)과 필요 없는 질문(DB에 있음, 무관한 질문)을 섞어,
자동 수집을 끈 상태의 **첫 답변과 근거**를 기록한다. 수집 신호 규칙은 이 기록으로 LLM 없이 다시 판정한다.

    --record   시험용 DB 복사본에서 질문을 실행해 기록(로컬 LLM: 질의변환·생성. 검증은 끔 — 신호는 검증 전 첫 답변으로 판단)
    --judge    기록으로 현재 규칙(점수 신호 + 답변 신호)과 후보 규칙의 판정표 출력(LLM 없음)

정답(should_fetch)은 실행 시점 DB에 기대 논문이 모두 있는지로 정한다. 무관한 질문은 수집하면 안 되는 쪽으로 센다.
기록에는 논문 본문 조각이 들어 있어 git 에 올리지 않는다(.gitignore). 결과 요약은 PROGRESS_REPORT §5.22.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "fetch_signal_record.jsonl"

# 코퍼스에 있는 논문을 평소 말투로 묻는 질문(제목에 이름이 없는 논문 포함: InstructGPT·T5)
CASUAL = [
    ("트랜스포머에서 멀티헤드 어텐션은 왜 여러 개를 쓰는 거야?", ["1706.03762"]),
    ("BERT랑 RoBERTa 차이를 간단히 알려줘", ["1810.04805", "1907.11692"]),
    ("LoRA는 왜 적은 파라미터만 학습해도 되나요?", ["2106.09685"]),
    ("DPR과 ColBERT 중 어느 쪽이 검색 품질과 속도에서 유리한가요?", ["2004.04906", "2004.12832"]),
    ("Self-RAG는 반성 토큰을 어떻게 쓰나요?", ["2310.11511"]),
    ("ReAct 프롬프트는 Chain-of-Thought와 뭐가 달라요?", ["2210.03629", "2201.11903"]),
    ("InstructGPT는 RLHF를 어떤 순서로 진행하나요?", ["2203.02155"]),
    ("T5는 모든 과제를 어떻게 텍스트-투-텍스트로 바꾸나요?", ["1910.10683"]),
    ("Llama 2 채팅 모델은 안전성을 어떻게 높였나요?", ["2307.09288"]),
    ("GraphRAG는 커뮤니티 요약을 어떻게 만드나요?", ["2404.16130"]),
    ("RAG 서베이에서 말하는 Naive RAG, Advanced RAG, Modular RAG의 차이는?", ["2312.10997"]),
    ("LLaMA는 공개 데이터만으로 어떻게 높은 성능을 냈나요?", ["2302.13971"]),
]
COMPARE = [("RoBERTa와 XLNet의 사전학습 방식은 어떻게 다른가요?", ["1907.11692", "1906.08237"])]
MULTITURN = [("BERT는 사전학습을 어떻게 하나요?", "그럼 XLNet은 그것과 어떻게 다른가요?", ["1906.08237"])]


def questions() -> list[dict]:
    from eval.auto_fetch_e2e import OOC, OOC_NEW, UNRELATED
    from eval.dataset import load_eval_set
    from eval.auto_fetch_scenarios import DOMAINS, ENGLISH  # 가져올 때 설정을 바꾸므로 record()에서 먼저 호출

    qs: list[dict] = []
    qs += [{"id": f"ooc{i:02d}", "kind": "ooc", "question": q, "keys": [k]} for i, (q, k) in enumerate(OOC + OOC_NEW, 1)]
    qs += [{"id": f"dom_{n}", "kind": "ooc", "question": q, "keys": [k]} for n, q, k in DOMAINS]
    qs += [{"id": f"en{i}", "kind": "ooc", "question": q, "keys": [k]} for i, (q, k) in enumerate(ENGLISH, 1)]
    qs += [{"id": f"cmp{i}", "kind": "compare", "question": q, "keys": ks} for i, (q, ks) in enumerate(COMPARE, 1)]
    qs += [{"id": f"mt{i}", "kind": "multiturn", "question": q2, "history_q": q1, "keys": ks}
           for i, (q1, q2, ks) in enumerate(MULTITURN, 1)]
    qs += [{"id": f"casual{i:02d}", "kind": "casual", "question": q, "keys": ks} for i, (q, ks) in enumerate(CASUAL, 1)]
    qs += [{"id": f"unrel{i}", "kind": "unrelated", "question": q, "keys": []} for i, q in enumerate(UNRELATED, 1)]
    qs += [{"id": r["id"], "kind": "hard", "question": r["question"], "keys": r.get("expected_sources") or []}
           for r in load_eval_set(difficulty="hard", limit=30)]
    return qs


def record() -> None:
    from eval.auto_fetch_e2e import prepare, use_test_db

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    qs = questions()
    settings.local_thinking = "off"
    settings.local_thinking_light = "off"
    settings.exclude_auto_papers = False
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    if not done:
        prepare()  # 본 DB 를 복사해 같은 상태에서 시작(수집은 꺼서 DB 는 바뀌지 않음)
    use_test_db()
    settings.auto_fetch = False
    settings.enable_verification = False
    from rag.pipeline import RAGPipeline

    pipe = RAGPipeline()
    in_db = {(m or {}).get("source") for m in pipe.vectorstore.get(include=["metadatas"])["metadatas"]}
    print(f"질문 {len(qs)}개 (기록됨 {len(done)}개 건너뜀), DB 논문 {len(in_db)}편")
    for q in qs:
        if q["id"] in done:
            continue
        t0 = time.perf_counter()
        hist = None
        if q.get("history_q"):
            first = pipe.run(q["history_q"])
            hist = [("user", q["history_q"]), ("assistant", first.answer)]
        res = pipe.run(q["question"], chat_history=hist)
        if res.answer.startswith("답변 생성 중 오류"):  # 서버 문제를 기록으로 남기지 않음(같은 명령으로 이어서 실행)
            raise SystemExit(f"[{q['id']}] 생성 실패로 중단: {res.answer[:120]}")
        keys_in_db = all(k in in_db for k in q["keys"]) if q["keys"] else None
        row = {
            **{k: v for k, v in q.items()},
            "keys_in_db": keys_in_db,
            "should_fetch": q["kind"] != "unrelated" and keys_in_db is False,
            "standalone": res.standalone_question or q["question"],
            "answer": res.answer,
            "top_score": max((c.rerank_score or 0.0 for c in res.reranked), default=0.0),
            "keys_in_topk": [k for k in q["keys"] if k in {c.source for c in res.reranked}],
            "evidence": [{"source": c.source, "title": c.title, "content": c.content, "score": c.rerank_score}
                         for c in res.reranked],
            "sec": round(time.perf_counter() - t0, 1),
        }
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {q['id']:<28} {row['sec']:>5}s 필요={row['should_fetch']!s:<5} top={row['top_score']:.2f} "
              f"| {q['question'][:30]}", flush=True)
    print(f"완료 → {OUT}")


# ---------------- 판정(LLM 없음) ----------------

def _chunks(row):
    from types import SimpleNamespace as NS

    return [NS(title=e["title"] or "", content=e["content"] or "", rerank_score=e["score"], source=e["source"])
            for e in row["evidence"]]


def rule_current(row) -> bool:
    from rag.auto_fetch import answer_signal, needs_more

    ev = _chunks(row)
    return needs_more(ev) or answer_signal(row["standalone"], row["answer"], ev)


def rule_sentence(row) -> bool:
    """후보: '찾을 수 없다' 문장에 근거 제목에 없는 명칭이 있을 때만 답변 신호."""
    from rag import auto_fetch as af

    ev = _chunks(row)
    if af.needs_more(ev):
        return True
    if not af.answer_signal(row["standalone"], row["answer"], ev):
        return False
    sents = [s for s in re.split(r"(?<=[.다])\s+|\n", row["answer"]) if af.answer_lacks_evidence(s)]
    names = af.named_terms(row["standalone"])
    absent = [n for n in names if not any(af._in_title(n, c.title) for c in ev)]
    return any(n.lower() in s.lower() for s in sents for n in absent)


def rule_markers(row) -> bool:
    """후보: 거절 표현 확장("답할 수 없습니다" 등)."""
    from rag import auto_fetch as af

    extra = ("답할 수 없습니다", "답변할 수 없습니다", "설명할 수 없습니다")
    orig = af._NO_EVIDENCE_MARKERS
    try:
        af._NO_EVIDENCE_MARKERS = orig + extra
        return rule_current(row)
    finally:
        af._NO_EVIDENCE_MARKERS = orig


RULES = {"현재": rule_current, "문장 단위": rule_sentence, "거절 표현 확장": rule_markers}


def judge() -> None:
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    need = [r for r in rows if r["should_fetch"]]
    no = [r for r in rows if not r["should_fetch"]]
    in_corpus = [r for r in no if r["kind"] != "unrelated"]
    print(f"기록 {len(rows)}개: 수집 필요 {len(need)} / 필요 없음 {len(no)}(코퍼스 안 {len(in_corpus)}, 무관 {len(no) - len(in_corpus)})")
    print(f"{'규칙':<10} {'필요한 질문 감지':>14} {'코퍼스 안 오작동':>14} {'hard 30 오작동':>12}")
    for name, f in RULES.items():
        hit = sum(f(r) for r in need)
        fp = sum(f(r) for r in in_corpus)
        hard = [r for r in in_corpus if r["kind"] == "hard"]
        fph = sum(f(r) for r in hard)
        print(f"{name:<10} {hit:>6}/{len(need):<7} {fp:>6}/{len(in_corpus):<7} {fph:>5}/{len(hard)}")
    print("\n수집이 필요한데 현재 규칙이 놓친 질문:")
    for r in need:
        if not rule_current(r):
            print(f"  {r['id']:<20} top={r['top_score']:.2f} | {r['question'][:50]}")
    print("코퍼스 안인데 현재 규칙이 켜진 질문:")
    for r in in_corpus:
        if rule_current(r):
            print(f"  {r['id']:<20} top={r['top_score']:.2f} | {r['question'][:50]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    if args.record:
        record()
    if args.judge or not args.record:
        judge()


if __name__ == "__main__":
    main()
