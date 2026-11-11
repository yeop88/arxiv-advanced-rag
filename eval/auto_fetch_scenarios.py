"""
eval/auto_fetch_scenarios.py — 자동 수집을 여러 상황에서 시험 (시험용 Chroma 복사본, 로컬 LLM 전용)

  domains     다른 학문 분야(물리·천문·수학·금융·생물·음성·로봇·보안), 옛 형식 arXiv ID 포함
  english     영어 질문
  multiturn   후속 질문에서 수집이 일어나는지, 수집한 논문을 다음 턴에서 쓰는지
  paraphrase  같은 주제를 다르게 물어도 같은 논문을 다시 넣지 않는지
  edge        빈 문자열·기호·아주 긴 질문·검색 문법 문자
  failure     검색 장애, PDF 실패, 저장 실패
  budget      수집 중 예산 초과 시 정리
  toggle      자동 수집을 끄면 수집하지 않는지
  concurrent  두 질문 동시 처리

사용:  python eval/auto_fetch_scenarios.py [--only 이름 ...]
결과:  eval/results/scenarios.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.local_thinking = "off"
settings.local_thinking_light = "off"
settings.exclude_auto_papers = False
settings.auto_fetch = True

import rag.arxiv_source as A  # noqa: E402
import rag.auto_fetch as af  # noqa: E402
from eval.auto_fetch_e2e import prepare, use_test_db  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "scenarios.jsonl"
DOMAINS = [
    ("물리", "ATLAS 실험은 힉스 보존을 어떻게 관측했나요?", "1207.7214"),
    ("천문", "LIGO는 GW150914 중력파를 어떻게 검출했나요?", "1602.03837"),
    ("양자", "Shor 알고리즘은 소인수분해를 어떻게 다항 시간에 푸나요?", "quant-ph/9508027"),
    ("수학", "Perelman의 Ricci flow 엔트로피 공식은 무엇인가요?", "math/0211159"),
    ("금융", "Deep Hedging은 파생상품 헤징을 어떻게 학습하나요?", "1802.03042"),
    ("생물", "ProtTrans는 단백질 서열을 언어 모델로 어떻게 학습하나요?", "2007.06225"),
    ("음성", "Whisper는 약지도 학습으로 음성 인식을 어떻게 하나요?", "2212.04356"),
    ("음성2", "wav2vec 2.0은 음성 표현을 어떻게 자기지도 학습하나요?", "2006.11477"),
    ("로봇", "RT-2는 웹 지식을 로봇 제어로 어떻게 옮기나요?", "2307.15818"),
    ("보안", "FGSM은 적대적 예제를 어떻게 만드나요?", "1412.6572"),
]
ENGLISH = [
    ("How does Mixtral route tokens to experts?", "2401.04088"),
    ("How does speculative decoding speed up transformer inference?", "2211.17192"),
]
rows: list[dict] = []


def ask(pipe, q, history=None):
    events: list[tuple[str, dict]] = []
    t0 = time.perf_counter()
    res = pipe.run(q, chat_history=history, progress=lambda ev, **kw: events.append((ev, kw)) if ev != "token" else None)
    return res, events, time.perf_counter() - t0


def info(res, events, sec, key=None) -> dict:
    added = [a["source"] for a in res.auto_fetched]
    return {"triggered": any(e == "fetch_start" for e, _ in events), "added": added,
            "reasons": [kw.get("reason") for e, kw in events if e == "fetch_start"],
            "titles": [a["title"][:60] for a in res.auto_fetched], "fulltext": [a["fulltext"] for a in res.auto_fetched],
            "key_added": key in added if key else None,
            "key_in_topk": key in [c.source for c in res.reranked] if key else None,
            "grounded": res.verification.grounded if res.verification else None,
            "errors": [str(kw.get("message"))[:120] for e, kw in events if e == "error"],
            "warnings": [str(kw.get("message"))[:120] for e, kw in events if e == "fetch_warn"],
            "evicted": [p["source"] for e, kw in events if e == "fetch_evicted" for p in kw["papers"]],
            "sec": round(sec, 1), "answer": res.answer[:300]}


def bad_answer(res) -> bool:
    return res.answer.startswith("답변 생성 중 오류") or not res.answer.strip()


def log(scenario: str, name: str, ok: bool | None, detail: dict) -> None:
    row = {"scenario": scenario, "name": name, "ok": ok, **detail}
    rows.append(row)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    mark = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    print(f"[{mark}] {scenario}/{name} {detail.get('sec', '')}초 수집={detail.get('triggered')} 추가={detail.get('added')} "
          f"원논문={detail.get('key_added')} top5={detail.get('key_in_topk')} {detail.get('note', '')}", flush=True)


def sources(pipe) -> set[str]:
    return {(m or {}).get("source") for m in pipe.vectorstore.get()["metadatas"]}


class force_trigger:
    """점수 신호를 강제로 켠다(기존 근거가 있어 수집이 안 일어나는 주제에서 장애 경로를 시험)."""

    def __enter__(self):
        self.orig = af.needs_more
        af.needs_more = lambda reranked: True

    def __exit__(self, *a):
        af.needs_more = self.orig


def domains(pipe):
    for name, q, key in DOMAINS:
        res, ev, sec = ask(pipe, q)
        d = info(res, ev, sec, key)
        log("domains", name, False if bad_answer(res) or d["errors"] else None, d)


def english(pipe):
    for q, key in ENGLISH:
        res, ev, sec = ask(pipe, q)
        d = info(res, ev, sec, key)
        log("english", q[:24], False if bad_answer(res) or d["errors"] else None, d)


def multiturn(pipe):
    r1, _, _ = ask(pipe, "BERT는 사전학습을 어떻게 하나요?")
    hist = [("user", "BERT는 사전학습을 어떻게 하나요?"), ("assistant", r1.answer)]
    q2 = "그럼 XLNet은 그것과 어떻게 다른가요?"
    r2, ev, sec = ask(pipe, q2, hist)
    d = info(r2, ev, sec, "1906.08237")
    d["note"] = f"문맥화: {r2.standalone_question[:50]}" if hasattr(r2, "standalone_question") else ""
    # 원 논문을 수집했거나, 근거 부족 없이 답했으면 통과("찾을 수 없다"로 끝나면 실패)
    collected = bool(d["key_added"])
    log("multiturn", "후속 질문 수집", not bad_answer(r2) and (collected or not af.answer_lacks_evidence(r2.answer)), d)
    hist += [("user", q2), ("assistant", r2.answer)]
    r3, ev, sec = ask(pipe, "그 논문의 순열 언어 모델링을 더 자세히 설명해 주세요.", hist)
    d = info(r3, ev, sec, "1906.08237")
    # 2턴에서 이미 수집했다면 다시 수집하지 않아야 한다(수집 안 했다면 이번 턴 수집은 정상)
    log("multiturn", "다음 턴 재사용", not bad_answer(r3) and not (collected and d["key_added"]), d)


def compare(pipe):
    """비교 질문: 한쪽 논문은 다른 논문에 언급만 된 상태 → 근거 부족이면 원 논문을 수집해야 한다(이슈 18)."""
    res, ev, sec = ask(pipe, "RoBERTa와 XLNet의 사전학습 방식은 어떻게 다른가요?")
    d = info(res, ev, sec, "1906.08237")
    lacks = af.answer_lacks_evidence(res.answer)
    d["note"] = f"근거 부족 답변={lacks}"
    log("compare", "RoBERTa·XLNet 비교", not bad_answer(res) and (bool(d["key_added"]) or not lacks), d)


def paraphrase(pipe):
    key = "2212.04356"
    had = key in sources(pipe)
    res, ev, sec = ask(pipe, "OpenAI의 Whisper 음성 인식 모델은 어떤 데이터로 학습되었나요?")
    d = info(res, ev, sec, key)
    d["note"] = f"이미 있던 논문={had}"
    log("paraphrase", "Whisper 다르게 묻기", not bad_answer(res) and key not in d["added"], d)


def edge(pipe):
    cases = {
        "빈 문자열": "", "공백": "   ", "기호": "???!!!", "이모지": "🍀🚀✨",
        "검색 문법": 'all:" AND (C++) OR ti:"\\ 는 무엇인가요?',
        "아주 긴 질문": "강화학습에서 탐험과 활용의 균형은 어떻게 맞추나요? " * 150,
    }
    for name, q in cases.items():
        try:
            res, ev, sec = ask(pipe, q)
            d = info(res, ev, sec)
            log("edge", name, not d["errors"] and bool(res.answer.strip()), d)
        except Exception as e:  # noqa: BLE001
            log("edge", name, False, {"note": f"예외 {type(e).__name__}: {str(e)[:150]}", "trace": traceback.format_exc()[-600:]})


def failure(pipe):
    def boom(*a, **k):
        raise RuntimeError("시험용 장애")

    q = "DETR은 객체 탐지를 어떻게 집합 예측으로 푸나요?"
    orig = (A.search_papers, A.search_openalex)
    A.search_papers = A.search_openalex = boom
    try:
        with force_trigger():
            res, ev, sec = ask(pipe, q)
    finally:
        A.search_papers, A.search_openalex = orig
    d = info(res, ev, sec)
    log("failure", "검색 전부 실패", not bad_answer(res) and not d["added"] and len(d["warnings"]) > 0, d)

    orig_pdf = A.download_pdf_text
    A.download_pdf_text = lambda *a, **k: ""
    try:
        with force_trigger():
            res, ev, sec = ask(pipe, "StyleGAN의 스타일 기반 생성자는 어떤 구조인가요?")
    finally:
        A.download_pdf_text = orig_pdf
    d = info(res, ev, sec)
    log("failure", "PDF 실패 시 초록", not bad_answer(res) and bool(d["added"]) and not any(d["fulltext"]), d)

    store, calls = pipe.vectorstore, {"n": 0}
    real_add = store.add_documents

    def flaky(docs, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("시험용 저장 실패")
        return real_add(docs, **k)

    store.add_documents = flaky
    try:
        with force_trigger():
            res, ev, sec = ask(pipe, "Swin Transformer의 이동 윈도우는 어떻게 동작하나요?")
    finally:
        del store.add_documents
    d = info(res, ev, sec)
    log("failure", "논문 1편 저장 실패", not bad_answer(res) and bool(d["added"]) and any("추가 실패" in w for w in d["warnings"]), d)


def budget(pipe):
    metas = [m or {} for m in pipe.vectorstore.get()["metadatas"]]
    auto = sum(m.get("origin") == "auto" for m in metas)
    orig = settings.auto_fetch_max_chunks
    settings.auto_fetch_max_chunks = auto + 10
    try:
        with force_trigger():
            res, ev, sec = ask(pipe, "U-Net은 의료 영상 분할을 어떻게 하나요?")
    finally:
        settings.auto_fetch_max_chunks = orig
    d = info(res, ev, sec, "1505.04597")
    now = [m or {} for m in pipe.vectorstore.get()["metadatas"]]
    srcs = {m.get("source") for m in now}
    manual = sum(m.get("origin") != "auto" for m in now)
    after = len(now) - manual
    d["note"] = f"자동 {auto} → {after}, 예산 {auto + 10}, 수동 {manual}"
    ok = bool(d["evicted"]) and after <= auto + 10 and all(a in srcs for a in d["added"]) \
        and not (set(d["evicted"]) & srcs) and not (set(d["evicted"]) & {c.source for c in res.reranked}) \
        and manual == sum(m.get("origin") != "auto" for m in metas)
    log("budget", "수집 중 예산 초과", ok, d)


def toggle(pipe):
    settings.auto_fetch = False
    try:
        with force_trigger():
            res, ev, sec = ask(pipe, "EfficientDet의 BiFPN은 어떤 구조인가요?")
    finally:
        settings.auto_fetch = True
    d = info(res, ev, sec)
    log("toggle", "자동 수집 끔", not d["triggered"] and not d["added"], d)


def concurrent(pipe):
    out: dict[str, object] = {}

    def work(name, q):
        try:
            out[name] = ask(pipe, q)
        except Exception as e:  # noqa: BLE001
            out[name] = e

    with force_trigger():
        ts = [threading.Thread(target=work, args=a) for a in
              [("a", "Mask R-CNN은 인스턴스 분할을 어떻게 하나요?"), ("b", "Faster R-CNN의 RPN은 어떻게 동작하나요?")]]
        [t.start() for t in ts]
        [t.join() for t in ts]
    for name, r in out.items():
        if isinstance(r, Exception):
            log("concurrent", name, False, {"note": f"예외 {type(r).__name__}: {str(r)[:150]}"})
        else:
            res, ev, sec = r
            d = info(res, ev, sec)
            log("concurrent", name, not bad_answer(res) and not d["errors"], d)
    try:
        json.loads(Path(settings.auto_usage_path).read_text(encoding="utf-8"))
        log("concurrent", "사용 기록 파일", True, {"note": "JSON 정상"})
    except Exception as e:  # noqa: BLE001
        log("concurrent", "사용 기록 파일", False, {"note": str(e)[:150]})


ALL = {"domains": domains, "english": english, "multiturn": multiturn, "compare": compare, "paraphrase": paraphrase,
       "edge": edge,
       "failure": failure, "budget": budget, "toggle": toggle, "concurrent": concurrent}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(ALL))
    args = ap.parse_args()
    global OUT
    if args.only:  # 일부만 돌릴 때는 전체 결과 파일을 덮어쓰지 않는다
        OUT = OUT.with_name(f"scenarios_{'_'.join(args.only)}.jsonl")
    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    prepare()  # 본 DB 를 복사해 매번 같은 상태에서 시작
    use_test_db()
    Path(settings.auto_usage_path).unlink(missing_ok=True)
    OUT.unlink(missing_ok=True)
    from rag.pipeline import RAGPipeline

    pipe = RAGPipeline()
    for name in args.only or ALL:
        try:
            ALL[name](pipe)
        except Exception as e:  # noqa: BLE001
            log(name, "시나리오 전체", False, {"note": f"예외 {type(e).__name__}: {str(e)[:200]}", "trace": traceback.format_exc()[-800:]})
    judged = [r for r in rows if r["ok"] is not None]
    print(f"\n판정 {sum(r['ok'] for r in judged)}/{len(judged)} 통과")
    for r in judged:
        if not r["ok"]:
            print(f"  FAIL {r['scenario']}/{r['name']}: {r.get('note', '')} {r.get('errors', '')}")


if __name__ == "__main__":
    main()
