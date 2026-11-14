"""
eval/answer_quality.py — 답변 단위 품질 실험 (기준 세트 50문항, 로컬 LLM)

기준 세트(storage/human_set.jsonl)에는 문항마다 그때 검색한 근거 원문이 저장돼 있다. 같은 근거로 답만 다시 쓰게 하면
검색 차이를 빼고 '생성 프롬프트'의 효과만 잴 수 있다.

  --regen TAG [--prompt v2] [--ids a,b]  같은 근거로 답을 다시 생성해 storage/answer_runs/TAG.jsonl 에 저장(이어서 실행 가능)
  --rerun TAG [--kind multi_hop]         검색부터 파이프라인 전체를 다시 돌려 저장(검색 개선 실험용, 자동 수집·검증 끔)
  --summary TAG [TAG ...]                규칙 신호 비교: 거절 표현·(추론) 표시·인용 확인 경보 비율, 판정이 있으면 맞음 비율
  --dump TAG --out FILE [--ids ...]      판정용으로 질문·근거·이전 답·새 답을 한 파일에 모은다(근거 원문 포함 — storage 밖에 두지 말 것)

판정은 storage/answer_runs/TAG_judge.json 에 {문항 id: {"overall": correct|partial|wrong, "note", "annotator"}} 로 둔다.
annotator 가 human 이 아니면 결과는 '독립 모델 판정 기준'으로 읽는다.
결과 수치(문장 원문 없이)는 eval/results/answer_quality.json.
LLM: --regen 은 문항당 생성 1회, --rerun 은 질의 변환 1 + 생성 1회.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

RUNS = Path(settings.human_set_path).parent / "answer_runs"
OUT = Path(__file__).resolve().parent / "results" / "answer_quality.json"


def load_items() -> dict[str, dict]:
    p = Path(settings.human_set_path)
    return {r["id"]: r for r in (json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip())}


def load_run(tag: str) -> dict[str, dict]:
    if tag == "base":  # 기준 세트를 만들 때의 답(v1 + 검증·재작성 포함)
        return {k: {"id": k, "answer": v["answer"], "citation_check": v["citation_check"]} for k, v in load_items().items()}
    p = RUNS / f"{tag}.jsonl"
    if not p.exists():
        return {}
    return {r["id"]: r for r in (json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip())}


def load_judge(tag: str) -> dict[str, dict]:
    if tag == "base":
        p = Path(settings.human_labels_path)
    else:
        p = RUNS / f"{tag}_judge.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def chunks_of(item: dict):
    from rag.core.schemas import RetrievedChunk
    return [RetrievedChunk(content=x["content"], source=x["source"], title=x["title"], section=x["section"],
                           kind=x.get("kind") or "") for x in item["sources"]]


def _check(answer: str, chunks, question: str) -> dict:
    from dataclasses import asdict

    from rag.stages import citation_check as cc
    rep = cc.check(answer, chunks, question=question, nli=False)
    return {"summary": rep.summary(), "sentences": [asdict(s) for s in rep.sentences]}


def _local_only() -> None:
    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.auto_fetch = False
    settings.local_thinking = settings.local_thinking_light = "off"


def regen(tag: str, prompt: str, ids: list[str] | None) -> None:
    from rag.stages import stage3_generate as s3

    _local_only()
    settings.generation_prompt = prompt
    items = load_items()
    done = set(load_run(tag))
    todo = [k for k in (ids or items) if k not in done]
    print(f"{tag}: 프롬프트 {prompt}, 이미 {len(done)}, 이번 {len(todo)}문항", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(todo, 1):
        it, t = items[k], time.perf_counter()
        chunks = chunks_of(it)
        try:
            ans = s3.generate(it["question"], chunks)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(todo)}] {k} 실패: {e}", flush=True)
            continue
        row = {"id": k, "prompt": prompt, "answer": ans, "citation_check": _check(ans, chunks, it["question"]),
               "sec": round(time.perf_counter() - t, 1)}
        with (RUNS / f"{tag}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {k} {row['sec']}s", flush=True)


def rerun(tag: str, prompt: str, kind: str | None, ids: list[str] | None) -> None:
    """검색부터 다시(검색 개선 실험). 근거도 새로 저장한다."""
    from rag.pipeline import RAGPipeline

    _local_only()
    settings.generation_prompt = prompt
    settings.enable_verification = False
    settings.exclude_auto_papers = False
    items = load_items()
    done = set(load_run(tag))
    todo = [k for k, v in items.items() if (not kind or v["kind"] == kind) and (not ids or k in ids) and k not in done]
    print(f"{tag}: 전체 재실행, 프롬프트 {prompt}, 이번 {len(todo)}문항", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    pipe = RAGPipeline()
    for i, k in enumerate(todo, 1):
        it, t = items[k], time.perf_counter()
        try:
            r = pipe.run(it["question"])
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(todo)}] {k} 실패: {e}", flush=True)
            continue
        row = {"id": k, "prompt": prompt, "answer": r.answer, "queries": r.transformed_queries,
               "sources": [{"source": c.source, "title": c.title, "section": c.section, "kind": c.kind,
                            "content": c.content} for c in r.reranked],
               "citation_check": _check(r.answer, r.reranked, it["question"]),
               "sec": round(time.perf_counter() - t, 1)}
        with (RUNS / f"{tag}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        exp = set(it.get("expected_sources") or [])
        got = {c.source for c in r.reranked}
        print(f"[{i}/{len(todo)}] {k} {row['sec']}s 기대 논문 {len(exp & got)}/{len(exp)}", flush=True)


def signals(run: dict[str, dict], items: dict[str, dict]) -> dict:
    from eval.human_agreement import citation_flag
    from rag.ingest.auto_fetch import answer_lacks_evidence

    n = len(run)
    if not n:
        return {"n": 0}
    refuse = [k for k, r in run.items() if answer_lacks_evidence(r["answer"])]
    infer = [k for k, r in run.items() if "(추론)" in r["answer"]]
    flagged = total = 0
    for r in run.values():
        for s in r["citation_check"]["sentences"]:
            total += 1
            flagged += citation_flag(s)
    by_kind = Counter(items[k]["kind"] for k in refuse)
    out = {"n": n, "refusal_phrase": len(refuse), "refusal_by_kind": dict(by_kind), "inference_marked": len(infer),
           "sentences": total, "citation_flag_rate": round(flagged / total, 3) if total else None,
           "avg_chars": round(sum(len(r["answer"]) for r in run.values()) / n)}
    srcs = [r for r in run.values() if r.get("sources")]
    if srcs:  # 다시 검색한 실행이면 기대 논문 재현율
        hit = tot = 0
        for r in srcs:
            exp = set(items[r["id"]].get("expected_sources") or [])
            hit += len(exp & {c["source"] for c in r["sources"]})
            tot += len(exp)
        out["expected_source_recall"] = round(hit / tot, 3) if tot else None
    return out


def judged(judge: dict, ids) -> dict:
    lab = [judge[k]["overall"] for k in ids if k in judge and judge[k].get("overall")]
    if not lab:
        return {}
    c = Counter(lab)
    return {"judged": len(lab), **dict(c), "correct_rate": round(c["correct"] / len(lab), 3),
            "annotators": sorted({judge[k].get("annotator", "human") for k in ids if k in judge})}


def summary(tags: list[str]) -> None:
    items = load_items()
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for tag in tags:
        run = load_run(tag)
        j = load_judge(tag)
        res[tag] = {"signals": signals(run, items), "judge": judged(j, run),
                    "judge_by_kind": {kd: judged(j, [k for k in run if items[k]["kind"] == kd])
                                      for kd in sorted({v["kind"] for v in items.values()})}}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({t: res[t] for t in tags}, ensure_ascii=False, indent=1))


def dump(tag: str, out: str, ids: list[str] | None, with_sources: bool) -> None:
    items, run = load_items(), load_run(tag)
    base_j = load_judge("base")
    lines = []
    for k in ids or list(run):
        if k not in run:
            continue
        it, r = items[k], run[k]
        lines += [f"## {k} ({it['kind']})", f"Q: {it['question']}", ""]
        srcs = r.get("sources") or it["sources"]
        for i, c in enumerate(srcs, 1):
            body = c["content"].replace("\n", " ")
            lines.append(f"[{i}] {c['title']} / {c['section']} / {c['source']}" + (f"\n    {body}" if with_sources else ""))
        bj = base_j.get(k, {})
        lines += ["", f"--- 이전 답 (판정 {bj.get('overall')}: {bj.get('note', '')})", load_run('base')[k]['answer'],
                  "", f"--- 새 답 ({tag})", r["answer"], "",
                  "인용 확인: " + " ".join(f"S{i}:{s['status']}" for i, s in enumerate(r["citation_check"]["sentences"])), ""]
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"{out} — {len(ids or run)}문항")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen")
    ap.add_argument("--rerun")
    ap.add_argument("--prompt", default="v2")
    ap.add_argument("--kind")
    ap.add_argument("--ids")
    ap.add_argument("--summary", nargs="+")
    ap.add_argument("--dump")
    ap.add_argument("--out")
    ap.add_argument("--no-sources", action="store_true")
    a = ap.parse_args()
    ids = a.ids.split(",") if a.ids else None
    if a.regen:
        regen(a.regen, a.prompt, ids)
    elif a.rerun:
        rerun(a.rerun, a.prompt, a.kind, ids)
    elif a.summary:
        summary(a.summary)
    elif a.dump:
        dump(a.dump, a.out, ids, not a.no_sources)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
