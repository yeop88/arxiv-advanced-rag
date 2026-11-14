"""
eval/contextual_header_eval.py — 본문 청크에 맥락(논문 제목·섹션) 붙이기 실험 (LLM 없음, 시험용 DB 사본)

표·그림·수식 청크는 앞에 논문 제목을 붙여 수식 적중이 1/5 → 3/5 가 됐다(§5.38). 본문 청크는 "We propose…" 처럼
어느 논문인지 드러나지 않는 문단이 많아, 논문 이름을 넣은 질문이 다른 논문의 비슷한 문단에 밀릴 수 있다.
저장된 본문 글(LLM 이 읽는 근거)은 그대로 두고, 검색에 쓰는 입력에만 `제목 · 섹션` 머리말을 붙여 비교한다:

  A 기준       지금 그대로
  B 임베딩     본문 청크 벡터를 '머리말 + 본문'으로 다시 계산(저장 글은 그대로)
  C B + BM25   BM25 색인도 머리말 포함
  D C + 리랭커 리랭커 입력도 머리말 포함

측정: 평가셋 294문항(기준 코퍼스 = 자동 수집 제외, 질의 변환 없이)과 B1 현실 질문(정답 논문이 DB 에 있는 것, 전체 코퍼스)의
recall@5·MRR, 문항별 좋아짐/나빠짐. 표·그림·수식 청크는 이미 제목이 붙어 있어 건드리지 않는다.

사용: python eval/contextual_header_eval.py [--work storage/_ctx_test]
결과: eval/results/contextual_header_eval.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

settings.auto_fetch = False
settings.query_transform_mode = "none"   # LLM 미사용
settings.use_hybrid = True

from eval.corpus import set_corpus  # noqa: E402
from eval.dataset import load_eval_set  # noqa: E402
from eval.retrieval_metrics import recall_at_k, reciprocal_rank  # noqa: E402
from rag.core import llm as L  # noqa: E402
from rag.core.schemas import VISUAL_KINDS  # noqa: E402
from rag.stages import stage2_rerank as s2  # noqa: E402
from rag.store import vectorstore as V  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "contextual_header_eval.json"
B1 = Path(__file__).resolve().parent / "results" / "fetch_signal_record.jsonl"


def header(meta: dict) -> str:
    """검색 입력에 붙일 머리말: '논문 제목 · 섹션'."""
    parts = [(meta.get("title") or "").strip(), (meta.get("section") or "").strip()]
    return " · ".join(p for p in parts if p)


def with_header(meta: dict, text: str) -> str:
    h = header(meta)
    return f"{h}\n{text}" if h and not text.startswith(h) else text


def question_sets() -> dict[str, list[dict]]:
    sets = {"eval294": [{"id": r["id"], "q": r["question"], "keys": r.get("expected_sources") or []}
                        for r in load_eval_set()]}
    if B1.exists():
        rows = [json.loads(x) for x in B1.read_text(encoding="utf-8").splitlines() if x.strip()]
        # 평가셋과 겹치는 hard 는 빼고, 정답 논문이 지금 DB 에 있는 것만
        sets["b1"] = [{"id": r["id"], "q": r.get("standalone") or r["question"], "keys": r["keys"]}
                      for r in rows if r["kind"] != "hard" and r.get("keys")]
    return sets


def measure(qs: list[dict], corpus: str) -> dict[str, tuple[float, float]]:
    set_corpus(corpus)
    s2.reset_bm25()
    vs = V.load_vectorstore()
    out = {}
    for r in qs:
        _, rr = s2.retrieve_and_rerank(vs, r["q"], [r["q"]])
        out[r["id"]] = (recall_at_k(rr, r["keys"], settings.top_k_rerank), reciprocal_rank(rr, r["keys"]))
    return out


def summarize(per: dict[str, tuple[float, float]], base: dict[str, tuple[float, float]] | None) -> dict:
    n = len(per)
    row = {"n": n, "recall": round(sum(v[0] for v in per.values()) / n, 4),
           "mrr": round(sum(v[1] for v in per.values()) / n, 4)}
    if base is not None:
        row["better"] = [k for k in per if (per[k][0], per[k][1]) > (base[k][0], base[k][1])]
        row["worse"] = [k for k in per if (per[k][0], per[k][1]) < (base[k][0], base[k][1])]
    return row


def reembed_body(chroma) -> int:
    """본문 청크의 벡터만 '머리말 + 본문'으로 다시 계산해 바꾼다(저장 글·메타데이터는 그대로)."""
    col = chroma._collection
    d = col.get(include=["documents", "metadatas"])
    ids, texts = [], []
    for i, t, m in zip(d["ids"], d["documents"], d["metadatas"]):
        m = m or {}
        if m.get("kind") in VISUAL_KINDS:
            continue
        h = with_header(m, t)
        if h != t:
            ids.append(i)
            texts.append(h)
    emb = L.get_embeddings()
    for k in range(0, len(ids), 256):
        col.update(ids=ids[k:k + 256], embeddings=emb.embed_documents(texts[k:k + 256]))
    return len(ids)


def patch_bm25() -> None:
    """BM25 색인을 '머리말 + 본문'으로 만들되, 돌려주는 문서는 원래 글로(RRF·근거가 같은 청크로 묶이게)."""
    from langchain_core.documents import Document

    orig_get_all = V.get_all_documents

    def get_all(vs):
        out = []
        for d in orig_get_all(vs):
            if d.metadata.get("kind") in VISUAL_KINDS:
                out.append(d)
            else:
                m = dict(d.metadata, _body=d.page_content)
                out.append(Document(page_content=with_header(d.metadata, d.page_content), metadata=m, id=d.id))
        return out

    class Unwrap:
        def __init__(self, r):
            self.r = r

        def invoke(self, q):
            return [Document(page_content=d.metadata.get("_body", d.page_content),
                             metadata={k: v for k, v in d.metadata.items() if k != "_body"}, id=d.id)
                    for d in self.r.invoke(q)]

    s2.get_all_documents = get_all
    inner = s2._get_bm25.__wrapped__

    def get_bm25(vid):
        body, visual, by_source = inner(vid)
        return Unwrap(body), visual, by_source

    s2._get_bm25 = s2.lru_cache(maxsize=4)(get_bm25)
    s2.reset_bm25 = s2._get_bm25.cache_clear


def patch_reranker(meta_by_text: dict[str, dict]) -> None:
    """리랭커가 본문 청크를 '머리말 + 본문'으로 보게 한다."""
    real = L.get_reranker()

    class R:
        def predict(self, pairs, **kw):
            return real.predict([(q, with_header(meta_by_text[p], p) if p in meta_by_text else p) for q, p in pairs], **kw)

    s2.get_reranker = lambda: R()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="storage/_ctx_test", help="시험용 DB 사본 위치(매번 새로 복사)")
    args = ap.parse_args()
    if settings.vector_backend != "chroma":
        raise SystemExit("로컬 백엔드(VECTOR_BACKEND=chroma)에서만 실행합니다.")
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(settings.chroma_dir, work)
    settings.chroma_dir = str(work)
    print(f"시험용 사본: {work}")

    sets = question_sets()
    plan = [("eval294", "base"), ("b1", "all")] if "b1" in sets else [("eval294", "base")]
    results: dict = {}

    def run(variant: str) -> None:
        t = time.perf_counter()
        results[variant] = {}
        for name, corpus in plan:
            per = measure(sets[name], corpus)
            base = results.get("A_baseline", {}).get(name, {}).get("_per")
            results[variant][name] = {**summarize(per, base), "_per": per}
        print(variant, {k: {kk: vv for kk, vv in v.items() if kk in ("recall", "mrr")} for k, v in results[variant].items()},
              f"{time.perf_counter() - t:.0f}s", flush=True)

    run("A_baseline")
    set_corpus("all")
    n = reembed_body(V._load_chroma())
    print(f"본문 청크 {n}개 벡터를 머리말 포함으로 다시 계산")
    run("B_embed")
    patch_bm25()
    run("C_embed_bm25")
    d = V._load_chroma()._collection.get(include=["documents", "metadatas"])
    patch_reranker({t: (m or {}) for t, m in zip(d["documents"], d["metadatas"]) if (m or {}).get("kind") not in VISUAL_KINDS})
    run("D_embed_bm25_rerank")

    for v in results.values():
        for s in v.values():
            s.pop("_per", None)
    OUT.write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "reembedded": n, "results": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    if work.exists():   # Windows 에서는 Chroma 가 파일을 잡고 있어 못 지울 수 있다
        print(f"시험용 사본을 지우지 못했습니다(파일 사용 중) — 직접 지우세요: {work}")


if __name__ == "__main__":
    main()
