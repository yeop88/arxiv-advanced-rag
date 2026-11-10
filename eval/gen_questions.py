"""
eval/gen_questions.py — 코퍼스에서 평가 질문 후보를 반자동 생성(사람 검수 전제)

큰 평가셋을 손으로 다 쓰긴 어렵다. 이 스크립트는 코퍼스의 각 논문에서 **그 논문만으로
답할 수 있는 질문**을 LLM으로 뽑아 후보(JSONL)로 저장한다. `expected_sources` 는 질문을
만든 바로 그 논문으로 자동 채워지므로, 사람은 "이 질문이 타당한가"만 검수하면 된다.

워크플로:
  1) python eval/gen_questions.py --n 20 --out eval/eval_candidates.jsonl   # 후보 생성(LLM)
  2) 사람이 eval/eval_candidates.jsonl 을 검수(이상한 질문 삭제, expected_sources 보강)
  3) 검수 통과분을 eval/eval_set.jsonl 로 옮긴다
  4) python eval/compare.py --retrieval-only  # 무료 검색지표로 대량 스크리닝

주의: 자동 생성 질문은 **단일 홉**(한 논문) 중심이다. 다중 홉(논문 비교) 질문은 품질이
민감해 사람이 직접 추가하는 것을 권장한다. 생성 품질도 검수 전까지 신뢰하지 말 것.

--dry-run 은 LLM 호출 없이 '어떤 논문이 쓰일지'만 출력한다(비용 0, 연결 점검용).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from rag.vectorstore import get_all_documents, load_vectorstore  # noqa: E402

# 오프라인 평가 도구 전용 프롬프트(파이프라인 프롬프트가 아니므로 여기 둔다)
GEN_PROMPT = """당신은 학술 RAG 시스템의 평가셋을 만드는 출제자입니다.
아래는 한 논문의 발췌입니다. 이 발췌 **하나만으로 답할 수 있는** 한국어 질문 1개를 만드세요.
- 논문 제목을 그대로 복창하지 말고, 핵심 기여/방법/결과를 묻는 구체적 질문으로.
- 너무 지엽적이지 않게, 이 논문을 아는 사람이라면 답할 수 있는 수준으로.

제목: {title}
발췌:
{excerpt}

JSON만 출력:
{{"question": "<질문>", "category": "<짧은 주제 슬러그>", "difficulty": "easy|medium|hard"}}"""

# 다중 홉: 서로 관련 있는 두 논문을 '비교·종합'해야 답하는 질문
MULTI_HOP_PROMPT = """당신은 학술 RAG 시스템의 평가셋을 만드는 출제자입니다.
아래는 **서로 다른 두 논문**의 발췌입니다. **두 논문을 모두 봐야** 답할 수 있는
비교·종합형 한국어 질문 1개를 만드세요(예: 방식 차이, 트레이드오프, 상호보완).
- 한쪽 논문만으로 답할 수 있으면 안 됩니다.
- 두 논문이 비교할 만큼 관련이 없으면 질문을 만들지 말고 skip 하세요.

[논문 A] {title_a}
{excerpt_a}

[논문 B] {title_b}
{excerpt_b}

JSON만 출력(둘 중 하나):
{{"question": "<질문>", "category": "<슬러그>", "difficulty": "hard"}}
또는 {{"skip": true}}"""


def _pick_chunk_per_source(docs) -> dict[str, dict]:
    """source별 대표 청크 1개 선택(초록/서론 우선, 없으면 첫 청크)."""
    prefer = ("abstract", "초록", "introduction", "서론", "body")
    best: dict[str, dict] = {}
    for d in docs:
        src = d.metadata.get("source")
        if not src:
            continue
        sec = (d.metadata.get("section") or "").lower()
        rank = next((i for i, p in enumerate(prefer) if p in sec), len(prefer))
        cur = best.get(src)
        if cur is None or rank < cur["rank"]:
            best[src] = {
                "rank": rank,
                "title": d.metadata.get("title", ""),
                "content": d.page_content,
            }
    return best


def _nearest_pairs(per_source: dict, sources: list[str]) -> list[tuple[str, str]]:
    """각 논문의 '대표 청크 임베딩' 최근접 다른 논문을 짝지어 (a,b) 쌍 반환(중복 쌍 제거)."""
    import numpy as np

    from rag.llm import get_embeddings

    embs = np.array(
        get_embeddings().embed_documents([per_source[s]["content"][:1500] for s in sources]),
        dtype=float,
    )
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    sim = embs @ embs.T
    np.fill_diagonal(sim, -1.0)  # 자기 자신 제외

    seen: set[frozenset] = set()
    pairs: list[tuple[str, str]] = []
    for i, s in enumerate(sources):
        j = int(sim[i].argmax())
        key = frozenset((s, sources[j]))
        if len(key) == 2 and key not in seen:
            seen.add(key)
            pairs.append((s, sources[j]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="질문을 생성할 논문(또는 쌍) 수")
    ap.add_argument("--out", default="eval/eval_candidates.jsonl", help="후보 출력 경로(JSONL)")
    ap.add_argument("--seed", type=int, default=0, help="논문 샘플링 시드(재현성)")
    ap.add_argument("--light", action="store_true", help="경량 모델로 생성(대량 생성 비용 절감)")
    ap.add_argument("--multi-hop", action="store_true",
                    help="두 논문을 비교하는 다중 홉 질문 생성(임베딩 최근접 쌍)")
    ap.add_argument("--dry-run", action="store_true", help="LLM 없이 대상만 출력")
    args = ap.parse_args()

    docs = get_all_documents(load_vectorstore())
    per_source = _pick_chunk_per_source(docs)
    sources = sorted(per_source)
    random.Random(args.seed).shuffle(sources)
    sources = sources[: args.n * (2 if args.multi_hop else 1)]

    if args.multi_hop:
        pairs = _nearest_pairs(per_source, sources)[: args.n]
        print(f"다중 홉: 최근접 논문 쌍 {len(pairs)}개 (전체 {len(per_source)}편 중)")
        if args.dry_run:
            for a, b in pairs:
                print(f"  [{a}] ↔ [{b}]  {per_source[a]['title'][:30]} / {per_source[b]['title'][:30]}")
            print("(dry-run: LLM 호출 없음)")
            return
    else:
        sources = sources[: args.n]
        print(f"대상 논문 {len(sources)}편 (전체 {len(per_source)}편 중)")
        if args.dry_run:
            for s in sources:
                print(f"  [{s}] {per_source[s]['title'][:60]}")
            print("(dry-run: LLM 호출 없음)")
            return

    from rag.llm import get_llm
    from rag.stage4_verify import _parse_json

    llm = get_llm(light=args.light)
    out_path = Path(args.out)
    n_ok = n_skip = 0

    with out_path.open("w", encoding="utf-8") as f:
        if args.multi_hop:
            chain = ChatPromptTemplate.from_template(MULTI_HOP_PROMPT) | llm | StrOutputParser()
            for a, b in pairs:
                try:
                    raw = chain.invoke({
                        "title_a": per_source[a]["title"], "excerpt_a": per_source[a]["content"][:1200],
                        "title_b": per_source[b]["title"], "excerpt_b": per_source[b]["content"][:1200],
                    })
                    data = _parse_json(raw)
                    if data.get("skip"):
                        n_skip += 1
                        print(f"  – [{a}↔{b}] skip(관련 낮음)")
                        continue
                    rec = {
                        "id": f"genmh_{a}_{b}", "question": data["question"],
                        "category": data.get("category", "auto"), "difficulty": data.get("difficulty", "hard"),
                        "multi_hop": True, "expected_sources": [a, b],
                        "review": "pending", "generated": True,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1
                    print(f"  ✓ [{a}↔{b}] {rec['question'][:55]}")
                except Exception as e:  # noqa: BLE001
                    print(f"  ✗ [{a}↔{b}] 실패: {e}")
        else:
            chain = ChatPromptTemplate.from_template(GEN_PROMPT) | llm | StrOutputParser()
            for s in sources:
                info = per_source[s]
                try:
                    raw = chain.invoke({"title": info["title"], "excerpt": info["content"][:1500]})
                    data = _parse_json(raw)
                    rec = {
                        "id": f"gen_{s}", "question": data["question"],
                        "category": data.get("category", "auto"), "difficulty": data.get("difficulty", "medium"),
                        "multi_hop": False, "expected_sources": [s],
                        "review": "pending", "generated": True,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1
                    print(f"  ✓ [{s}] {rec['question'][:60]}")
                except Exception as e:  # noqa: BLE001
                    print(f"  ✗ [{s}] 생성 실패: {e}")

    print(f"\n완료: {n_ok}개 후보"
          + (f" (skip {n_skip})" if args.multi_hop else "")
          + f" → {out_path}")
    print("※ 사람이 검수(review:pending 확인·수정) 후 eval/eval_set.jsonl 로 옮기세요.")


if __name__ == "__main__":
    main()
