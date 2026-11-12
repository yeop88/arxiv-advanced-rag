"""
rag/compare_table.py — 논문 비교표(C1)

근거 논문 2~5편을 방법·데이터·결과·한계로 비교하는 표를 만든다. 칸마다 근거 번호 [n]을 단다.

1) 논문별 청크를 DB 에서 모으고, 항목별 영어 질의로 로컬 리랭커 채점 → 항목마다 가장 관련 있는 청크
2) 논문마다 LLM 1회: 항목별 한두 문장 + [n] (근거에 없으면 "근거 없음")
3) 형식 점검: 칸마다 그 논문의 근거 번호를 인용했는가 → 채운 칸을 주장 목록으로 바꿔 4단계 검증(stage4_verify)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.schemas import RetrievedChunk, VerificationResult

ASPECTS = {
    "방법": "proposed method, model architecture and key idea",
    "데이터": "datasets, benchmarks and experimental setup",
    "결과": "main results, performance numbers compared with baselines",
    "한계": "limitations, failure cases and future work",
}
NO_EVIDENCE = "근거 없음"
MAX_PAPERS = 5
CHUNK_CHARS = 1200


@dataclass
class CompareRow:
    source: str
    title: str
    cells: dict[str, str]
    uncited: list[str] = field(default_factory=list)  # 근거 번호 없이 쓴 칸(형식 위반)
    flagged: list[str] = field(default_factory=list)  # 검증에서 근거와 맞지 않는다고 걸린 칸


@dataclass
class CompareTable:
    rows: list[CompareRow]
    chunks: list[RetrievedChunk]          # 전역 번호 [1]..[N] 순서
    verification: VerificationResult | None = None

    @property
    def cells_total(self) -> int:
        return sum(len(r.cells) for r in self.rows)

    @property
    def cells_filled(self) -> int:
        return sum(v.strip() != NO_EVIDENCE for r in self.rows for v in r.cells.values())

    @property
    def cells_uncited(self) -> int:
        return sum(len(r.uncited) for r in self.rows)

    def claims_text(self) -> str:
        """검증용: 채운 칸만 '논문 · 항목: 내용' 목록으로("근거 없음" 칸은 주장이 아니라 빼고 검증)."""
        return "\n".join(f"- {r.title[:60]} · {a}: {t}" for r in self.rows for a, t in r.cells.items()
                         if t.strip() != NO_EVIDENCE)

    def to_markdown(self, marks: bool = False) -> str:
        """표. marks=True 면 검증에서 걸렸거나 근거 번호가 없는 칸 앞에 ⚠."""
        esc = lambda t: (t or "").replace("|", "\\|").replace("\n", " ")  # noqa: E731

        def cell(r: CompareRow, a: str) -> str:
            t = esc(r.cells.get(a, NO_EVIDENCE))
            return f"⚠ {t}" if marks and (a in r.flagged or a in r.uncited) else t

        head = "| 논문 | " + " | ".join(ASPECTS) + " |\n|" + "---|" * (len(ASPECTS) + 1)
        body = [f"| {esc(r.title[:60])} | " + " | ".join(cell(r, a) for a in ASPECTS) + " |" for r in self.rows]
        return "\n".join([head, *body])


def _paper_chunks(vectorstore, sources: list[str]) -> dict[str, list[RetrievedChunk]]:
    from rag.vectorstore import get_all_documents

    want = set(sources)
    out: dict[str, list[RetrievedChunk]] = {s: [] for s in sources}
    for d in get_all_documents(vectorstore):
        m = d.metadata or {}
        if m.get("source") in want:
            out[m["source"]].append(RetrievedChunk(
                content=d.page_content, id=d.id, source=m.get("source", ""), title=m.get("title", ""),
                section=m.get("section", ""), url=m.get("url", ""), origin=m.get("origin", "")))
    return out


def pick_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """한 논문의 청크에서 항목별로 가장 관련 있는 청크를 하나씩(중복 제거). 초록·짧은 논문은 그대로."""
    if len(chunks) <= len(ASPECTS):
        return chunks
    from rag.llm import get_reranker

    pairs = [(q, c.content[:CHUNK_CHARS]) for q in ASPECTS.values() for c in chunks]
    scores = get_reranker().predict(pairs)
    n = len(chunks)
    picked: list[RetrievedChunk] = []
    for i in range(len(ASPECTS)):
        row = scores[i * n:(i + 1) * n]
        for j in sorted(range(n), key=lambda k: -row[k]):
            if chunks[j] not in picked:
                picked.append(chunks[j])
                break
    return picked


def _format(chunks: list[RetrievedChunk], start: int) -> str:
    return "\n\n".join(f"[{start + i}] ({c.section or '본문'})\n{c.content[:CHUNK_CHARS]}" for i, c in enumerate(chunks))


def check_citations(cells: dict[str, str], allowed: range) -> list[str]:
    """근거 번호가 없거나 그 논문 범위 밖 번호만 단 칸 목록("근거 없음" 칸은 통과)."""
    bad = []
    for aspect, text in cells.items():
        if text.strip() == NO_EVIDENCE:
            continue
        nums = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        if not nums or not nums & set(allowed):
            bad.append(aspect)
    return bad


def _words(t: str) -> set[str]:
    return set(re.findall(r"[0-9A-Za-z가-힣.]{2,}", t or ""))


def flag_cells(rows: list[CompareRow], unsupported: list[str]) -> None:
    """검증이 지적한 주장을 가장 많이 겹치는 칸에 표시한다(겹침 50% 이상일 때만)."""
    for claim in unsupported or []:
        cw = _words(claim)
        if not cw:
            continue
        best, score = None, 0.0
        for r in rows:
            for a, t in r.cells.items():
                if t.strip() == NO_EVIDENCE:
                    continue
                s = len(cw & _words(t)) / len(cw)
                if s > score:
                    best, score = (r, a), s
        if best and score >= 0.5 and best[1] not in best[0].flagged:
            best[0].flagged.append(best[1])


def _summarize(title: str, context: str) -> dict[str, str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from rag import prompts
    from rag.llm import get_llm
    from rag.stage4_verify import _parse_json

    chain = ChatPromptTemplate.from_template(prompts.COMPARE_TABLE_PROMPT) | get_llm(light=False) | StrOutputParser()
    raw = chain.invoke({"title": title, "context": context})
    try:
        data = _parse_json(raw)
    except Exception:  # noqa: BLE001
        data = {}
    return {a: str(data.get(a) or NO_EVIDENCE).strip() for a in ASPECTS}


def build_table(sources: list[str], vectorstore, verify: bool = True, progress=None) -> CompareTable:
    """근거 논문 목록(arXiv ID, 중복 무시) → 비교표. 최대 MAX_PAPERS 편."""
    progress = progress or (lambda *a, **k: None)
    sources = list(dict.fromkeys(s for s in sources if s))[:MAX_PAPERS]
    by_paper = _paper_chunks(vectorstore, sources)
    rows: list[CompareRow] = []
    all_chunks: list[RetrievedChunk] = []
    for i, src in enumerate(sources, 1):
        chunks = pick_context(by_paper.get(src) or [])
        if not chunks:
            continue
        title = chunks[0].title or src
        progress("table_paper", i=i, n=len(sources), title=title)
        start = len(all_chunks) + 1
        cells = _summarize(title, _format(chunks, start))
        rows.append(CompareRow(src, title, cells, check_citations(cells, range(start, start + len(chunks)))))
        all_chunks += chunks
    table = CompareTable(rows, all_chunks)
    if verify and rows:
        from rag import stage4_verify as s4

        progress("table_verify")
        try:
            claims = table.claims_text()
            table.verification = s4.verify(claims, all_chunks) if claims else None
            if table.verification:
                flag_cells(rows, table.verification.unsupported_claims)
        except Exception:  # noqa: BLE001 — 검증 실패는 표를 막지 않는다
            table.verification = None
    return table
