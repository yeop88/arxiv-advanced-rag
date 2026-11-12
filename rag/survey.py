"""
rag/survey.py — 문헌 조사 초안(C3)

주제 하나를 받아 관련 논문들을 넓게 모으고, 소주제별 흐름을 인용과 함께 정리한 초안을 만든다.

1) 질의 변환(경량 LLM) → 검색·리랭크를 평소보다 넓게(근거 최대 SURVEY_CHUNKS개, 논문당 1개 → 서로 다른 논문)
2) 근거마다 인용 그래프의 연도·DB 안 선행 관계를 머리에 붙임(흐름 서술용)
3) 메인 LLM 1회로 개요·주제별 흐름·남은 과제 작성(문장마다 [n])
4) 참고문헌은 코드가 근거 목록에서 만든다(LLM 이 지어내지 않게) → 4단계 검증
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass

from config import settings
from rag.schemas import RetrievedChunk, VerificationResult

SURVEY_CHUNKS = 8
SURVEY_MIN_SCORE = 0.005
CHUNK_CHARS = 1200


@dataclass
class Survey:
    topic: str
    text: str
    chunks: list[RetrievedChunk]
    verification: VerificationResult | None = None

    @property
    def cited(self) -> set[int]:
        return {int(n) for n in re.findall(r"\[(\d+)\]", self.text)}

    def references_md(self) -> str:
        """근거 번호 → 논문(코드가 만든 참고문헌)."""
        from rag.citations import is_arxiv_id

        lines = []
        for i, c in enumerate(self.chunks, 1):
            link = f"https://arxiv.org/abs/{c.source}" if is_arxiv_id(c.source) else ""  # 로컬 논문은 링크 없음
            mark = "" if i in self.cited else " (본문 미인용)"
            lines.append(f"[{i}] {c.title or c.source}" + (f" — {link}" if link else "") + mark)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        return f"# 문헌 조사 초안: {self.topic}\n\n{self.text.strip()}\n\n## 참고문헌\n" + \
            "\n".join(f"- {l}" for l in self.references_md().splitlines())


@contextmanager
def _wide_retrieval():
    """조사용으로 검색 폭을 잠시 넓힌다(근거 수↑, 논문당 1개 → 여러 논문)."""
    keep = (settings.top_k_rerank, settings.rerank_max_per_source, settings.top_k_retrieve)
    settings.top_k_rerank, settings.rerank_max_per_source = SURVEY_CHUNKS, 1
    settings.top_k_retrieve = max(settings.top_k_retrieve, 40)
    try:
        yield
    finally:
        settings.top_k_rerank, settings.rerank_max_per_source, settings.top_k_retrieve = keep


def gather(topic: str, vectorstore) -> list[RetrievedChunk]:
    from rag import stage1_query_transform as s1
    from rag import stage2_rerank as s2

    try:
        queries = s1.transform_query(topic)
    except Exception:  # noqa: BLE001
        queries = [topic]
    with _wide_retrieval():
        _, reranked = s2.retrieve_and_rerank(vectorstore, topic, queries)
    # 폭을 넓히면 무관한 논문(리랭크 0점대)이 끼므로 뺀다. 측정: 무관 논문은 0.000, 관련 논문은 0.013 이상(§5.27)
    kept = [c for c in reranked if (c.rerank_score or 0) >= SURVEY_MIN_SCORE]
    return kept if len(kept) >= 2 else reranked[:2]


def year_of(source: str, graph: dict) -> int | None:
    """처음 공개된 연도. arXiv ID 의 앞 두 자리(2103.00020 → 2021, math/0211159 → 2002)를 우선 쓰고,
    없으면 인용 그래프 연도. OpenAlex 연도는 학술지 게재 연도라 흐름 서술에 늦게 나올 수 있다."""
    m = re.match(r"^(?:[a-z\-]+(?:\.[A-Z]{2})?/)?(\d{2})\d{2}", source or "")
    if m:
        yy = int(m.group(1))
        return (1900 if yy >= 91 else 2000) + yy
    return (graph["papers"].get(source) or {}).get("year")


def headers(chunks: list[RetrievedChunk], graph: dict | None = None) -> list[str]:
    """근거마다 '제목, 연도 · 인용: [k]'(이 근거 목록 안 번호로 DB 안 선행 관계)."""
    from rag import citations

    g = graph if graph is not None else citations.load()
    num = {c.source: i for i, c in enumerate(chunks, 1)}
    out = []
    for c in chunks:
        y = year_of(c.source, g)
        year = f", {y}" if y else ""
        prior = ([f"[{num[a]}]" for a in citations.related_in_corpus(c.source, g)["prior"] if a in num]
                 if c.source in g["papers"] else [])
        rel = f" · 인용: {', '.join(prior)}" if prior else ""
        out.append(f"{c.title or c.source}{year}{rel}")
    return out


def format_context(chunks: list[RetrievedChunk], graph: dict | None = None) -> str:
    """근거 머리에 제목·연도·DB 안 선행 관계를 붙인다."""
    return "\n\n".join(f"[{i}] {h}\n{c.content[:CHUNK_CHARS]}"
                       for i, (h, c) in enumerate(zip(headers(chunks, graph), chunks), 1))


def _write(topic: str, context: str) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from rag import prompts
    from rag.llm import get_llm

    chain = ChatPromptTemplate.from_template(prompts.SURVEY_PROMPT) | get_llm(light=False) | StrOutputParser()
    return chain.invoke({"topic": topic, "context": context}).strip()


def build_survey(topic: str, vectorstore, verify: bool = True, progress=None) -> Survey:
    progress = progress or (lambda *a, **k: None)
    progress("survey_gather")
    chunks = gather(topic, vectorstore)
    if not chunks:
        return Survey(topic, "관련 근거를 찾지 못했습니다.", [])
    progress("survey_write", n=len(chunks), papers=len({c.source for c in chunks}))
    text = _write(topic, format_context(chunks))
    survey = Survey(topic, text, chunks)
    if verify:
        from rag import stage4_verify as s4

        progress("survey_verify")
        # 작성 때 준 연도·인용 관계를 검증에도 같이 준다(제목 칸에 붙여 근거 머리에 나오게)
        shown = [c.model_copy(update={"title": h}) for h, c in zip(headers(chunks), chunks)]
        try:
            survey.verification = s4.verify(text, shown)
        except Exception:  # noqa: BLE001
            survey.verification = None
    return survey
