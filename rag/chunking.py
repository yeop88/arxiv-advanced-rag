"""
rag/chunking.py — 섹션 인식(section-aware) 청킹

왜 필요한가?
  논문 전문을 그냥 1000자씩 자르면, 하나의 청크가 Method 끝과 Results 시작에
  걸치는 등 '의미 경계'를 무시합니다. 학술 논문은 섹션 구조가 뚜렷하므로,
  먼저 섹션으로 나눈 뒤 섹션 안에서만 청킹하면:
    - 각 청크가 하나의 섹션에 속해 의미가 깔끔해지고
    - 청크 메타데이터에 section 이 붙어 인용/필터링이 정확해집니다.

초록만 수집하는 경우엔 섹션이 없으므로 그대로 통과합니다(하위 호환).
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# 흔한 논문 섹션 제목 (영문). 대소문자 무시.
_COMMON_SECTIONS_EN = (
    r"abstract|introduction|related work|background|preliminaries|"
    r"methodology|methods?|approach|model|architecture|"
    r"experiments?|experimental setup|setup|datasets?|"
    r"results?|evaluation|analysis|ablation|discussion|"
    r"limitations?|conclusions?|future work|"
    r"references|acknowledge?ments?|appendix"
)
# 흔한 논문 섹션 제목 (국문).
_COMMON_SECTIONS_KO = (
    r"초\s*록|요\s*약|서\s*론|소\s*개|관련\s*연구|배\s*경|이론적\s*배경|"
    r"연구\s*방법|방\s*법|방법론|제안\s*방법|모\s*델|실\s*험|실험\s*설계|"
    r"데이터|결\s*과|실험\s*결과|분\s*석|평\s*가|고\s*찰|논\s*의|토\s*의|"
    r"결\s*론|한\s*계|향후\s*연구|참고\s*문헌|감사의\s*글|부\s*록"
)
_COMMON_SECTIONS = f"{_COMMON_SECTIONS_EN}|{_COMMON_SECTIONS_KO}"

# (a) 번호형 제목:  "3 Method", "3.1. Experimental Setup", "2 방법", "Ⅲ. 결과"
#     번호 뒤 제목의 첫 글자로 라틴 대문자 또는 한글 허용.
_NUMBERED = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z가-힣][\w\-\s가-힣]{1,60})\s*$"
)
# (b) 이름형 제목:  "Introduction", "RELATED WORK", "서론", "결론" (짧은 줄 단독)
_NAMED = re.compile(rf"^\s*({_COMMON_SECTIONS})\s*$", re.IGNORECASE)


def _detect_section(line: str) -> str | None:
    """한 줄이 섹션 제목처럼 보이면 정규화된 제목을, 아니면 None 을 반환."""
    line_stripped = line.strip()
    if not line_stripped or len(line_stripped) > 70:
        return None
    m = _NUMBERED.match(line_stripped)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    m = _NAMED.match(line_stripped)
    if m:
        return m.group(1).strip().title()
    return None


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """전문 텍스트를 [(섹션명, 섹션본문), ...] 로 분리.

    섹션 제목을 하나도 못 찾으면 [("body", 전체)] 를 반환(하위 호환).
    """
    sections: list[tuple[str, list[str]]] = []
    current = "header"  # 첫 섹션 제목 이전(제목·저자·초록 전) 영역
    buffer: list[str] = []
    found_any = False  # 실제 섹션 제목을 하나라도 찾았는지

    for line in text.splitlines():
        sec = _detect_section(line)
        if sec is not None:
            found_any = True
            # 이전 섹션 저장 후 새 섹션 시작
            sections.append((current, buffer))
            current = sec
            buffer = []
        else:
            buffer.append(line)
    sections.append((current, buffer))

    # 섹션 제목을 하나도 못 찾았으면(초록만 등) 전체를 'body' 로 폴백
    if not found_any:
        body = text.strip()
        return [("body", body)] if body else []

    # 본문 없는 섹션 제거 + 문자열화
    result = [(name, "\n".join(lines).strip()) for name, lines in sections]
    result = [(n, b) for n, b in result if b]
    return result or [("body", text.strip())]


def section_aware_split(text: str, base_metadata: dict) -> list[Document]:
    """전문 텍스트를 섹션으로 나눈 뒤, 섹션 안에서 청킹.

    각 청크 metadata 에 base_metadata + {"section": 섹션명} 을 부여.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    docs: list[Document] = []
    for section_name, section_text in split_into_sections(text):
        for chunk in splitter.split_text(section_text):
            meta = {**base_metadata, "section": section_name}
            docs.append(Document(page_content=chunk, metadata=meta))
    return docs
