"""
rag/visual.py — 표·그림 이해(로컬 멀티모달 모델)

학술 PDF 의 표는 세로줄이 없어 규칙 기반 추출(pdfplumber)로는 쓸 수 없었다(§5.34).
대신 캡션("Table N", "Figure N")이 있는 페이지를 이미지로 바꿔 로컬 Qwen3.6(이미지 입력 가능)에 보내
표는 마크다운 표로, 그림은 캡션과 보이는 내용만 짧게 옮기게 하고, 결과를 근거 청크로 만든다.

검사(LLM 없음): 옮긴 표의 숫자가 그 페이지의 PDF 텍스트에 실제로 있는 비율 — 기준 미만이면 버린다(잘못 읽은 표).
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass

from config import settings

# 캡션은 줄 맨 앞("Table 2: …")에 온다 — 본문 속 언급("see Table 2.")은 빼서 처리할 페이지를 줄인다. "Figure1:" 처럼 붙은 경우도
CAPTION = re.compile(r"(?m)^\s*(Table|Figure|Fig\.)\s*(\d+)\s*[:.]", re.I)
MIN_NUMBER_MATCH = 0.8   # 표 숫자 중 페이지 텍스트에 있어야 하는 비율
RESOLUTION = 110         # 페이지 이미지 해상도(dpi). 110 에서 DPR Table 2 를 정확히 옮겼다
MAX_PAGES = 12           # 논문당 처리할 최대 페이지 수(시간 제한: 페이지당 약 30초)

PROMPT = """이 논문 페이지 이미지에 있는 표와 그림을 옮기세요. 이 페이지의 캡션: {captions}

형식(각 항목은 ### 로 시작):
### Table N: <캡션 원문>
<마크다운 표 — 보이는 행과 열, 숫자를 그대로. 계산하거나 추측하지 말 것>

### Figure N: <캡션 원문>
<그림에서 보이는 것만 한국어 2~3문장: 축·범례·눈에 보이는 값이나 경향. 보이지 않는 내용은 쓰지 말 것>

이 페이지에 실제로 있는 표·그림만 쓰고, 다른 설명은 출력하지 마세요."""


@dataclass
class VisualItem:
    kind: str        # table | figure
    label: str       # "Table 2"
    caption: str
    body: str
    page: int
    number_match: float | None = None  # 표만: 숫자가 페이지 텍스트에 있는 비율

    @property
    def text(self) -> str:
        return f"{self.label}: {self.caption}\n{self.body}".strip()


def caption_pages(pdf) -> list[tuple[int, list[str], str]]:
    """캡션이 있는 페이지 (번호, 캡션 라벨들, 페이지 텍스트)."""
    out = []
    for pg in pdf.pages:
        text = pg.extract_text() or ""
        labels = list(dict.fromkeys(f"{'Figure' if m.group(1).lower().startswith('fig') else 'Table'} {m.group(2)}"
                                    for m in CAPTION.finditer(text)))
        if not settings.visual_figures:  # 그림을 안 쓰면 표 캡션이 있는 페이지만(시간도 준다)
            labels = [lb for lb in labels if lb.startswith("Table")]
        if labels:
            out.append((pg.page_number, labels, text))
    return out


def _numbers(text: str) -> list[str]:
    return re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text or "")


def number_match(table_md: str, page_text: str) -> float:
    """표의 숫자 중 페이지 텍스트에도 있는 비율(숫자가 없으면 1.0)."""
    nums = [n for n in _numbers(table_md) if len(n) > 1 or "." in n]
    if not nums:
        return 1.0
    have = set(_numbers(page_text))
    return sum(n in have for n in nums) / len(nums)


def parse(raw: str, page: int) -> list[VisualItem]:
    items = []
    for block in re.split(r"(?m)^###\s+", raw or ""):
        head, _, body = block.strip().partition("\n")
        m = re.match(r"(Table|Figure)\s+(\d+)\s*[:.]?\s*(.*)", head.strip(), re.I)
        if not m or not body.strip():
            continue
        kind = "table" if m.group(1).lower() == "table" else "figure"
        body = re.sub(r"^```\w*\n|```$", "", body.strip(), flags=re.M).strip()
        items.append(VisualItem(kind, f"{m.group(1).title()} {m.group(2)}", m.group(3).strip(), body, page))
    return items


def _ask(png: bytes, captions: list[str]) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=settings.local_base_url, api_key=settings.local_api_key, timeout=600, max_retries=1)
    r = client.chat.completions.create(
        model=settings.llm_model, max_tokens=3000, temperature=0.2,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(captions=", ".join(captions))},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.choices[0].message.content or ""


def extract(pdf_bytes: bytes, progress=None) -> list[VisualItem]:
    """PDF → 표·그림 항목(숫자 대조를 통과한 표와 그림)."""
    import pdfplumber

    if settings.llm_provider != "local":
        raise RuntimeError("표·그림 이해는 로컬 멀티모달 모델(LLM_PROVIDER=local)에서만 씁니다.")
    progress = progress or (lambda *a, **k: None)
    items: list[VisualItem] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = caption_pages(pdf)[:MAX_PAGES]
        for i, (no, labels, text) in enumerate(pages, 1):
            progress("visual_page", i=i, n=len(pages), page=no, labels=labels)
            buf = io.BytesIO()
            pdf.pages[no - 1].to_image(resolution=RESOLUTION).original.save(buf, format="PNG")
            for it in parse(_ask(buf.getvalue(), labels), no):
                if it.label not in labels:   # 이 페이지 캡션에 없는 항목(지어낸 것)과 끈 그림은 버린다
                    continue
                if it.kind == "table":
                    it.number_match = round(number_match(it.body, text), 3)
                    if it.number_match < MIN_NUMBER_MATCH:
                        continue
                items.append(it)
    return items
