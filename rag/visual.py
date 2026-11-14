"""
rag/visual.py — 표·그림·수식 이해(로컬 멀티모달 모델)

학술 PDF 의 표는 세로줄이 없어 규칙 기반 추출(pdfplumber)로는 쓸 수 없었고(§5.34), 수식은 PDF 텍스트에서
기호가 깨진다("C(r)= T(t)σ(r(t))c(r(t),d)dt" — 적분 기호가 사라짐). 그래서 표 캡션("Table N")·그림 캡션("Figure N")·
번호 붙은 수식("… (N)")이 있는 페이지를 이미지로 바꿔 로컬 Qwen3.6(이미지 입력 가능)에 보내
표는 마크다운 표로, 그림은 보이는 내용만 영어 2~3문장으로, 수식은 LaTeX 와 기호 뜻으로 옮기게 하고 근거 청크로 만든다.
그림·수식 설명은 영어로 쓴다 — 코퍼스가 영어라 한국어 설명은 무관한 한국어 질문과 두루 비슷하게 매겨졌다(§5.35).

검사(LLM 없음): 페이지에 없는 라벨은 버린다. 표는 숫자가 그 페이지 PDF 텍스트에 있는 비율이 기준 미만이면 버리고,
수식은 관계 기호(=, ≤ 등)가 없거나 그리스 문자가 페이지 텍스트와 맞지 않으면 버린다.
"""
from __future__ import annotations

import base64
import io
import re
import unicodedata
from dataclasses import dataclass

from config import settings

KINDS = ("table", "figure", "equation")
# 캡션은 줄 맨 앞("Table 2: …")에 온다 — 본문 속 언급("see Table 2.")은 빼서 처리할 페이지를 줄인다. "Figure1:" 처럼 붙은 경우도
CAPTION = re.compile(r"(?m)^\s*(Table|Figure|Fig\.)\s*(\d+)\s*[:.]", re.I)
# 번호 붙은 수식: 줄 끝의 "(N)". 수식 기호가 있는 줄만(본문 문장 끝의 "(3)" 은 빼려고)
EQ_LINE = re.compile(r"(?m)^(.*\S)\s*\((\d{1,3})\)\s*$")
_MATHY = re.compile(r"[=≤≥≈∝∼∑∏∫√σαβγδθλμπφψωΔΣΩ∈→]|[\U0001D400-\U0001D7FF]")
MIN_NUMBER_MATCH = 0.8   # 표 숫자 중 페이지 텍스트에 있어야 하는 비율
MIN_GREEK_MATCH = 0.5    # 수식의 그리스 문자 중 페이지 텍스트에 있어야 하는 비율
RESOLUTION = 110         # 페이지 이미지 해상도(dpi). 110 에서 DPR Table 2 를 정확히 옮겼다
MAX_PAGES = 12           # 논문당 표·그림 페이지 최대 수(시간 제한: 페이지당 약 30초)
MAX_EQ_PAGES = 6         # 논문당 수식 페이지 최대 수(본문 앞쪽부터 — 부록의 유도 과정까지 다 옮기지 않게)

_INSTRUCTIONS = {
    "table": "### Table N: <caption text>\n<markdown table — rows, columns and numbers exactly as shown; do not compute or guess>",
    "figure": "### Figure N: <caption text>\n<2-3 English sentences describing only what is visible: axes, legend, visible values or trends>",
    "equation": "### Equation N: <short English name of what it computes>\n$$<LaTeX of equation (N) exactly as shown>$$\n"
                "<1-2 English sentences on what the symbols denote, only as defined on this page>",
}
PROMPT = """Transcribe the following items from this paper page image: {labels}

Format (each item starts with ###):
{formats}

Only output items that actually appear on this page. No other text."""


@dataclass
class VisualItem:
    kind: str        # table | figure | equation
    label: str       # "Table 2", "Equation 3"
    caption: str
    body: str
    page: int
    number_match: float | None = None  # 표: 숫자가 페이지 텍스트에 있는 비율 / 수식: 그리스 문자 일치 비율

    @property
    def text(self) -> str:
        return f"{self.label}: {self.caption}\n{self.body}".strip()


def _kind_of(label: str) -> str:
    return label.split()[0].lower()


def default_kinds() -> tuple[str, ...]:
    return tuple(k for k, on in (("table", True), ("figure", settings.visual_figures),
                                 ("equation", settings.visual_equations)) if on)


def equation_labels(text: str) -> list[str]:
    return list(dict.fromkeys(f"Equation {m.group(2)}" for m in EQ_LINE.finditer(text or "")
                              if _MATHY.search(m.group(1))))


def page_labels(pdf, kinds=("table",)) -> list[tuple[int, list[str], str]]:
    """처리할 페이지 (번호, 라벨들, 페이지 텍스트). 표·그림은 캡션, 수식은 번호 붙은 수식 줄로 찾는다."""
    out, eq_pages = [], 0
    for pg in pdf.pages:
        text = pg.extract_text() or ""
        labels = [f"{'Figure' if m.group(1).lower().startswith('fig') else 'Table'} {m.group(2)}"
                  for m in CAPTION.finditer(text)]
        eqs = equation_labels(text) if "equation" in kinds and eq_pages < MAX_EQ_PAGES else []
        eq_pages += bool(eqs)
        labels = [lb for lb in dict.fromkeys(labels + eqs) if _kind_of(lb) in kinds]
        if labels:
            out.append((pg.page_number, labels, text))
    visual = [p for p in out if any(_kind_of(lb) != "equation" for lb in p[1])][:MAX_PAGES]
    eq_only = [p for p in out if all(_kind_of(lb) == "equation" for lb in p[1])]
    return sorted(visual + eq_only)


def _numbers(text: str) -> list[str]:
    return re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text or "")


def number_match(table_md: str, page_text: str) -> float:
    """표의 숫자 중 페이지 텍스트에도 있는 비율(숫자가 없으면 1.0)."""
    nums = [n for n in _numbers(table_md) if len(n) > 1 or "." in n]
    if not nums:
        return 1.0
    have = set(_numbers(page_text))
    return sum(n in have for n in nums) / len(nums)


_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "theta": "θ",
          "lambda": "λ", "mu": "μ", "pi": "π", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "psi": "ψ",
          "omega": "ω", "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "eta": "η", "rho": "ρ", "xi": "ξ", "nu": "ν", "kappa": "κ"}


def greek_match(latex: str, page_text: str) -> float:
    """수식 LaTeX 의 그리스 문자(\\sigma 등) 중 페이지 텍스트에도 있는 비율(없으면 1.0). 수학 이탤릭(𝜎)은 NFKC 로 맞춘다."""
    letters = {_GREEK[m] for m in re.findall(r"\\([A-Za-z]+)", latex or "") if m in _GREEK}
    if not letters:
        return 1.0
    page = unicodedata.normalize("NFKC", page_text or "")
    return sum(g in page for g in letters) / len(letters)


def check_equation(body: str, page_text: str) -> float | None:
    """수식 검사: 관계 기호가 있고 그리스 문자가 페이지와 맞으면 일치 비율, 아니면 None(버림)."""
    m = re.search(r"\$\$(.+?)\$\$", body or "", re.S)
    if not m or not re.search(r"=|\\le|\\ge|\\approx|\\propto|\\sim|\\in|\\to|<|>", m.group(1)):
        return None
    g = greek_match(m.group(1), page_text)
    return g if g >= MIN_GREEK_MATCH else None


def parse(raw: str, page: int) -> list[VisualItem]:
    items = []
    for block in re.split(r"(?m)^###\s+", raw or ""):
        head, _, body = block.strip().partition("\n")
        m = re.match(r"(Table|Figure|Equation)\s+\(?(\d+)\)?\s*[:.]?\s*(.*)", head.strip(), re.I)
        if not m or not body.strip():
            continue
        kind = m.group(1).lower()
        body = re.sub(r"^```\w*\n|```$", "", body.strip(), flags=re.M).strip()
        items.append(VisualItem(kind, f"{m.group(1).title()} {m.group(2)}", m.group(3).strip(), body, page))
    return items


def _ask(png: bytes, labels: list[str]) -> str:
    from openai import OpenAI

    kinds = [k for k in KINDS if any(_kind_of(lb) == k for lb in labels)]
    prompt = PROMPT.format(labels=", ".join(labels), formats="\n\n".join(_INSTRUCTIONS[k] for k in kinds))
    client = OpenAI(base_url=settings.local_base_url, api_key=settings.local_api_key, timeout=600, max_retries=1)
    r = client.chat.completions.create(
        model=settings.llm_model, max_tokens=3000, temperature=0.2,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.choices[0].message.content or ""


def extract(pdf_bytes: bytes, progress=None, kinds: tuple[str, ...] | None = None) -> list[VisualItem]:
    """PDF → 표·그림·수식 항목(검사를 통과한 것)."""
    import pdfplumber

    if settings.llm_provider != "local":
        raise RuntimeError("표·그림·수식 이해는 로컬 멀티모달 모델(LLM_PROVIDER=local)에서만 씁니다.")
    kinds = kinds or default_kinds()
    progress = progress or (lambda *a, **k: None)
    items: list[VisualItem] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = page_labels(pdf, kinds)
        for i, (no, labels, text) in enumerate(pages, 1):
            progress("visual_page", i=i, n=len(pages), page=no, labels=labels)
            buf = io.BytesIO()
            pdf.pages[no - 1].to_image(resolution=RESOLUTION).original.save(buf, format="PNG")
            for it in parse(_ask(buf.getvalue(), labels), no):
                if it.label not in labels:   # 이 페이지에 없는 항목(지어낸 것)과 요청하지 않은 종류는 버린다
                    continue
                if it.kind == "table":
                    it.number_match = round(number_match(it.body, text), 3)
                    if it.number_match < MIN_NUMBER_MATCH:
                        continue
                elif it.kind == "equation":
                    g = check_equation(it.body, text)
                    if g is None:
                        continue
                    it.number_match = round(g, 3)
                items.append(it)
    return items
