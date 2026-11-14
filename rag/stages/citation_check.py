"""
rag/stages/citation_check.py — 4b단계: 문장 단위 인용 확인 (LLM 없음)

4단계 검증은 LLM 이 답변 전체를 한 번에 채점한다. 여기서는 답변을 문장으로 나눠
"이 문장이 인용한 [n] 근거가 실제로 이 문장을 뒷받침하는가"를 문장마다 따로 본다.
점수는 이미 올라와 있는 리랭커(bge-reranker-v2-m3, 다국어 크로스인코더)로 (문장, 근거 청크) 쌍마다 매긴다.
리랭커는 '관련도' 모델이라 함의(entailment)를 직접 판정하지는 못한다 — 그래서 숫자는 따로 규칙으로 대조한다.

문장 상태:
  supported  인용한 근거 중 하나의 점수가 기준 이상
  miscited   인용한 근거는 기준 미만인데 다른 근거가 기준 이상 → 번호를 잘못 단 것으로 보임(맞는 번호 제안)
  weak       인용했지만 어느 근거도 기준 미만 → 근거가 약함
  uncited    인용 번호가 없는 주장 문장
  bad_ref    없는 근거 번호를 인용
  contradicted 리랭커는 뒷받침으로 봤지만 함의(NLI) 판정이 인용한 근거 모두와 모순으로 봄(같은 주제, 다른 말) §5.44
  table_row  표의 데이터 행(문장이 아니라 점수는 매기지 않고 숫자만 대조, 주장 문장 수에는 넣지 않음)
숫자 대조(상태와 별도): 문장의 숫자(소수·두 자리 이상, 연도 제외)가 인용한 근거 원문에 없으면 numbers_missing 에 담는다.
리랭커는 '78.4' 와 '68.4' 를 거의 같은 점수로 보므로, 표 숫자를 잘못 옮기는 환각은 이 규칙이 잡는다.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

from config import settings
from rag.core.schemas import RetrievedChunk

_CITE = re.compile(r"\[(?:근거\s*)?(\d+(?:\s*[,，\-–]\s*\d+)*)\]")
_SENT_END = re.compile(r"(?<=[.!?。])\s+")
_MD_PREFIX = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)+")
_MD_MARK = re.compile(r"\*\*|__|`")
MIN_CLAIM_CHARS = 12   # 이보다 짧은 문장(제목·연결 문구)은 확인하지 않음
# 주장이 아닌 문장: '근거에 없다'는 정직한 답(81개 답변 중 미인용 문장의 대부분)과 도입 문구
_NOT_CLAIM = re.compile(
    r"찾을 수 없|포함되어 있지 않|포함돼 있지 않|언급(?:이|은|되어 있지|하지|되지) ?(?:않|없)|언급이 없|설명(?:할 수 없|하지 않|이 없)"
    r"|제공하지 않|(?:답변|답|비교 분석을 수행|설명)할 수 없|나와 있지 않|다루(?:고 있지|지) 않|확인할 수 없|알 수 없|정보가 없|무관합니다"
    r"|다음과 같(?:습니다|이 [가-힣]+(?:니다|있습니다))\s*$|정리하면 다음|요약하면 다음")

# 답변 쪽 숫자: 영문·숫자·하이픈에 붙은 것(GPT-4, Top-20, bge-m3, v1.5)과 수식 안은 제외. 한국어 단위(2.7일, 3.5배)는 허용
_ANSWER_NUM = re.compile(r"(?<![A-Za-z0-9_.\-/^])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![A-Za-z0-9_^]|\.\d)")
# 근거 쪽 숫자: 넓게(표·본문 어디든)
_CHUNK_NUM = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
_INLINE_MATH = re.compile(r"\$[^$]+\$")


# 큰 수 단위: 답변은 한국어(100만), 근거는 영어(1 million, 1M)로 쓰는 일이 많다 → 실제 값으로도 비교
_KO_UNIT = re.compile(r"\s?(천만|천억|만|억|조|천)")
_KO_SCALE = {"천": 1e3, "만": 1e4, "천만": 1e7, "억": 1e8, "천억": 1e11, "조": 1e12}
_EN_UNIT = re.compile(r"\s?(thousand|million|billion|trillion|[KMB])\b")
_EN_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "trillion": 1e12}
# 근거가 숫자를 영어 낱말로 쓴 경우("an order-of-magnitude larger" → 답변 "10배"). 실제 답변에서 오경보로 발견(§5.42)
_EN_WORD_NUMS = [(re.compile(r"\btwo orders?[\s-]of[\s-]magnitude", re.I), "100"),
                 (re.compile(r"\b(?:an|one) orders?[\s-]of[\s-]magnitude|\border[\s-]of[\s-]magnitude", re.I), "10"),
                 (re.compile(r"\bten\b|\btenfold\b", re.I), "10"), (re.compile(r"\btwenty\b", re.I), "20"),
                 (re.compile(r"\bhundred\b|\bhundredfold\b", re.I), "100"), (re.compile(r"\bthousand\b", re.I), "1000")]


def _norm(int_part: str, frac: str | None) -> str:
    i = int_part.replace(",", "").lstrip("0") or "0"
    f = (frac or "").rstrip("0").rstrip(".")
    return i + (f if f and f != "." else "")


def _num_str(v: float) -> str:
    ip, fp = f"{v:.6f}".split(".")
    return _norm(ip, "." + fp)


def answer_numbers(text: str, ignore: set[str] | None = None) -> list[str]:
    """대조할 숫자들(정규화). 한 자리 정수(개수·순서)와 연도(1900~2099)는 근거 없이도 흔해 뺀다.
    '100만'·'1.2억' 처럼 단위가 붙으면 실제 값(1000000, 120000000)으로 바꾼다."""
    out: list[str] = []
    t = _INLINE_MATH.sub(" ", text)
    for m in _ANSWER_NUM.finditer(t):
        n = _norm(m.group(1), m.group(2))
        unit = _KO_UNIT.match(t, m.end())
        if unit:
            n = _num_str(float(n) * _KO_SCALE[unit.group(1)])
        elif "." not in n and (len(n) < 2 or (len(n) == 4 and 1900 <= int(n) <= 2099)):
            continue
        if n not in out and n not in (ignore or set()):
            out.append(n)
    return out


def chunk_numbers(text: str) -> set[str]:
    """근거에 있는 숫자들. 0.784 ↔ 78.4(%) 백분율 표기, '1 million'·'1M' 단위 표기도 같은 값으로 본다."""
    out: set[str] = set()
    text = text or ""
    for m in _CHUNK_NUM.finditer(text):
        n = _norm(m.group(1), m.group(2))
        out.add(n)
        v = float(n)
        unit = _EN_UNIT.match(text, m.end())
        if unit:
            out.add(_num_str(v * _EN_SCALE[unit.group(1).lower()]))
        if m.group(2):
            out.update((_num_str(v * 100), _num_str(v / 100)))
    for pat, n in _EN_WORD_NUMS:
        if pat.search(text):
            out.add(n)
    return out


# (문장, 근거) 쌍 목록 → 점수 목록. 기본은 리랭커. 테스트에서는 가짜를 넣는다.
Scorer = Callable[[list[tuple[str, str]]], Sequence[float]]


@dataclass
class SentenceCheck:
    text: str                       # 화면에 보이는 문장(마크다운 기호 제거)
    cites: list[int]                # 인용 번호(1부터)
    status: str
    support: float | None = None    # 인용한 근거 중 최고 점수
    best: int | None = None         # 모든 근거 중 가장 잘 맞는 번호
    best_score: float | None = None
    numbers_missing: list[str] = field(default_factory=list)   # 인용한 근거에 없는 숫자
    numbers_found_in: list[int] = field(default_factory=list)  # 그 숫자가 모두 있는 다른 근거 번호
    entail: float | None = None        # 인용한 근거 중 최고 함의 확률(NLI)
    contradiction: float | None = None  # 인용한 근거 중 최저 모순 확률(NLI) — 모두와 모순이어야 높다

    @property
    def flagged(self) -> bool:
        return self.status not in ("supported", "table_row") or bool(self.numbers_missing)


@dataclass
class CitationReport:
    sentences: list[SentenceCheck] = field(default_factory=list)
    threshold: float = 0.0
    numeric: int = 0                # 숫자를 대조한 문장·표 행 수
    table_rows: list[SentenceCheck] = field(default_factory=list)   # 답변 속 표의 데이터 행(숫자만 대조)

    def summary(self) -> dict:
        n = len(self.sentences)
        cited = [s for s in self.sentences if s.cites]
        ok = sum(s.status == "supported" for s in self.sentences)
        count = lambda st: sum(s.status == st for s in self.sentences)  # noqa: E731
        return {
            "claims": n,                    # 확인한 주장 문장 수
            "cited": len(cited),            # 인용 번호가 있는 문장
            "supported": ok,
            "miscited": count("miscited"),
            "weak": count("weak"),
            "uncited": count("uncited"),
            "bad_ref": count("bad_ref"),
            "contradicted": count("contradicted"),
            # 인용 정확도: 인용 단 문장 중 인용한 근거가 뒷받침하는 비율
            "precision": round(ok / len(cited), 3) if cited else None,
            # 인용 재현율: 주장 문장 중 뒷받침되는 인용이 달린 비율
            "recall": round(ok / n, 3) if n else None,
            "numeric": self.numeric,        # 숫자를 대조한 문장·표 행 수
            "number_mismatch": sum(bool(s.numbers_missing) for s in self.sentences + self.table_rows),
            "table_rows": len(self.table_rows),
        }

    def to_json(self) -> dict:
        return {"threshold": self.threshold, "summary": self.summary(),
                "flagged": [asdict(s) for s in self.sentences + self.table_rows if s.flagged]}


def parse_cites(text: str) -> list[int]:
    """'[1]', '[1, 3]', '[2-4]' → [1], [1, 3], [2, 3, 4] (순서 유지, 중복 제거)."""
    out: list[int] = []
    for m in _CITE.finditer(text):
        for part in re.split(r"\s*[,，]\s*", m.group(1)):
            if re.search(r"[\-–]", part):
                a, b = (int(x) for x in re.split(r"\s*[\-–]\s*", part)[:2])
                nums = range(a, b + 1) if 0 < b - a < 20 else [a, b]
            else:
                nums = [int(part)]
            for n in nums:
                if n not in out:
                    out.append(n)
    return out


def split_sentences(answer: str) -> list[str]:
    """답변 → 확인할 문장들. 표·코드·수식 블록과 짧은 문구는 뺀다."""
    out: list[str] = []
    fence = False
    for line in (answer or "").splitlines():
        s = line.strip()
        if s.startswith("```") or s == "$$":   # 코드·수식 블록 시작/끝
            fence = not fence
            continue
        if fence or not s or s.startswith("|") or (s.startswith("$$") and s.endswith("$$")):
            continue
        s = _MD_MARK.sub("", _MD_PREFIX.sub("", s))
        for part in _SENT_END.split(s):
            part = part.strip()
            if part:
                out.append(part)
    # 문장 끝 마침표 뒤에 단 인용('...입니다. [1]')은 앞 문장에 붙인다
    merged: list[str] = []
    for s in out:
        if merged and _CITE.fullmatch(s.rstrip(". ")):
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged


_TABLE_SEP = re.compile(r"^\|?[\s:|\-]+\|?$")


def table_rows(answer: str) -> list[str]:
    """답변 속 마크다운 표의 데이터 행('셀 | 셀' 글). 머리글 행·구분선·코드 블록 안은 뺀다."""
    out: list[str] = []
    fence, in_table = False, False
    for line in (answer or "").splitlines():
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if fence or not s.startswith("|"):
            in_table = False
            continue
        if _TABLE_SEP.match(s):
            continue
        if not in_table:          # 표의 첫 행 = 머리글
            in_table = True
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        out.append(" | ".join(c for c in cells if c))
    return out


def claim_text(sentence: str) -> str:
    """인용 번호를 뺀 주장 본문."""
    t = _CITE.sub("", sentence)
    t = re.sub(r"\(\s*(?:근거)?[\s,，~]*\)", "", t)      # '([1], [2])' 를 지우고 남은 '(, )'
    t = re.sub(r"\s+([,.])", lambda m: m.group(1), t)   # 번호를 지운 자리의 ' ,' → ','
    return re.sub(r"\s+", " ", t).strip(" .,")


def _is_claim(sentence: str) -> bool:
    c = claim_text(sentence)
    return len(c) >= MIN_CLAIM_CHARS and not c.endswith(":") and not _NOT_CLAIM.search(c)


_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
_nli = None


def get_nli():
    """다국어 함의(NLI) 판정 모델(한국어 문장 ↔ 영어 근거). 처음 쓸 때 한 번만 올린다(약 280M, GPU fp16)."""
    global _nli
    if _nli is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = settings.resolve_device(settings.rerank_device)
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if device.startswith("cuda") else {}
        _nli = CrossEncoder(settings.citation_nli_model or _NLI_MODEL, max_length=512, device=device, **kw)
    return _nli


def nli_scores(pairs: list[tuple[str, str]]) -> tuple[list[float], list[float]]:
    """(근거, 문장) 쌍마다 (함의 확률, 모순 확률). 라벨 순서는 모델 설정에서 읽는다."""
    if not pairs:
        return [], []
    m = get_nli()
    labels = {v.lower(): k for k, v in m.model.config.id2label.items()}
    probs = m.predict(pairs, apply_softmax=True, batch_size=16)
    return ([float(p[labels["entailment"]]) for p in probs], [float(p[labels["contradiction"]]) for p in probs])


def _rerank_scorer(pairs: list[tuple[str, str]]) -> Sequence[float]:
    from rag.core.llm import get_reranker
    return get_reranker().predict(pairs, batch_size=32)


def check_numbers(sentence: str, cites: list[int], chunk_nums: list[set[str]],
                  ignore: set[str] | None = None) -> tuple[list[str], list[int], bool]:
    """문장의 숫자 중 인용한 근거(없으면 모든 근거)에 없는 것, 그 숫자가 모두 있는 다른 근거 번호, 대조했는지."""
    nums = answer_numbers(claim_text(sentence), ignore)
    if not nums:
        return [], [], False
    k = len(chunk_nums)
    valid = [n for n in cites if 1 <= n <= k] or list(range(1, k + 1))
    have = set().union(*(chunk_nums[n - 1] for n in valid))
    missing = [x for x in nums if x not in have]
    elsewhere = [i + 1 for i in range(k) if i + 1 not in valid and missing and set(missing) <= chunk_nums[i]]
    return missing, elsewhere, True


def _check_rows(rows: list[str], chunk_nums: list[set[str]], q_nums: set[str]) -> list[SentenceCheck]:
    out = []
    for r in rows:   # 표 행: 숫자만(인용 번호가 없으면 모든 근거와)
        cites = parse_cites(r)
        missing, elsewhere, numeric = check_numbers(r, cites, chunk_nums, q_nums)
        if numeric:
            out.append(SentenceCheck(text=claim_text(r), cites=cites, status="table_row",
                                     numbers_missing=missing, numbers_found_in=elsewhere))
    return out


def check_table_numbers(markdown: str, chunks: list[RetrievedChunk]) -> dict:
    """마크다운 표(논문 비교표 등)의 숫자가 칸에 단 근거 번호의 원문에 있는지(LLM·리랭커 없음)."""
    rows = _check_rows(table_rows(markdown), [chunk_numbers(c.content) for c in chunks], set())
    return {"rows": len(rows), "mismatch": [asdict(r) for r in rows if r.numbers_missing]}


NLIScorer = Callable[[list[tuple[str, str]]], tuple[list[float], list[float]]]


def check(answer: str, chunks: list[RetrievedChunk], scorer: Scorer | None = None,
          threshold: float | None = None, question: str = "", nli: NLIScorer | bool | None = None) -> CitationReport:
    """답변의 주장 문장마다 인용한 근거가 뒷받침하는지 점수로, 숫자가 근거에 있는지 규칙으로 확인.
    question 의 숫자(예: '상위 20개')는 근거에 없어도 문제 삼지 않는다."""
    thr = settings.citation_support_threshold if threshold is None else threshold
    report = CitationReport(threshold=thr)
    sents = [s for s in split_sentences(answer) if _is_claim(s)]
    rows = table_rows(answer)
    if not (sents or rows) or not chunks:
        return report
    pairs = [(claim_text(s), c.content) for s in sents for c in chunks]
    flat = [float(x) for x in (scorer or _rerank_scorer)(pairs)] if pairs else []
    k = len(chunks)
    chunk_nums = [chunk_numbers(c.content) for c in chunks]
    q_nums = set(answer_numbers(question)) | {_norm(*m.groups()) for m in _CHUNK_NUM.finditer(question or "")}
    report.table_rows = _check_rows(rows, chunk_nums, q_nums)
    report.numeric += len(report.table_rows)
    for i, s in enumerate(sents):
        row = flat[i * k:(i + 1) * k]
        best = max(range(k), key=row.__getitem__)
        cites = parse_cites(s)
        valid = [n for n in cites if 1 <= n <= k]
        support = max((row[n - 1] for n in valid), default=None)
        if not cites:
            status = "uncited"
        elif not valid:
            status = "bad_ref"
        elif support >= thr:
            status = "supported"
        elif row[best] >= thr:
            status = "miscited"
        else:
            status = "weak"
        missing, elsewhere, numeric = check_numbers(s, cites, chunk_nums, q_nums)
        report.numeric += numeric
        report.sentences.append(SentenceCheck(
            text=claim_text(s), cites=cites, status=status,
            support=None if support is None else round(support, 3),
            best=best + 1, best_score=round(row[best], 3),
            numbers_missing=missing, numbers_found_in=elsewhere))
    _apply_nli(report, sents, chunks, nli)
    return report


def _apply_nli(report: CitationReport, sents: list[str], chunks: list[RetrievedChunk], nli) -> None:
    """리랭커가 '뒷받침'으로 본 문장만 함의 판정: 인용한 근거 모두와 모순 확률이 기준 이상이면 contradicted."""
    if nli is False or (nli is None and not settings.citation_nli):
        return
    fn = nli if callable(nli) else nli_scores
    k = len(chunks)
    jobs = [(i, n) for i, sc in enumerate(report.sentences) if sc.status == "supported"
            for n in sc.cites if 1 <= n <= k]
    if not jobs:
        return
    ent, con = fn([(chunks[n - 1].content, claim_text(sents[i])) for i, n in jobs])
    by: dict[int, list[tuple[float, float]]] = {}
    for (i, _), e, c in zip(jobs, ent, con):
        by.setdefault(i, []).append((e, c))
    for i, vals in by.items():
        sc = report.sentences[i]
        sc.entail = round(max(e for e, _ in vals), 3)
        sc.contradiction = round(min(c for _, c in vals), 3)
        if sc.contradiction >= settings.citation_nli_contradiction:
            sc.status = "contradicted"
