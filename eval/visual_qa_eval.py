"""
eval/visual_qa_eval.py — 표·그림·수식 질문으로 표·그림·수식 근거 청크의 효과 측정

같은 질문을 근거 청크를 넣기 전 DB(백업 복사본, CHROMA_DIR 로 지정)와 넣은 뒤 DB(본 DB)에 묻고,
답에 기대 문자열이 모두 들어 있는지(대안 중 하나면 됨)와 그 논문의 해당 종류 청크가 근거로 쓰였는지 본다.
자동 수집·검증은 끈다(정답 여부만 비교). LLM: 로컬(질문당 질의 변환 + 생성).
    CHROMA_DIR=./storage/backup_before_fig_eq/chroma python eval/visual_qa_eval.py --label fe_before --kinds figure equation
    python eval/visual_qa_eval.py --label fe_after --kinds figure equation
    python eval/visual_qa_eval.py --report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
# (질문, 답에 모두 있어야 하는 것, 근거 논문, 종류). 기대값의 각 항목은 문자열 또는 대안 목록(하나만 있으면 됨).
# 표: 정답은 추출 표에서 고르되 원문 PDF 텍스트에도 있는 값. 수식: 원문 페이지 이미지와 대조한 기호(§5.38)
# 표 앞 7개는 시험 DB 4편(§5.35), 다음 4개는 본 DB 적용 뒤 추가(§5.37), 수식·그림은 §5.38
QUESTIONS = [
    ("DPR 논문에서 단일 데이터셋으로 학습한 DPR의 Natural Questions Top-20 검색 정확도는 얼마인가요?", ["78.4"], "2004.04906", "table"),
    ("DPR 논문에서 BM25의 TriviaQA Top-100 검색 정확도는 얼마인가요?", ["76.7"], "2004.04906", "table"),
    ("FlashAttention으로 GPT-2 small을 학습하면 학습 시간이 며칠이고 Huggingface 대비 몇 배 빠른가요?", ["2.7", "3.5"], "2205.14135", "table"),
    ("FlashAttention 논문에서 BERT-large 학습 시간은 몇 분이었나요?", ["17.4"], "2205.14135", "table"),
    ("FlashAttention의 Long-Range-Arena 평균 점수와 속도 향상 배수는 얼마인가요?", ["59.8", "2.4"], "2205.14135", "table"),
    ("NeRF 논문의 ablation에서 위치 인코딩을 제거했을 때 PSNR은 얼마인가요?", ["28.77"], "2003.08934", "table"),
    ("NeRF의 realistic synthetic 데이터셋에서 Lego 장면의 PSNR은 얼마인가요?", ["32.54"], "2003.08934", "table"),
    ("Lost in the Middle 논문에서 문서 20개 설정일 때 GPT-3.5-Turbo는 정답 문서가 맨 앞(index 0)일 때와 index 9일 때 정확도가 각각 얼마인가요?",
     ["75.8", "53.8"], "2307.03172", "table"),
    ("CRAG 논문의 검색 평가기는 PopQA에서 정확도가 얼마이고 ChatGPT는 얼마인가요?", ["84.3", "58.0"], "2401.15884", "table"),
    ("CRAG의 토큰당 TFLOPs와 인스턴스당 실행 시간은 얼마인가요?", ["27.2", "0.512"], "2401.15884", "table"),
    ("CRAG ablation에서 LLaMA2-hf-7b로 Incorrect 행동만 썼을 때 PopQA 정확도는 얼마인가요?", ["47.0"], "2401.15884", "table"),
    ("NeRF에서 카메라 광선의 기대 색 C(r)을 구하는 볼륨 렌더링 적분식과 T(t)의 정의를 수식으로 알려주세요.",
     ["T(t)", "exp"], "2003.08934", "equation"),
    ("Mamba 논문에서 연속 파라미터 Δ, A, B를 이산 파라미터로 바꾸는 ZOH 이산화 공식은 무엇인가요?",
     [["exp(ΔA)", "exp(Δ A)", r"exp(\Delta A)", r"exp(\Delta \mathbf{A})", r"exp(\Delta\mathbf{A})"], ["^{-1}", "⁻¹"]],
     "2312.00752", "equation"),
    ("DPR 리더에서 답 구간의 시작 위치 확률은 어떤 수식으로 계산하나요?", ["softmax", "start"], "2004.04906", "equation"),
    ("Diffusion Policy에서 관측을 조건으로 행동을 디노이징하는 한 단계의 수식은 무엇인가요?",
     [["γ", r"\gamma"], ["α", r"\alpha"], ["O_t", "O_{t}", r"\mathbf{O}_t"]], "2303.04137", "equation"),
    ("Diffusion Policy의 관측 조건부 학습 손실 함수 수식은 무엇인가요?", ["MSE", ["ε", r"\epsilon", r"\varepsilon"]], "2303.04137", "equation"),
    ("FlashAttention 논문 Figure 1에 따르면 GPT-2에서 attention 계산이 PyTorch 대비 몇 배 빨라지나요?", ["7.6"], "2205.14135", "figure"),
    ("NeRF 논문 Figure 2는 장면 표현과 렌더링 과정을 어떤 단계로 보여주나요?",
     [["MLP", "network", "네트워크"], ["volume rendering", "볼륨 렌더링"], ["loss", "손실"]], "2003.08934", "figure"),
]


def _has(answer: str, expected) -> bool:
    return any(a in answer for a in (expected if isinstance(expected, list) else [expected]))


def run(label: str, kinds: list[str] | None = None) -> None:
    from rag.pipeline import RAGPipeline

    if settings.llm_provider != "local":
        raise SystemExit("로컬 LLM(LLM_PROVIDER=local)에서만 실행합니다.")
    settings.local_thinking = settings.local_thinking_light = "off"
    settings.auto_fetch = False
    settings.enable_verification = False
    settings.exclude_auto_papers = False
    out = RESULTS / f"visual_qa_{label}.jsonl"
    out.unlink(missing_ok=True)
    pipe = RAGPipeline()
    for q, expected, src, kind in QUESTIONS:
        if kinds and kind not in kinds:
            continue
        t0 = time.perf_counter()
        res = pipe.run(q)
        row = {"question": q, "kind": kind, "source": src, "expected": expected,
               "correct": all(_has(res.answer, e) for e in expected),
               "visual_in_evidence": any(c.kind == kind and c.source == src for c in res.reranked),
               "sec": round(time.perf_counter() - t0, 1), "answer": res.answer[:800]}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  정답={row['correct']!s:<5} 근거={row['visual_in_evidence']!s:<5} {row['sec']}s | {kind} | {q[:40]}", flush=True)


def report() -> None:
    for p in sorted(RESULTS.glob("visual_qa_*.jsonl")):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_kind: dict[str, list] = {}
        for r in rows:
            by_kind.setdefault(r.get("kind", "table"), []).append(r)
        parts = [f"{k} 정답 {sum(r['correct'] for r in rs)}/{len(rs)} 근거 "
                 f"{sum(r.get('visual_in_evidence', r.get('table_in_evidence', False)) for r in rs)}/{len(rs)}"
                 for k, rs in by_kind.items()]
        print(f"{p.stem.removeprefix('visual_qa_'):<22} " + " | ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="결과 파일 구분자(before, after 등)")
    ap.add_argument("--kinds", nargs="+", choices=["table", "figure", "equation"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.label:
        run(args.label, args.kinds)
    report()


if __name__ == "__main__":
    main()
