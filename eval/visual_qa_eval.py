"""
eval/visual_qa_eval.py — 표 숫자 질문으로 표·그림 근거 청크의 효과 측정

같은 질문을 표 청크가 없는 DB(--label before, 본 DB)와 있는 DB(--label after, CHROMA_DIR 로 지정한 복사본)에 묻고,
답에 정답 숫자가 모두 들어 있는지와 표 청크가 근거로 쓰였는지 본다. 자동 수집·검증은 끈다(정답 여부만 비교).
LLM: 로컬(질문당 질의 변환 + 생성).
    python eval/visual_qa_eval.py --label before
    CHROMA_DIR=./storage/chroma_e2e python eval/visual_qa_eval.py --label after
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
# (질문, 답에 모두 있어야 하는 숫자, 표가 있는 논문) — 정답은 추출 표에서 고르되 원문 PDF 텍스트에도 있는 값
# 앞 7개는 시험 DB 4편(§5.35), 뒤 4개는 본 DB 적용 뒤 추가한 다른 논문(§5.37)
QUESTIONS = [
    ("DPR 논문에서 단일 데이터셋으로 학습한 DPR의 Natural Questions Top-20 검색 정확도는 얼마인가요?", ["78.4"], "2004.04906"),
    ("DPR 논문에서 BM25의 TriviaQA Top-100 검색 정확도는 얼마인가요?", ["76.7"], "2004.04906"),
    ("FlashAttention으로 GPT-2 small을 학습하면 학습 시간이 며칠이고 Huggingface 대비 몇 배 빠른가요?", ["2.7", "3.5"], "2205.14135"),
    ("FlashAttention 논문에서 BERT-large 학습 시간은 몇 분이었나요?", ["17.4"], "2205.14135"),
    ("FlashAttention의 Long-Range-Arena 평균 점수와 속도 향상 배수는 얼마인가요?", ["59.8", "2.4"], "2205.14135"),
    ("NeRF 논문의 ablation에서 위치 인코딩을 제거했을 때 PSNR은 얼마인가요?", ["28.77"], "2003.08934"),
    ("NeRF의 realistic synthetic 데이터셋에서 Lego 장면의 PSNR은 얼마인가요?", ["32.54"], "2003.08934"),
    ("Lost in the Middle 논문에서 문서 20개 설정일 때 GPT-3.5-Turbo는 정답 문서가 맨 앞(index 0)일 때와 index 9일 때 정확도가 각각 얼마인가요?",
     ["75.8", "53.8"], "2307.03172"),
    ("CRAG 논문의 검색 평가기는 PopQA에서 정확도가 얼마이고 ChatGPT는 얼마인가요?", ["84.3", "58.0"], "2401.15884"),
    ("CRAG의 토큰당 TFLOPs와 인스턴스당 실행 시간은 얼마인가요?", ["27.2", "0.512"], "2401.15884"),
    ("CRAG ablation에서 LLaMA2-hf-7b로 Incorrect 행동만 썼을 때 PopQA 정확도는 얼마인가요?", ["47.0"], "2401.15884"),
]


def run(label: str) -> None:
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
    for q, nums, _src in QUESTIONS:
        t0 = time.perf_counter()
        res = pipe.run(q)
        row = {"question": q, "expected": nums, "correct": all(n in res.answer for n in nums),
               "table_in_evidence": any((c.section or "").startswith(("Table", "Figure")) for c in res.reranked),
               "sec": round(time.perf_counter() - t0, 1), "answer": res.answer[:600]}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  정답={row['correct']!s:<5} 표근거={row['table_in_evidence']!s:<5} {row['sec']}s | {q[:40]}", flush=True)


def report() -> None:
    for p in sorted(RESULTS.glob("visual_qa_*.jsonl")):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"{p.stem.removeprefix('visual_qa_'):<22} 정답 {sum(r['correct'] for r in rows)}/{len(rows)}, "
              f"표 근거 사용 {sum(r['table_in_evidence'] for r in rows)}/{len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="결과 파일 구분자(before, after 등)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.label:
        run(args.label)
    report()


if __name__ == "__main__":
    main()
