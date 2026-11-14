"""
eval/title_candidates_eval.py — 원 논문 제목 후보 여러 개 추정 효과(§5.43). 경량 LLM(질문당 1회) + arXiv 제목 검색

원 논문 찾기(A3)는 경량 LLM 이 추정한 제목 하나로 arXiv 를 검색한다. 통칭과 실제 제목이 크게 다른 논문(ViT, Whisper …)은
추정이 자주 틀려, 제목 후보를 최대 3개 받아 검색하면 원 논문이 후보에 드는지 본다.
'제목 1개'는 original_paper_eval 기록의 결과, '후보 3개'는 이번에 새로 추정·검색한 결과다(조정용 34·확인용 12문항).
사용: python eval/title_candidates_eval.py   결과: eval/results/title_candidates_eval.json
"""
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
settings.local_thinking_light = "off"
from rag.ingest.auto_fetch import guess_original_titles, title_candidates
R = str(Path(__file__).resolve().parent / "results") + "/"
qmap = {json.loads(l)["id"]: json.loads(l) for l in open(R + "fetch_signal_record.jsonl", encoding="utf-8")}
out = {}
for name, f in (("조정용", "original_paper_eval.jsonl"), ("확인용", "original_paper_eval_heldout.jsonl")):
    rows = [json.loads(l) for l in open(R + f, encoding="utf-8")]
    one = multi = 0
    res = []
    for r in rows:
        key = r["key"].split("v")[0]
        q = (qmap.get(r["id"]) or {}).get("standalone") or r["question"]
        a = bool(r["key_in_pool"].get("title"))
        titles = guess_original_titles(q)
        b = any(any(p.pid.split("v")[0] == key for p in title_candidates(t)) for t in titles)
        one += a; multi += b
        res.append({"id": r["id"], "single_found": a, "multi_found": b, "titles": titles})
        print(name, r["id"], a, b, titles, flush=True)
    out[name] = {"n": len(rows), "single": one, "multi": multi, "rows": res}
    print(f"{name} {len(rows)}문항: 제목 1개 {one} → 후보 3개 {multi}", flush=True)
for v in out.values():
    v["union"] = sum(r["single_found"] or r["multi_found"] for r in v["rows"])
(Path(R) / "title_candidates_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
