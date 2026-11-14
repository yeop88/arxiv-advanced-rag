"""사람 판정 기준 세트(eval/human_set.py) 판정 화면용 API. LLM 호출 없음.

GET  /api/label/items        문항(질문·답변·근거 원문·문장별 자동 판정)과 지금까지의 판정
POST /api/label/{item_id}    한 문항의 판정 저장(문장별 + 답변 전체 + 메모 + 판정 주체)

판정 주체(annotator)가 human 이 아닌 판정은 '초안'이다(예: 다른 모델이 먼저 채운 것). 화면은 초안을 미리 채워 보여 주고,
사람이 검토해 저장하면 human 판정이 되며 원래 초안은 model_draft 로 함께 남긴다(사람이 얼마나 고쳤는지 볼 수 있게).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException

from config import settings
from server.schemas import LabelRequest

router = APIRouter(prefix="/api/label")
_lock = threading.Lock()


def _items() -> list[dict]:
    p = Path(settings.human_set_path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _labels() -> dict:
    p = Path(settings.human_labels_path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@router.get("/items")
def items() -> dict:
    """판정할 문항. 자동 판정(LLM 검증·인용 확인)은 사람 판정이 흔들리지 않게 화면에서 접어 둔다."""
    labels = _labels()
    rows = _items()
    human = sum(bool(labels.get(r["id"], {}).get("overall")) and labels[r["id"]].get("annotator", "human") == "human" for r in rows)
    return {"items": rows, "labels": labels, "done": sum(bool(labels.get(r["id"], {}).get("overall")) for r in rows),
            "human_done": human}


@router.post("/{item_id}")
def save(item_id: str, req: LabelRequest) -> dict:
    ids = {r["id"]: r for r in _items()}
    if item_id not in ids:
        raise HTTPException(404, "기준 세트에 없는 문항입니다.")
    n = len(ids[item_id]["citation_check"]["sentences"])
    bad = [k for k in req.sentences if not k.isdigit() or int(k) >= n]
    if bad:
        raise HTTPException(422, f"없는 문장 번호: {bad}")
    with _lock:
        labels = _labels()
        new = req.model_dump()
        old = labels.get(item_id)
        if old and new["annotator"] == "human":
            if old.get("annotator", "human") != "human":   # 모델 초안을 사람이 검토해 저장 → 초안은 따로 남긴다
                new["model_draft"] = {k: old.get(k) for k in ("sentences", "overall", "note", "annotator")}
            elif old.get("model_draft"):
                new["model_draft"] = old["model_draft"]
        labels[item_id] = new
        p = Path(settings.human_labels_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
    return {"ok": True, "done": sum(bool(v.get("overall")) for v in labels.values()),
            "human_done": sum(bool(v.get("overall")) and v.get("annotator", "human") == "human" for v in labels.values())}
