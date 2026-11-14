"""대화 저장 API(server/conversations.py). LLM 호출 없음.

GET    /api/conversations          최근 대화 목록(id·제목·메시지 수·시각)
GET    /api/conversations/{id}     대화 하나(메시지 포함)
PUT    /api/conversations/{id}     대화 전체 저장(빈 목록이면 삭제)
DELETE /api/conversations/{id}
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from server.conversations import ConversationStore
from server.schemas import ConversationPut

router = APIRouter(prefix="/api/conversations")


def get_store(request: Request) -> ConversationStore | None:
    return request.app.state.conversations


def _need(store: ConversationStore | None) -> ConversationStore:
    if store is None:
        raise HTTPException(404, "대화 저장이 꺼져 있습니다(CONVERSATIONS=0).")
    return store


@router.get("")
def list_(store: ConversationStore | None = Depends(get_store)) -> dict:
    return {"enabled": store is not None, "items": store.list() if store else []}


@router.get("/{cid}")
def get(cid: str, store: ConversationStore | None = Depends(get_store)) -> dict:
    c = _need(store).get(cid)
    if c is None:
        raise HTTPException(404, "없는 대화입니다.")
    return c


@router.put("/{cid}")
def put(cid: str, req: ConversationPut, store: ConversationStore | None = Depends(get_store)) -> dict:
    return _need(store).put(cid, req.messages, req.title)


@router.delete("/{cid}")
def delete(cid: str, store: ConversationStore | None = Depends(get_store)) -> dict:
    return {"ok": _need(store).delete(cid)}
