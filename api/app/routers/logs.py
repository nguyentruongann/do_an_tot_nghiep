from __future__ import annotations
from fastapi import APIRouter
from ..redis_store import read_logs

router = APIRouter(prefix="/logs", tags=["logs"])

@router.get("/{session_id}")
def read(session_id: str, last_n: int = 200):
    return {"session_id": session_id, "items": read_logs(session_id, last_n=last_n)}
