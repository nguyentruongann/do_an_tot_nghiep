from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
"""
This router provides the ``/score`` endpoint used to batch score leads.  It
imports ``score_rows_and_save`` from the top‑level ``model`` module (rather
than a nested ``agents`` package) and logs events using ``redis_store``.
"""

from app.agents.model import score_rows_and_save
from app.redis_store import log_event

router = APIRouter(prefix="/score", tags=["score"])

class ScoreRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "session_id": "s1",
            "rows": [
                {"lead_id":"L-1","order_count":3,"revenue":1200,"currency":"USD","notes":"khách phản hồi tốt"},
                {"lead_id":"L-2","order_count":1,"revenue":200}
            ]
        }
    })
    session_id: str = Field(default="s1")
    rows: List[Dict[str, Any]] = Field(default_factory=list)

class ScoreResponse(BaseModel):
    saved_session_csv: str
    updated_global_csv: str
    rows_scored: int
    results_preview: list

@router.post("", response_model=ScoreResponse)
def score(req: ScoreRequest):
    result = score_rows_and_save(req.session_id, req.rows)
    log_event(req.session_id, "score_batch", {"count": len(req.rows)})
    return ScoreResponse(**result)
