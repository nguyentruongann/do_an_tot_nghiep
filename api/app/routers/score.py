from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from ..agents.features import extract_features
from ..agents.model import score_from_features
from ..redis_store import log_event

router = APIRouter(prefix="/score", tags=["score"])

class ScoreRequest(BaseModel):
    session_id: str = "default"
    text: str

class ScoreResponse(BaseModel):
    score: int
    explain: str
    cached: bool

@router.post("", response_model=ScoreResponse)
def score(req: ScoreRequest):
    features = extract_features({"text": req.text})
    result = score_from_features(features)
    log_event(req.session_id, "score", {"features": features, **result})
    return ScoreResponse(**result)
