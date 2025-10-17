from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from database.session import get_db
from agents.orchestrator import Orchestrator

router = APIRouter()
orch = Orchestrator()

class ScoreRequest(BaseModel):
    data_type: Optional[str] = None  # e.g., "bank_marketing", "crm_lead"
    payload: Dict[str, Any]

@router.post("/score")
def score_endpoint(req: ScoreRequest, db: Session = Depends(get_db)):
    try:
        result = orch.handle_score(db, data_type=req.data_type, payload=req.payload)
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/score/batch")
def score_batch(
    data_type: str = Query(..., description="Adapter key, e.g. bank_marketing"),
    limit: Optional[int] = 100,
    db: Session = Depends(get_db),
):
    try:
        result = orch.handle_score_batch(db, data_type=data_type, limit=limit)
        return {"status": "ok", "count": len(result), "data": result[:20]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
