from __future__ import annotations
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    cols: List[str]
    prompt: str
    target: Optional[str]

class ScoreRunResponse(BaseModel):
    rows: int
    metrics: Dict[str, Any]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    intent: str
    text: str
    payload: Optional[Dict[str, Any]] = None