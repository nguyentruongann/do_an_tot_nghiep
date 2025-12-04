from __future__ import annotations

import os
from typing import List, Dict, Any

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict

from app.agents.model import score_rows_and_save
from app.redis_store import log_event

router = APIRouter(prefix="/score", tags=["score"])


class ScoreRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "s1",
                "rows": [
                    {
                        "lead_id": "L-1",
                        "order_count": 3,
                        "revenue": 1200,
                        "currency": "USD",
                        "notes": "khách phản hồi tốt",
                    },
                    {"lead_id": "L-2", "order_count": 1, "revenue": 200},
                ],
            }
        }
    )
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


@router.post("/csv")
def score_csv(
    session_id: str = Form("s1"),
    file: UploadFile = File(..., description="File CSV chứa các lead cần chấm"),
):
    """
    Nhận 1 file CSV, chấm điểm toàn bộ dòng bằng cùng logic score_rows_and_save
    rồi trả về đúng 1 file CSV đã chấm điểm để user tải về.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Vui lòng upload file .csv")

    # Đọc CSV vào DataFrame
    try:
        file.file.seek(0)
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File CSV không có dữ liệu.")

    # Chuyển DataFrame -> list[dict] để tái sử dụng score_rows_and_save
    rows = df.to_dict(orient="records")

    # Chấm điểm và lưu session/global như logic hiện tại
    result = score_rows_and_save(session_id, rows)
    saved_session_csv = result.get("saved_session_csv")

    if not saved_session_csv or not os.path.exists(saved_session_csv):
        raise HTTPException(
            status_code=500,
            detail="Không tìm thấy file CSV đã chấm điểm trên server.",
        )

    # Log lại event cho tiện debug
    log_event(
        session_id,
        "score_csv",
        {
            "filename": filename,
            "rows": len(rows),
            "session_csv": saved_session_csv,
        },
    )

    out_name = f"scored_{os.path.basename(filename) or 'leads.csv'}"
    # Trả về file CSV đã chấm điểm để browser tự tải về
    return FileResponse(
        saved_session_csv,
        media_type="text/csv",
        filename=out_name,
    )
