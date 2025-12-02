from __future__ import annotations
"""
Router cho chức năng chấm điểm lead qua mô tả tự nhiên trong API VIP Scoring.
Endpoint này cho phép người dùng cung cấp mô tả hội thoại về lead, sau đó
trích xuất các trường có cấu trúc bằng LLM và chấm điểm sử dụng cùng
logic như endpoint scoring chuẩn.

Bước trích xuất cần cấu hình endpoint LLM; nếu không có, record trích xuất
sẽ rỗng và quá trình chấm điểm sẽ quay về heuristic. Tất cả sự kiện
được ghi vào ``redis_store``.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any

from ..agents.model import extract_info_from_text, score_lead
from ..redis_store import log_event

router = APIRouter(prefix="/nl-score", tags=["nl-score"])


class NLScoreRequest(BaseModel):
    """Mô hình yêu cầu cho endpoint chấm điểm qua mô tả tự nhiên."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "session_id": "s1",
            "description": "Khách hàng cần tư vấn gói dịch vụ cho doanh nghiệp 200 người ở Đà Nẵng, chức vụ CEO, đã để lại số điện thoại."
        }
    })
    session_id: str = Field(default="s1")
    description: str = Field(..., description="Mô tả bằng ngôn ngữ tự do về lead")


class NLScoreResponse(BaseModel):
    """Mô hình phản hồi cho endpoint chấm điểm qua mô tả tự nhiên."""
    extracted_record: Dict[str, Any]
    score: int
    qualification_status: str
    explain: str
    raw_reply_extract: str = Field(default="")
    raw_reply_score: str = Field(default="")


@router.post("", response_model=NLScoreResponse)
def nl_score(req: NLScoreRequest):
    """Xử lý yêu cầu chấm điểm mô tả tự nhiên: trích xuất record, chấm điểm
    và trả về kết quả."""
    extracted, raw_reply_extract = extract_info_from_text(req.description)
    result = score_lead(extracted)
    log_event(req.session_id, "nl_score_request", {
        "description": req.description,
        "extracted": extracted,
        "result": result
    })
    return NLScoreResponse(
        extracted_record=extracted,
        score=result.get("score", 0),
        qualification_status=result.get("qualification_status", ""),
        explain=result.get("explain", ""),
        raw_reply_extract=raw_reply_extract,
        raw_reply_score=result.get("raw_reply", "")
    )