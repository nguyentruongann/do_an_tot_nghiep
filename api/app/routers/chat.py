from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List

from app.llm import chat_completion
from app.redis_store import log_event
from app.agents.model import (
    USEFUL_COLS,
    extract_info_from_text,
    score_rows_and_save,
)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "session_id": "s1",
            "question": "Tôi có một khách hàng is_vip = 1 ở Hà Nội, chấm điểm giúp mình"
        }
    })
    session_id: str = Field(default="s1")
    question: str = Field(default="xin chào")


class ChatResponse(BaseModel):
    reply: str
    sources: list
    priority_chain: list
    useful_info: dict


def is_scoring_question(question: str) -> bool:
    """
    Heuristic nhẹ để nhận diện câu hỏi chấm điểm.
    Tránh để LLM tự đoán lung tung.
    """
    q = (question or "").lower()
    scoring_keywords = [
        "chấm điểm",
        "đánh giá",
        "score",
        "scoring",
        "xếp hạng",
        "phân loại",
        "lead vip",
        "lead tiềm năng",
        "ưu tiên",
    ]
    return any(kw in q for kw in scoring_keywords)


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = req.question or ""

    # === 1. Câu hỏi KHÔNG chấm điểm -> chat thường ===
    if not is_scoring_question(q):
        reply = chat_completion(
            [{"role": "user", "content": q}],
            temperature=0.2,
        )
        log_event(req.session_id, "chat_reply_normal", {
            "question": q,
            "reply": reply,
        })
        return ChatResponse(
            reply=reply,
            sources=[],
            priority_chain=[],
            useful_info={},
        )

    # === 2. Câu hỏi chấm điểm ===

    # 2.1. Dùng LLM trích xuất đúng 5 field USEFUL_COLS từ câu hỏi
    extracted_record, raw_reply_extract = extract_info_from_text(q)

    if not extracted_record:
        # Không trích xuất được thì báo lại cho user
        reply = (
            "Mình hiểu đây là câu hỏi CHẤM ĐIỂM lead, "
            "nhưng chưa trích xuất được các trường cần thiết "
            f"({', '.join(USEFUL_COLS)}) từ mô tả.\n\n"
            "Bạn thử mô tả rõ hơn, ví dụ:\n"
            "- \"Khách hàng is_vip = 1, city = Hà Nội, status = New, số nhân viên khoảng 200\""
        )
        log_event(req.session_id, "chat_scoring_extract_failed", {
            "question": q,
            "raw_reply_extract": raw_reply_extract,
        })
        return ChatResponse(
            reply=reply,
            sources=[],
            priority_chain=["nl_score"],
            useful_info={"raw_reply_extract": raw_reply_extract},
        )

    # Đảm bảo đủ key cho USEFUL_COLS
    normalized_record: Dict[str, Any] = {
        col: extracted_record.get(col, "") for col in USEFUL_COLS
    }

    # 2.2. Chấm điểm + LƯU CSV bằng hàm có sẵn trong model.py
    score_result = score_rows_and_save(req.session_id, [normalized_record])
    preview: List[Dict[str, Any]] = score_result.get("results_preview") or []
    scored_row: Dict[str, Any] = preview[0] if preview else {}

    score = int(scored_row.get("score", 0))
    qualification_status = scored_row.get("qualification_status", "")
    explain = scored_row.get("explain", "")

    # 2.3. Soạn câu trả lời cuối gửi cho user
    reply_lines = [
        "Mình đã chấm điểm lead dựa trên thông tin bạn cung cấp:",
        "",
        f"- Điểm: {score}",
        f"- Phân loại: {qualification_status or '(không xác định)'}",
    ]
    if explain:
        reply_lines.append(f"- Giải thích: {explain}")
    reply = "\n".join(reply_lines)

    # 2.4. Gửi kèm useful_info cho frontend (đúng 5 cột + meta)
    useful_info: Dict[str, Any] = {
        col: normalized_record.get(col) for col in USEFUL_COLS
    }
    useful_info.update({
        "score": score,
        "qualification_status": qualification_status,
        "explain": explain,
        "raw_reply_extract": raw_reply_extract,
        "saved_session_csv": score_result.get("saved_session_csv"),
        "updated_global_csv": score_result.get("updated_global_csv"),
        "rows_scored": score_result.get("rows_scored"),
        "results_preview": preview,
    })

    log_event(req.session_id, "chat_reply_scoring", {
        "question": q,
        "extracted_record": normalized_record,
        "score": score,
        "qualification_status": qualification_status,
    })

    return ChatResponse(
        reply=reply,
        # Cho UI xem nhanh record đã chấm điểm trong khung "Xem dữ liệu liên quan"
        sources=preview,
        # Đặt tên chain cho vui, để debug
        priority_chain=["nl_score"],
        useful_info=useful_info,
    )
