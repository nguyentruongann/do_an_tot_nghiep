from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict

from app.llm import chat_completion
from app.redis_store import log_event
from app.agents.model import (
    USEFUL_COLS,
    extract_info_from_text,
    score_rows_and_save,
)
from app.agents.analytics import compute_session_summary

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "s1",
                "question": "Tôi có một khách hàng is_vip = 1 ở Hà Nội, chấm điểm giúp mình",
            }
        }
    )
    session_id: str = Field(default="s1")
    question: str = Field(default="xin chào")


class ChatResponse(BaseModel):
    reply: str
    sources: list
    priority_chain: list
    useful_info: dict


# ===== Intent classification (score / analytics / general) =====

_INTENT_SYSTEM_EXTRA = """
Bạn là bộ phân loại intent cho một chatbot chấm điểm lead VIP cho đội sales.

Nhiệm vụ:
- Đọc tin nhắn tiếng Việt của người dùng.
- Phân loại vào đúng 1 trong 3 intent: "score", "analytics", "general".

Định nghĩa chi tiết:

1) "score":
   - Người dùng đang nói về MỘT lead/khách hàng CỤ THỂ và muốn CHẤM ĐIỂM / ĐÁNH GIÁ lead đó.
   - Họ thường mô tả thuộc tính của lead (is_vip, số nhân viên, thành phố, nguồn, trạng thái...),
     hoặc nhắc rõ "lead A", "khách hàng X" rồi yêu cầu chấm điểm.
   - Ví dụ:
     - "Khách này is_vip=1, 200 nhân viên, ở Hà Nội, chấm điểm giúp mình"
     - "Lead A có tiềm năng không, cho mình score"
   - Quan trọng: "score" là KHI NGƯỜI DÙNG CUNG CẤP THÔNG TIN ĐẦU VÀO CHO MỘT LEAD CỤ THỂ.

2) "analytics":
   - Người dùng muốn THỐNG KÊ / BÁO CÁO / SO SÁNH / TÌM TOP trên CẢ TẬP DỮ LIỆU đã chấm của session hiện tại,
     chứ không mô tả một lead mới.
   - Các từ khoá thường gặp: "thống kê", "tổng hợp", "báo cáo", "dashboard",
     "top", "nhất", "nhiều nhất", "ít nhất", "cao nhất", "thấp nhất",
     "so sánh", "phân tích", "xu hướng", "tỷ lệ", "tỉ lệ", "phân bố".
   - Ví dụ:
     - "Thống kê file mình vừa chấm điểm"
     - "Trong session này có bao nhiêu lead Qualified / Unqualified / Junk?"
     - "Nguồn nào mang lại nhiều VIP nhất?"
     - "Ai là khách hàng có tiềm năng nhất trong file này?"
   - Nếu câu hỏi nói về "top lead", "khách hàng tiềm năng nhất", "top 5 lead" v.v.
     thì GẦN NHƯ CHẮC CHẮN là "analytics", KHÔNG phải "score".

3) "general":
   - Mọi trường hợp khác: chào hỏi, hỏi về cách hệ thống hoạt động,
     hỏi về file CSV nhưng KHÔNG yêu cầu thống kê cụ thể,
     hỏi về kết quả chấm nói chung, v.v.
   - Ví dụ:
     - "bạn có thấy file bạn vừa chấm điểm không"
     - "tại sao hệ thống lại chấm điểm như vậy"
     - "giải thích lại giúp mình cách hoạt động"

Lưu ý:
- Câu kiểu "ai là khách hàng có tiềm năng nhất", "top lead trong file này là ai"
  -> PHẢI là "analytics" (vì đang hỏi trên tập dữ liệu đã có).
- Chỉ dùng "score" khi người dùng ĐƯA RA THÔNG TIN MÔ TẢ LEAD để hệ thống chấm điểm lead đó.

Đầu ra:
- Trả về DUY NHẤT MỘT JSON trên một dòng, không giải thích thêm.
- Cấu trúc JSON:
  {"intent": "score"}
  hoặc
  {"intent": "analytics"}
  hoặc
  {"intent": "general"}.
"""


def detect_intent(question: str, session_id: str) -> str:
    """
    Phân loại intent với 2 bước:
    1) Heuristic mạnh cho các case rõ ràng (top/nhất -> analytics, mô tả field -> score)
    2) Gọi LLM cho các trường hợp mơ hồ.

    Trả về: "score" | "analytics" | "general".
    Nếu lỗi hoặc không parse được JSON -> "general".
    """
    q = (question or "").strip()
    if not q:
        return "general"

    q_lower = q.lower()

    # --- 1. Heuristic ưu tiên analytics ---
    analytics_kw = [
        "thống kê",
        "tổng hợp",
        "báo cáo",
        "dashboard",
        "top ",
        " top",
        "nhất",
        "nhiều nhất",
        "ít nhất",
        "cao nhất",
        "thấp nhất",
        "phân tích",
        "so sánh",
        "xu hướng",
        "tỷ lệ",
        "tỉ lệ",
        "phân bố",
    ]
    entity_kw = ["lead", "khách hàng", "khach hang", "file", "csv", "dữ liệu", "du lieu"]

    if any(kw in q_lower for kw in analytics_kw) and any(
        kw in q_lower for kw in entity_kw
    ):
        # ví dụ: "ai là khách hàng có tiềm năng nhất trong csv vừa chấm"
        return "analytics"

    # --- 2. Heuristic cho score (mô tả field rõ ràng) ---
    score_trigger_kw = ["chấm điểm", "score", "scoring", "đánh giá", "tiềm năng"]
    feature_kw = [
        "is_vip",
        "vip",
        "status",
        "trạng thái",
        "city",
        "thành phố",
        "nguồn",
        "source",
        "số nhân viên",
        "nhân viên",
        "employee",
        "no_of_employees",
    ]

    if any(kw in q_lower for kw in score_trigger_kw) and any(
        kw in q_lower for kw in feature_kw
    ):
        # có cả từ chấm điểm và các field -> gần như chắc là score
        return "score"

    # --- 3. Gọi LLM cho các trường hợp còn lại ---
    try:
        raw = chat_completion(
            [{"role": "user", "content": q}],
            temperature=0.0,
            system_extra=_INTENT_SYSTEM_EXTRA,
        )
    except Exception as e:
        log_event(
            session_id,
            "intent_detect_failed",
            {"question": q, "error": str(e)},
        )
        return "general"

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        log_event(
            session_id,
            "intent_detect_no_json",
            {"question": q, "raw": raw},
        )
        return "general"

    try:
        obj = json.loads(m.group(0))
        intent = (obj.get("intent") or "").lower().strip()
    except Exception as e:
        log_event(
            session_id,
            "intent_detect_json_error",
            {"question": q, "raw": raw, "error": str(e)},
        )
        return "general"

    if intent not in {"score", "analytics", "general"}:
        intent = "general"

    log_event(
        session_id,
        "intent_detect_ok",
        {"question": q, "intent": intent},
    )
    return intent


# ===== Helper: record có thực sự có thông tin cột không? =====

def is_meaningful_record(rec: Dict[str, Any] | None) -> bool:
    """
    Trả về True nếu rec có ÍT NHẤT 1 field USEFUL_COLS có giá trị "có ý nghĩa":
    - không rỗng
    - không phải unknown / n/a / none
    Ý tưởng: nếu không có thông tin cột thì KHÔNG coi là câu chấm điểm lead.
    """
    if not rec:
        return False

    bad_tokens = {"unknown", "không rõ", "khong ro", "n/a", "na", "none", "null", "?"}
    non_empty = 0

    for col in USEFUL_COLS:
        val = rec.get(col, None)
        if val is None:
            continue

        if isinstance(val, str):
            s = val.strip()
            if not s:
                continue
            if s.lower() in bad_tokens:
                continue
        # các kiểu khác (int, float, bool) cứ có là coi như có thông tin
        non_empty += 1

    return non_empty > 0


# ===== Analytics helper (session-only) =====

def handle_analytics_question(session_id: str, question: str) -> tuple[str, Dict[str, Any]]:
    """
    Đọc file CSV đã chấm điểm của SESSION HIỆN TẠI, tính summary,
    rồi nhờ LLM diễn giải.
    Trả về: (reply_text, summary_dict)
    """
    try:
        data = compute_session_summary(session_id)
    except FileNotFoundError:
        reply = (
            "Hiện tại mình chưa tìm thấy file CSV đã chấm điểm cho session này.\n"
            "Bạn hãy chấm ít nhất một lead (hoặc upload CSV) trước, "
            "sau đó hỏi lại: ví dụ 'thống kê file mình vừa chấm điểm'."
        )
        return reply, {}

    summary_json = json.dumps(data, ensure_ascii=False)
    system_msg = (
    "Bạn là trợ lý phân tích dữ liệu cho đội sales.\n"
    "Bạn nhận được:\n"
    "- Câu hỏi của người dùng (tiếng Việt)\n"
    "- Một JSON chứa thống kê và DANH SÁCH CHI TIẾT các lead đã chấm điểm cho SESSION HIỆN TẠI.\n\n"
    "Cấu trúc JSON (quan trọng nhất):\n"
    "- session_summary: các số liệu tổng quan.\n"
    "- rows_for_llm: DANH SÁCH TỪNG LEAD (mỗi phần tử là một object với name, score, "
    "qualification_status, source, status, is_vip, no_of_employees, city...).\n"
    "- top_leads: subset các lead có score cao nhất.\n"
    "- leads_by_status: map từ qualification_status -> danh sách lead tương ứng.\n\n"
    "Hãy trả lời NGẮN GỌN, RÕ RÀNG, bằng tiếng Việt, có gạch đầu dòng nếu phù hợp.\n"
    "- Với các câu hỏi kiểu 'những khách hàng nào là Qualified', 'liệt kê khách hàng Unqualified', "
    "hãy duyệt rows_for_llm hoặc leads_by_status để liệt kê tên (name) + score.\n"
    "- Với câu hỏi 'ai là khách hàng tiềm năng nhất', hãy dùng top_leads[0] hoặc lead có score cao nhất.\n"
    "Đừng bịa số liệu, chỉ dựa trên JSON được cung cấp.\n"
    )


    reply = chat_completion(
        [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": f"Câu hỏi: {question}\n\nDữ liệu thống kê (JSON):\n{summary_json}",
            },
        ],
        temperature=0.1,
    )

    log_event(
        session_id,
        "chat_reply_analytics",
        {"question": question, "summary": data, "reply": reply},
    )

    return reply, data


# ===== Chat router =====

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = req.question or ""

    # 1. Phân loại intent
    intent = detect_intent(q, req.session_id)

    # === 1a. Intent GENERAL -> chat thường ===
    if intent == "general":
        reply = chat_completion(
            [{"role": "user", "content": q}],
            temperature=0.2,
        )
        log_event(
            req.session_id,
            "chat_reply_general",
            {"question": q, "reply": reply},
        )
        return ChatResponse(
            reply=reply,
            sources=[],
            priority_chain=[],
            useful_info={"intent": intent},
        )

    # === 1b. Intent ANALYTICS -> thống kê trên CSV đã chấm của SESSION HIỆN TẠI ===
    if intent == "analytics":
        reply, summary = handle_analytics_question(req.session_id, q)
        return ChatResponse(
            reply=reply,
            sources=[summary] if summary else [],
            priority_chain=["analytics"],
            useful_info={
                "intent": intent,
                "analytics_summary": summary,
            },
        )

    # === 1c. Intent SCORE -> chấm điểm lead ===
    # 2.1. Dùng LLM trích xuất các field USEFUL_COLS từ câu hỏi
    extracted_record, raw_reply_extract = extract_info_from_text(q)

    # 2.1b. Nếu không trích được thông tin cột -> fallback về analytics / general
    if not is_meaningful_record(extracted_record):
        q_lower = q.lower()
        maybe_analytics = any(
            kw in q_lower for kw in ["csv", "file", "tập tin", "tap tin", "dữ liệu", "du lieu"]
        ) or any(
            kw in q_lower
            for kw in ["thống kê", "tổng hợp", "báo cáo", "top", "nhất", "nhiều nhất", "ít nhất"]
        )

        log_event(
            req.session_id,
            "score_intent_but_empty_record",
            {
                "question": q,
                "raw_reply_extract": raw_reply_extract,
                "extracted_record": extracted_record,
                "fallback": "analytics" if maybe_analytics else "general",
            },
        )

        if maybe_analytics:
            # Thực ra là câu hỏi về dữ liệu đã chấm -> analytics
            reply, summary = handle_analytics_question(req.session_id, q)
            return ChatResponse(
                reply=reply,
                sources=[summary] if summary else [],
                priority_chain=["analytics"],
                useful_info={
                    "intent": "analytics",
                    "analytics_summary": summary,
                    "raw_reply_extract": raw_reply_extract,
                },
            )
        else:
            # Coi như chat thường
            reply = chat_completion(
                [{"role": "user", "content": q}],
                temperature=0.2,
            )
            return ChatResponse(
                reply=reply,
                sources=[],
                priority_chain=[],
                useful_info={
                    "intent": "general",
                    "raw_reply_extract": raw_reply_extract,
                },
            )

    # Đảm bảo đủ key cho USEFUL_COLS
    normalized_record: Dict[str, Any] = {
        col: extracted_record.get(col, "") for col in USEFUL_COLS
    }

    # 2.2. Chấm điểm + LƯU CSV
    score_result = score_rows_and_save(req.session_id, [normalized_record])
    preview: List[Dict[str, Any]] = score_result.get("results_preview") or []
    scored_row: Dict[str, Any] = preview[0] if preview else {}

    score = int(scored_row.get("score", 0))
    qualification_status = scored_row.get("qualification_status", "")
    explain = scored_row.get("explain", "")

    # 2.3. Soạn reply
    reply_lines = [
        "Mình đã chấm điểm lead dựa trên thông tin bạn cung cấp:",
        "",
        f"- Điểm: {score}",
        f"- Phân loại: {qualification_status or '(không xác định)'}",
    ]
    if explain:
        reply_lines.append(f"- Giải thích: {explain}")
    reply = "\n".join(reply_lines)

    useful_info: Dict[str, Any] = {col: normalized_record.get(col) for col in USEFUL_COLS}
    useful_info.update(
        {
            "intent": "score",
            "score": score,
            "qualification_status": qualification_status,
            "explain": explain,
            "raw_reply_extract": raw_reply_extract,
            "saved_session_csv": score_result.get("saved_session_csv"),
            "updated_global_csv": score_result.get("updated_global_csv"),
            "rows_scored": score_result.get("rows_scored"),
            "results_preview": preview,
        }
    )

    log_event(
        req.session_id,
        "chat_reply_scoring",
        {
            "question": q,
            "extracted_record": normalized_record,
            "score": score,
            "qualification_status": qualification_status,
        },
    )

    return ChatResponse(
        reply=reply,
        sources=preview,
        priority_chain=["nl_score"],
        useful_info=useful_info,
    )
