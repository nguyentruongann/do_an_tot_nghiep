from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from app.llm import chat_completion
from app.redis_store import log_event
from app.data_index import get_index
from app.agents.model import calculate_statistics, calculate_total_status_count, build_model_prompt

router = APIRouter(prefix="/chat", tags=["chat"])

# Các cột quan trọng bạn cần trích xuất
USEFUL_COLS = ["source", "status", "is_vip", "no_of_employees", "city"]

class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "session_id": "s1",
            "question": "Tổng kết giúp mình các lead tiềm năng cao"
        }
    })
    session_id: str = Field(default="s1")
    question: str = Field(default="xin chào")

class ChatResponse(BaseModel):
    reply: str
    sources: list
    priority_chain: list
    useful_info: dict  # Cột thông tin quan trọng trả về

def build_prompt_to_identify_question_type(question: str) -> str:
    """
    Tạo prompt để LLM phân biệt giữa câu hỏi thông thường và câu hỏi chấm điểm.
    Khi câu hỏi yêu cầu chấm điểm, yêu cầu LLM trả về một đối tượng JSON có các trường cột cần thống kê.
    """
    prompt = f"""
    Bạn là một hệ thống AI thông minh. Bạn sẽ nhận được một câu hỏi. Dựa trên nội dung câu hỏi, hãy xác định nếu câu hỏi yêu cầu chấm điểm hay không.

    Câu hỏi: "{question}"

    Nếu đây là một câu hỏi chấm điểm, vui lòng phân tích và trả về thông tin về các cột dưới đây:
        - source
        - status
        - is_vip
        - no_of_employees
        - city

    Đầu ra của bạn cần phải là một đối tượng JSON có dạng:
    {{
        "source": "<giá trị>",
        "status": "<giá trị>",
        "is_vip": <giá trị>,
        "no_of_employees": "<giá trị>",
        "city": "<giá trị>"
    }}

    Nếu câu hỏi không yêu cầu chấm điểm, vui lòng trả lời ngay mà không cần phân tích thống kê. 
    Trả lời trực tiếp câu hỏi của người dùng mà không cần trả về JSON hay thông tin thống kê.
    """
    return prompt


# Xử lý câu hỏi và phân biệt câu hỏi thông thường với câu hỏi chấm điểm
@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    idx = get_index()
    q = req.question or ""

    # Tạo prompt để LLM phân biệt câu hỏi thông thường và câu hỏi chấm điểm
    prompt = build_prompt_to_identify_question_type(q)

    # Giả sử gửi prompt cho LLM và nhận lại phản hồi (giả lập quá trình gửi đến LLM)
    response = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)

    if "chấm điểm" in response or "đánh giá" in response:
        # Nếu là câu hỏi chấm điểm, thực hiện phân tích và chấm điểm
        rows, grounded = idx.retrieve(req.session_id, q, k_each=3, k_total=9)
        priority = [layer.name for layer in idx.get_layers(req.session_id)]

        # Chuyển rows thành DataFrame từ dữ liệu trả về từ LLM
        df_rows = pd.DataFrame(rows)

        # Lấy dữ liệu từ file CSV để thống kê
        data_csv_path = "/app/data/training/data.csv"  # Cập nhật đường dẫn file CSV nếu cần
        df_csv = pd.read_csv(data_csv_path)

        # Kiểm tra nếu cột 'qualification_status' có trong DataFrame không
        if 'qualification_status' not in df_csv.columns:
            return ChatResponse(
                reply="Không có thông tin về trạng thái chấm điểm trong dữ liệu.",
                sources=[],
                priority_chain=[],
                useful_info={}
            )

        # Trích xuất các thông tin quan trọng từ rows
        useful_info = {}
        for col in USEFUL_COLS:
            useful_info[col] = [row.get(col, None) for row in rows]

        # Tính toán thống kê dữ liệu dựa trên cả thông tin từ CSV và thống kê từ LLM
        statistics = calculate_statistics(df_csv, useful_info)  # Truyền DataFrame vào hàm
        total_status_count = calculate_total_status_count(df_csv)  # Truyền DataFrame vào hàm

        # Gửi prompt tới LLM để chấm điểm
        prompt_for_llm = build_model_prompt(statistics, useful_info, total_status_count)

        # Gửi thông tin này cho LLM
        messages = [{"role": "user", "content": prompt_for_llm}]
        reply = chat_completion(messages, temperature=0.2)

        log_event(req.session_id, "chat_reply", {"reply": reply})

        return ChatResponse(
            reply=reply,
            sources=rows[:5],
            priority_chain=priority,
            useful_info=useful_info  # Trả về thông tin quan trọng
        )
    else:
        # Nếu không phải câu hỏi chấm điểm, trả lời ngay
        reply = chat_completion([{"role": "user", "content": q}], temperature=0.2)
        return ChatResponse(
            reply=reply,
            sources=[],
            priority_chain=[],
            useful_info={}
        )
