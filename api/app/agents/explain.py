from __future__ import annotations
from typing import Dict, Any
from ..llm import chat_completion

def explain_vi(context: Dict[str, Any]) -> str:
    messages = [
        {"role": "user", "content": f"Hãy giải thích ngắn gọn bằng tiếng Việt cho người dùng: {context}"}
    ]
    return chat_completion(messages, temperature=0.2)
