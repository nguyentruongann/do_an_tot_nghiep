from __future__ import annotations
import requests
from typing import List, Dict
from .config import load_config

_VI_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Bạn là trợ lý AI nói tiếng Việt. Luôn trả lời bằng **tiếng Việt rõ ràng, ngắn gọn**, "
        "ngoại trừ khi người dùng **yêu cầu một ngôn ngữ khác**. Nếu người dùng viết bằng tiếng Việt "
        "thì giữ phong cách tự nhiên, thân thiện."
    ),
}

def chat_completion(messages: List[Dict[str, str]], temperature: float | None = None) -> str:
    cfg = load_config()
    url = cfg.llm.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.llm.api_key}"}
    # Ép hệ thống dùng tiếng Việt bằng cách chèn system message nếu chưa có
    if not messages or messages[0].get("role") != "system":
        messages = [_VI_SYSTEM_MESSAGE] + messages
    payload = {
        "model": cfg.llm.model,
        "temperature": cfg.llm.temperature if temperature is None else temperature,
        "messages": messages,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)
