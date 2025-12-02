from __future__ import annotations
import requests
from fastapi import HTTPException
from typing import List, Dict, Optional
from app.config import load_config

_BASE_SYSTEM = "Bạn là trợ lý AI nói tiếng Việt. Luôn trả lời bằng tiếng Việt rõ ràng, ngắn gọn, trừ khi người dùng yêu cầu ngôn ngữ khác."

def chat_completion(messages: List[Dict[str, str]], temperature: float | None = None, system_extra: Optional[str] = None) -> str:
    cfg = load_config()
    url = cfg.llm.base_url.rstrip('/') + '/chat/completions'
    headers = {"Authorization": f"Bearer {cfg.llm.api_key}", "Content-Type": "application/json"}
    system_content = _BASE_SYSTEM + (" " + system_extra.strip() if system_extra else "")
    if not messages or messages[0].get("role") != "system":
        messages = [{"role":"system","content":system_content}] + messages
    else:
        messages[0]["content"] = (messages[0].get("content") or "") + " " + system_content
    payload = {"model": cfg.llm.model, "temperature": cfg.llm.temperature if temperature is None else temperature, "messages": messages}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Unexpected response schema: {data}")
