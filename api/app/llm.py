# app/llm.py
from __future__ import annotations
import requests
from typing import List, Dict
from .config import load_config

class LLMTimeout(Exception):
    pass

def chat_completion(messages: List[Dict[str, str]], temperature: float | None = None) -> str:
    cfg = load_config()
    url = cfg.llm.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.llm.api_key}"}
    payload = {
        "model": getattr(cfg.llm, "model", cfg.llm.model),
        "messages": messages,
        "temperature": temperature if temperature is not None else cfg.llm.temperature,
        "stream": False,
    }

    try:
        # timeout(connect=10s, read=30s) -> nếu quá sẽ raise Timeout
        r = requests.post(url, headers=headers, json=payload, timeout=(10, 30))
        r.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise LLMTimeout("timeout") from e

    data = r.json()
    # Không áp đặt chính sách, trả về nguyên văn content
    return data["choices"][0]["message"]["content"]
