from __future__ import annotations
import requests
from typing import List, Dict
from .config import load_config

def chat_completion(messages: List[Dict[str,str]], temperature: float | None = None) -> str:
    cfg = load_config()
    url = cfg.llm.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.llm.api_key}"}
    payload = {"model": cfg.llm.model, "temperature": temperature if temperature is not None else cfg.llm.temperature, "messages": messages}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)