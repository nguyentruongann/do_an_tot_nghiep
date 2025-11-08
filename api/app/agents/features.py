from __future__ import annotations
from typing import Dict, Any
import hashlib, json

def extract_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Giản lược: chỉ băm nội dung để tạo id, và trả lại một vài thống kê
    text = (payload.get("text") or "").strip()
    f = {
        "len": len(text),
        "has_numbers": any(ch.isdigit() for ch in text),
        "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
    }
    return f
