from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, Optional

try:
    import redis
except Exception:
    redis = None

# Kết nối Redis; fallback in-memory nếu không kết nối được
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

_client = None
_memory_kv: Dict[str, Dict[str, Any]] = {}  # fallback

def _ensure_client():
    global _client
    if _client is not None:
        return _client
    if redis is None:
        return None
    try:
        _client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            decode_responses=True, socket_timeout=2, socket_connect_timeout=2
        )
        # ping thử
        _client.ping()
        return _client
    except Exception:
        _client = None
        return None

def set_json(key: str, value: Any, ttl: Optional[int] = None):
    c = _ensure_client()
    payload = json.dumps(value, ensure_ascii=False)
    if c:
        c.set(key, payload, ex=ttl)
    else:
        _memory_kv[key] = {"v": value, "exp": time.time() + ttl if ttl else None}

def get_json(key: str) -> Optional[Any]:
    c = _ensure_client()
    if c:
        val = c.get(key)
        if not val:
            return None
        try:
            return json.loads(val)
        except Exception:
            return None
    # fallback
    rec = _memory_kv.get(key)
    if not rec:
        return None
    exp = rec.get("exp")
    if exp and time.time() > exp:
        _memory_kv.pop(key, None)
        return None
    return rec.get("v")

def log_event(event: str, payload: Optional[Dict[str, Any]] = None, sid: Optional[str] = None):
    """
    Ghi log ngắn gọn vào Redis list 'logs' (nếu có).
    Chữ ký này CHO PHÉP gọi chỉ với (event) hoặc (event, payload).
    """
    c = _ensure_client()
    entry = {
        "ts": time.time(),
        "event": event,
        "sid": sid,
        "payload": payload or {},
    }
    if c:
        try:
            c.lpush("logs", json.dumps(entry, ensure_ascii=False))
            c.ltrim("logs", 0, 999)  # giữ ~1000 bản ghi
        except Exception:
            pass
    else:
        # fallback: không làm gì (hoặc in ra stdout tuỳ bạn)
        pass
