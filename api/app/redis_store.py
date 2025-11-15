from __future__ import annotations
import os, json, time, redis
from typing import Any, Dict, List, Optional
from .config import load_config

SAVE_LOGS = os.environ.get("SAVE_LOGS","true").lower()=="true"
REDIS_URL = os.environ.get("REDIS_URL","redis://redis:6379/0")

_client: Optional[redis.Redis] = None
def client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _client

def set_json(key: str, value: Dict[str, Any], ttl: int | None = None):
    ttl = ttl or load_config().storage.ttl_seconds
    client().set(key, json.dumps(value, ensure_ascii=False), ex=ttl)

def get_json(key: str):
    s = client().get(key)
    return None if s is None else json.loads(s)

def log_event(session_id: str, kind: str, payload: Dict[str, Any]):
    if not SAVE_LOGS: return
    key = f"{load_config().storage.logs_key_prefix}:{session_id}"
    entry = {"ts": int(time.time()), "kind": kind, **payload}
    c = client()
    c.rpush(key, json.dumps(entry, ensure_ascii=False))
    c.expire(key, load_config().storage.ttl_seconds)

def read_logs(session_id: str, last_n: int = 200) -> List[Dict[str, Any]]:
    key = f"{load_config().storage.logs_key_prefix}:{session_id}"
    items = client().lrange(key, -last_n, -1)
    return [json.loads(x) for x in items]
