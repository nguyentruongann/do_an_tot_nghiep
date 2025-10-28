from __future__ import annotations
import os, json, time, redis
from typing import Any, Dict, List

SAVE_LOGS = os.environ.get("SAVE_LOGS", "true").lower() == "true"
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
TTL_SECONDS = int(os.environ.get("REDIS_TTL", "259200"))

_client = None
def client():
    global _client
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _client

def set_json(key: str, value: Dict[str, Any]):
    client().set(key, json.dumps(value, ensure_ascii=False))

def get_json(key: str) -> Dict[str, Any] | None:
    v = client().get(key)
    return json.loads(v) if v else None

def log_event(session_id: str, kind: str, payload: Dict[str, Any]):
    if not SAVE_LOGS:
        print(f"[LOG disabled] {kind}: {payload}")
        return
    key = f"logs:{session_id}"
    entry = {"ts": int(time.time()), "kind": kind, **payload}
    c = client()
    c.rpush(key, json.dumps(entry, ensure_ascii=False))
    c.expire(key, TTL_SECONDS)
    # console mirror
    print(f"[LOG] {entry}")

def read_logs(session_id: str, last_n: int = 200) -> List[Dict[str, Any]]:
    items = client().lrange(f"logs:{session_id}", -last_n, -1)
    return [json.loads(x) for x in items]