from __future__ import annotations
import os, json, time
try:
    import redis  # type: ignore
except Exception:
    # Redis may not be installed in some environments.  When import fails,
    # logging is effectively disabled.
    redis = None  # type: ignore
from typing import Any, Dict, List, Optional
"""
Utility functions for working with Redis.  Updated to use absolute imports
for ``load_config`` so that this module can be imported directly.
"""

from app.config import load_config

SAVE_LOGS = os.environ.get("SAVE_LOGS","true").lower()=="true"
REDIS_URL = os.environ.get("REDIS_URL","redis://redis:6379/0")

_client: Optional[redis.Redis] = None
def client() -> Optional[redis.Redis]:
    """
    Lazily create and return a Redis client.  If the Redis server cannot be
    reached, return ``None`` so that callers can handle logging failures
    gracefully.  Connection errors are suppressed to avoid crashing the
    application in environments without a Redis instance.
    """
    global _client
    if _client is not None:
        return _client
    try:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        # Test the connection once
        _client.ping()
        return _client
    except Exception:
        # Disable logging if Redis is not available
        return None

def set_json(key: str, value: Dict[str, Any], ttl: int | None = None):
    ttl = ttl or load_config().storage.ttl_seconds
    c = client()
    if c is None:
        return
    try:
        c.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
    except Exception:
        pass

def get_json(key: str):
    c = client()
    if c is None:
        return None
    try:
        s = c.get(key)
    except Exception:
        return None
    return None if s is None else json.loads(s)

def log_event(session_id: str, kind: str, payload: Dict[str, Any]):
    if not SAVE_LOGS:
        return
    c = client()
    if c is None:
        return
    key = f"{load_config().storage.logs_key_prefix}:{session_id}"
    entry = {"ts": int(time.time()), "kind": kind, **payload}
    try:
        c.rpush(key, json.dumps(entry, ensure_ascii=False))
        c.expire(key, load_config().storage.ttl_seconds)
    except Exception:
        pass

def read_logs(session_id: str, last_n: int = 200) -> List[Dict[str, Any]]:
    c = client()
    if c is None:
        return []
    key = f"{load_config().storage.logs_key_prefix}:{session_id}"
    try:
        items = c.lrange(key, -last_n, -1)
    except Exception:
        return []
    return [json.loads(x) for x in items]
