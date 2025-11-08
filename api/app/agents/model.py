from __future__ import annotations
from typing import Dict, Any
import hashlib, json
from ..redis_store import get_json, set_json
from ..config import load_config

def _cache_key_for_score(features: Dict[str, Any]) -> str:
    h = hashlib.sha256(json.dumps(features, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{load_config().storage.cache_key_prefix}:score:{h}"

def score_from_features(features: Dict[str, Any]) -> Dict[str, Any]:
    # Cache check
    ck = _cache_key_for_score(features)
    cached = get_json(ck)
    if cached is not None:
        return {"cached": True, **cached}

    # Toy scoring logic
    base = features.get("len", 0)
    has_numbers = features.get("has_numbers", False)
    score = min(100, base // 5 + (10 if has_numbers else 0))

    result = {"score": int(score), "explain": f"Chiều dài {base}, số {'có' if has_numbers else 'không'}"}
    set_json(ck, result)
    return {"cached": False, **result}
