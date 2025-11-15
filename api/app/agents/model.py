from __future__ import annotations
from typing import Dict, Any, List
import hashlib, json, re, os
import pandas as pd
from ..redis_store import get_json, set_json
from ..config import load_config
from ..llm import chat_completion
from ..data_index import get_index

ALIASES = {
  "order_count": ["orders", "num_orders", "so_don", "don_hang"],
  "revenue": ["doanh_thu", "amount", "gmv", "turnover"],
  "notes": ["ghi_chu", "mota", "description", "text"]
}

def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    f = {k.lower(): v for k, v in (row or {}).items()}
    for canon, alist in ALIASES.items():
        if canon not in f:
            for a in alist:
                if a in f:
                    f[canon] = f.pop(a); break
    return f

def _row_text(f: Dict[str, Any]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in f.items())

def _ck(prefix: str, key_obj: Dict[str, Any]) -> str:
    h = hashlib.sha256(json.dumps(key_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"{load_config().storage.cache_key_prefix}:{prefix}:{h}"

def score_one_row(session_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    f = _normalize_row(row)
    query_text = _row_text(f)
    ck = _ck("llm_row", {"sid": session_id, "q": query_text})
    cached = get_json(ck)
    if cached is not None:
        return {"cached": True, **cached}

    idx = get_index()
    rows, grounded = idx.retrieve(session_id, query_text, k_each=3, k_total=6)
    system_extra = "Chỉ chấm điểm dựa trên CSV theo thứ tự: CSV session hiện tại, CSV tổng hợp, CSV gốc. Không bịa."
    user_prompt = (
        "NGỮ CẢNH CSV (JSON):\n" + json.dumps(rows, ensure_ascii=False) +
        "\n\nHÀNG CẦN CHẤM:\n" + json.dumps(f, ensure_ascii=False) +
        "\n\nTrả JSON: { \"score\": <0..100>, \"explain\": \"...\" }"
    )
    reply = chat_completion([{"role":"user","content":user_prompt}], temperature=0.1, system_extra=system_extra)
    try:
        m = re.search(r"\{.*\}", reply, flags=re.S)
        data = json.loads(m.group(0) if m else reply)
        score_val = int(max(0, min(100, int(data.get("score", 0)))))
        explain = str(data.get("explain", ""))
        result = {"score": score_val, "explain": explain, "grounded": grounded, "used_rows": rows[:3]}
    except Exception:
        result = {"score": 50, "explain": f"LLM trả về khó parse: {reply}", "grounded": grounded, "used_rows": rows[:3]}
    set_json(ck, result); return result

def _dedup_concat(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    def row_hash(row: pd.Series):
        d = row.drop(labels=[c for c in ["score","explain"] if c in row.index]).astype(str).to_dict()
        s = json.dumps(d, sort_keys=True, ensure_ascii=False)
        import hashlib as _h; return _h.sha256(s.encode("utf-8")).hexdigest()
    if existing is None or existing.empty:
        incoming["_hash"] = incoming.apply(row_hash, axis=1)
        return incoming.drop_duplicates("_hash").drop(columns=["_hash"])
    incoming["_hash"] = incoming.apply(row_hash, axis=1)
    existing["_hash"] = existing.apply(row_hash, axis=1)
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.drop_duplicates("_hash").drop(columns=["_hash"])
    return merged

def score_rows_and_save(session_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for r in rows:
        res = score_one_row(session_id, r)
        results.append({**r, "score": res["score"], "explain": res["explain"]})

    idx = get_index()
    os.makedirs(idx.sessions_dir, exist_ok=True); os.makedirs(idx.global_dir, exist_ok=True)
    session_path = idx.session_csv_path(session_id)
    df_new = pd.DataFrame(results)

    if os.path.exists(session_path):
        old = pd.read_csv(session_path, low_memory=False)
        df_session = _dedup_concat(old, df_new)
    else:
        df_session = df_new
    df_session.to_csv(session_path, index=False, encoding="utf-8")

    global_path = idx.global_path
    if os.path.exists(global_path):
        g_old = pd.read_csv(global_path, low_memory=False)
        g_new = _dedup_concat(g_old, df_new)
    else:
        g_new = df_new
    g_new.to_csv(global_path, index=False, encoding="utf-8")

    return {
        "saved_session_csv": session_path,
        "updated_global_csv": global_path,
        "rows_scored": len(df_new),
        "results_preview": results[:3],
    }
