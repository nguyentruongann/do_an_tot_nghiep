from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

# BASE_DIR = .../api
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

# === QUAN TRỌNG ===
# Nếu bạn MUỐN LLM thấy toàn bộ CSV, đặt = None.
# Nếu muốn an toàn hơn, đặt = số dòng tối đa (vd 500).
MAX_ROWS_FOR_LLM: Optional[int] = None  # None = không giới hạn rows


def load_session_scored_df(session_id: str) -> pd.DataFrame:
    """
    Đọc file CSV đã chấm điểm cho MỘT session cụ thể:
    data/sessions/{session_id}_scored.csv
    """
    path = SESSIONS_DIR / f"{session_id}_scored.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file đã chấm điểm cho session: {path}")
    return pd.read_csv(path)


def _safe_value_counts(series: pd.Series, top_n: int = 5) -> Dict[str, int]:
    vc = series.fillna("(trống)").astype(str).value_counts()
    if top_n is not None:
        vc = vc.head(top_n)
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def compute_basic_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summary cơ bản:
    - tổng số lead
    - phân bố theo qualification_status
    - điểm trung bình
    - tỉ lệ VIP
    - top 5 source, top 5 city
    """
    if df.empty:
        return {
            "total_leads": 0,
            "by_status": {},
            "avg_score": 0.0,
            "vip_ratio": 0.0,
            "by_source_top5": {},
            "by_city_top5": {},
        }

    total = len(df)

    if "qualification_status" in df.columns:
        by_status = _safe_value_counts(df["qualification_status"], top_n=None)
    else:
        by_status = {}

    if "score" in df.columns:
        avg_score = float(df["score"].mean())
    else:
        avg_score = 0.0

    vip_ratio = 0.0
    if "is_vip" in df.columns:
        vip_count = df[df["is_vip"] == 1].shape[0]
        vip_ratio = float(vip_count) / float(total) if total else 0.0

    by_source = (
        _safe_value_counts(df["source"], top_n=5) if "source" in df.columns else {}
    )
    by_city = (
        _safe_value_counts(df["city"], top_n=5) if "city" in df.columns else {}
    )

    return {
        "total_leads": int(total),
        "by_status": by_status,
        "avg_score": avg_score,
        "vip_ratio": vip_ratio,
        "by_source_top5": by_source,
        "by_city_top5": by_city,
    }


# ====== Chuyển từng lead thành object cho LLM ======

def _select_lead_columns(df: pd.DataFrame) -> List[str]:
    """
    Chọn những cột quan trọng để trả về cho LLM.
    Không đưa cột 'explain' để JSON đỡ nặng.
    Bạn có thể chỉnh thêm/bớt tuỳ schema.
    """
    preferred = [
        "name",
        "score",
        "qualification_status",
        "source",
        "status",
        "is_vip",
        "no_of_employees",
        "city",
    ]
    return [c for c in preferred if c in df.columns]


def _rows_to_list(df: pd.DataFrame, limit: Optional[int]) -> List[Dict[str, Any]]:
    """
    Convert một DataFrame -> list[dict] để nhét vào JSON cho LLM.
    - limit = None => lấy toàn bộ rows
    - limit = n     => lấy tối đa n rows
    """
    cols = _select_lead_columns(df)
    if not cols:
        return []

    if limit is not None:
        df = df.head(limit)

    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        item: Dict[str, Any] = {"row_index": int(idx)}
        for col in cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, pd.Timestamp):
                item[col] = val.isoformat()
            else:
                item[col] = val.item() if hasattr(val, "item") else val
        records.append(item)

    return records


def compute_session_summary(session_id: str) -> Dict[str, Any]:
    """
    Tính summary CHỈ cho session hiện tại.

    Trả về:
    {
      "session_id": ...,
      "session_summary": { ... },
      "rows_for_llm": [ {row_index, name, score, ...}, ... ],
      "top_leads": [...],         # top theo score (subset của rows_for_llm)
      "leads_by_status": {...}    # group theo qualification_status
    }
    """
    df = load_session_scored_df(session_id)
    summary = compute_basic_summary(df)

    # Sắp xếp theo score để lấy top lead
    if "score" in df.columns:
        df_sorted = df.sort_values(by="score", ascending=False)
    else:
        df_sorted = df

    # Toàn bộ rows (hoặc giới hạn theo MAX_ROWS_FOR_LLM)
    rows_for_llm = _rows_to_list(df_sorted, limit=MAX_ROWS_FOR_LLM)

    # Top 20 lead toàn session (dùng nhanh cho câu hỏi "top lead")
    top_leads = _rows_to_list(df_sorted, limit=20)

    # Lead theo từng qualification_status (mỗi status tối đa 30 lead)
    leads_by_status: Dict[str, List[Dict[str, Any]]] = {}
    if "qualification_status" in df.columns:
        grouped = df_sorted.groupby("qualification_status")
        for status, sub in grouped:
            leads_by_status[str(status)] = _rows_to_list(
                sub.sort_values(by="score", ascending=False)
                if "score" in sub.columns
                else sub,
                limit=30,
            )

    return {
        "session_id": session_id,
        "session_summary": summary,
        "rows_for_llm": rows_for_llm,
        "top_leads": top_leads,
        "leads_by_status": leads_by_status,
    }
