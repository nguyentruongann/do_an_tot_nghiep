from __future__ import annotations
import io, sqlite3
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

def load_any(path_or_bytes, table: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Read CSV/XLSX/Parquet or SQLite (with table=...)."""
    if isinstance(path_or_bytes, (bytes, bytearray, io.BytesIO)):
        for reader, name in ((pd.read_csv,"uploaded.csv"), (pd.read_excel,"uploaded.xlsx")):
            try:
                df = reader(io.BytesIO(path_or_bytes))
                return df, name
            except Exception:
                pass
        raise ValueError("Không thể đọc dữ liệu từ buffer (CSV/Excel).")
    p = Path(str(path_or_bytes))
    if not p.exists():
        raise FileNotFoundError(f"Không thấy file: {p}")
    suf = p.suffix.lower()
    if suf in [".csv",".txt"]:
        return pd.read_csv(p), str(p)
    if suf in [".xlsx",".xls"]:
        return pd.read_excel(p), str(p)
    if suf in [".parquet"]:
        return pd.read_parquet(p), str(p)
    if suf in [".db",".sqlite"]:
        if not table: raise ValueError("Với SQLite, cung cấp 'table=' để đọc.")
        con = sqlite3.connect(p)
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", con)
        finally:
            con.close()
        return df, f"{p}::{table}"
    raise ValueError(f"Định dạng không hỗ trợ: {suf}")