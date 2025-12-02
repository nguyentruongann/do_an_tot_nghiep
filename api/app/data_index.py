from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd, numpy as np, re, os, json, pathlib
"""
Data indexing utilities.  This module loads base, session and global CSV
layers and provides simple keyword retrieval over their text columns.  The
module has been updated to use absolute imports so it can run as a
standalone module.
"""

from app.config import load_config


def _ensure_dir(p: str): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

class DataLayer:
    def __init__(self, csv_path: Optional[str], name: str):
        self.name = name
        self.path = csv_path
        self.df = pd.DataFrame()
        self.text_col = "text"
        self.lower: List[str] = []
        if csv_path and os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path, low_memory=False).fillna("")
            except Exception:
                self.df = pd.DataFrame()
        if not self.df.empty:
            self.text_col = self._pick_text_col()
            self.lower = self.df[self.text_col].astype(str).str.lower().tolist()

    def _pick_text_col(self) -> str:
        cands = [c for c in self.df.columns if c.lower() in {"text","notes","description","memo"}]
        if cands: return cands[0]
        if len(self.df.columns)==0: return "text"
        lens = {c:self.df[c].astype(str).str.len().mean() for c in self.df.columns}
        return max(lens, key=lens.get)

    def retrieve(self, query: str, k: int = 3):
        if self.df.empty: return ([], False)
        q = re.findall(r"\w+", (query or "").lower())
        scores = []
        for i, t in enumerate(self.lower):
            s = sum(t.count(w) for w in q) if q else 0
            scores.append((s,i))
        scores.sort(reverse=True)
        picked = [idx for s,idx in scores[:k] if s>0] or [i for _,i in scores[:k]]
        rows = self.df.iloc[picked].to_dict(orient="records")
        for j, idx in enumerate(picked): rows[j]["_row_index"] = int(idx)
        grounded = any(s>0 for s,_ in scores[:k])
        return rows, grounded

class DataIndex:
    def __init__(self):
        cfg = load_config()
        self.base = DataLayer(cfg.training.data_path, "base")
        self.sessions_dir = getattr(cfg.training, "sessions_dir", "/app/data/sessions")
        self.global_dir = getattr(cfg.training, "global_dir", "/app/data/global")
        _ensure_dir(self.sessions_dir); _ensure_dir(self.global_dir)
        self.global_path = os.path.join(self.global_dir, "global_scored.csv")
        self.global_layer = DataLayer(self.global_path if os.path.exists(self.global_path) else None, "global")
        self.session_layers: dict[str, DataLayer] = {}

    def session_csv_path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, f"{session_id}_scored.csv")

    def load_session_layer(self, session_id: str):
        p = self.session_csv_path(session_id)
        if os.path.exists(p):
            layer = DataLayer(p, f"session:{session_id}")
            self.session_layers[session_id] = layer
            return layer
        return None

    def get_layers(self, session_id: str) -> List[DataLayer]:
        layers: List[DataLayer] = []
        # session
        if session_id in self.session_layers: layers.append(self.session_layers[session_id])
        else:
            sl = self.load_session_layer(session_id)
            if sl: layers.append(sl)
        # global
        if os.path.exists(self.global_path):
            self.global_layer = DataLayer(self.global_path, "global")
            layers.append(self.global_layer)
        # base
        layers.append(self.base)
        return layers

    def retrieve(self, session_id: str, query: str, k_each: int = 3, k_total: int = 6):
        out: List[Dict[str,Any]] = []; grounded = False
        for layer in self.get_layers(session_id):
            rows, g = layer.retrieve(query, k_each)
            out.extend([{"_layer": layer.name, **r} for r in rows])
            grounded = grounded or g
            if len(out) >= k_total: break
        return out[:k_total], grounded

_index: Optional[DataIndex] = None
def get_index() -> DataIndex:
    global _index
    if _index is None:
        _index = DataIndex()
    return _index
