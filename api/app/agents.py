from __future__ import annotations
import json, re, uuid, datetime as dt
from typing import Dict, Any, Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib, numpy as np, sqlite3, os

from .llm import chat_completion
from .config import load_config
from .redis_store import set_json, get_json, log_event

WAREHOUSE_MODELS_DIR = "/app/models"

def _guess_target(df: pd.DataFrame) -> str | None:
    priors = ["label","target","converted","is_customer","churn","y"]
    for c in priors:
        if c in df.columns and df[c].nunique() <= 5:
            return c
    for c in df.columns:
        if df[c].nunique()==2:
            return c
    return None

def _split_types(df: pd.DataFrame, target: str | None) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=["number","bool"]).columns.tolist()
    if target and target in num_cols: num_cols.remove(target)
    cat_cols = [c for c in df.columns if c not in num_cols + ([target] if target else [])]
    return num_cols, cat_cols

def agent1_build_prompt(df: pd.DataFrame) -> Dict[str, Any]:
    target = _guess_target(df)
    num_cols, cat_cols = _split_types(df, target)
    schema = {"columns": df.columns.tolist(), "numeric": num_cols, "categorical": cat_cols, "target": target}
    user_msg = ("Bạn là chuyên gia Marketing/Credit. Hãy tạo PROMPT chuẩn để chấm điểm 'khách hàng tiềm năng' "
                "trên dataset tabular. Trả JSON: {goal, features, target, method, explanation_style}. "
                f"Schema: {json.dumps(schema, ensure_ascii=False)}")
    try:
        content = chat_completion([{"role":"system","content":"Tạo prompt chuẩn hoá cho scoring khách hàng."},
                                   {"role":"user","content": user_msg}], temperature=0.0)
        m = re.search(r"\{.*\}", content, re.S)
        j = json.loads(m.group(0)) if m else json.loads(content)
        prompt = json.dumps(j, ensure_ascii=False)
    except Exception:
        j = {"goal": "Dự đoán xác suất khách hàng tiềm năng (1)",
             "features": num_cols + cat_cols,
             "target": target,
             "method": "Nếu có target -> logistic_regression; nếu không -> GMM 2 cụm",
             "explanation_style": "Top-5 đặc trưng + Next Best Action"}
        prompt = json.dumps(j, ensure_ascii=False)
    return {"prompt": prompt, "schema": schema, "target": target}