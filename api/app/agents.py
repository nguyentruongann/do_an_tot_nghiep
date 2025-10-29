# agents.py
from __future__ import annotations

import json
import os
import re
import uuid
import sqlite3
import datetime as dt
from typing import Dict, Any, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .llm import chat_completion
from .config import load_config
from .redis_store import set_json, get_json, log_event

# ---------------------------------------------------------------------
# Cấu hình & hằng số
# ---------------------------------------------------------------------
RANDOM_STATE = 42
WAREHOUSE_MODELS_DIR = "/app/models"  # nơi lưu .joblib nếu persist
SCORING_COL = "_potential_proba"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _normalize_boolish(series: pd.Series) -> pd.Series:
    """Chuẩn hoá chuỗi boole/binary: yes/no, true/false, y/n, '1'/'0'."""
    if series.dtype == "O":
        lower = series.astype(str).str.strip().str.lower()
        mapping = {
            "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, "ok": 1,
            "no": 0, "n": 0, "false": 0, "f": 0, "0": 0
        }
        mapped = lower.map(mapping)
        # chỉ trả mapped nếu không quá nhiều NaN (tức mapping có ý nghĩa)
        if mapped.notna().mean() > 0.8:
            return mapped.astype("float")
    return series


def _guess_target(df: pd.DataFrame) -> Optional[str]:
    """
    Đoán cột target nhị phân: ưu tiên theo tên, sau đó theo số lượng giá trị 2.
    Hỗ trợ cột dạng string 'yes/no'.
    """
    priors = ["label", "target", "converted", "is_customer", "churn", "y", "default", "responded"]
    for c in df.columns:
        if c in priors:
            s = _normalize_boolish(df[c])
            nunique = s.nunique(dropna=True)
            if nunique == 2 or (nunique <= 5 and s.dtype != "O"):
                return c

    for c in df.columns:
        s = _normalize_boolish(df[c])
        if s.nunique(dropna=True) == 2:
            return c
    return None


def _split_types(df: pd.DataFrame, target: Optional[str]) -> Tuple[List[str], List[str]]:
    """Chia cột numeric/categorical, loại bỏ target."""
    # trước tiên thử chuẩn hoá các cột boolean string
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = _normalize_boolish(df2[c])

    num_cols = df2.select_dtypes(include=["number", "bool"]).columns.tolist()
    if target and target in num_cols:
        num_cols.remove(target)
    cat_cols = [c for c in df2.columns if c not in num_cols + ([target] if target else [])]
    return num_cols, cat_cols


def _safe_json_from_text(text: str) -> Dict[str, Any]:
    """
    Trích JSON an toàn từ câu trả lời LLM:
    - Nếu bao ngoài text → bắt { ... } đầu tiên.
    - Nếu vẫn lỗi → raise để caller fallback.
    """
    if not text:
        raise ValueError("Empty content")
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    return json.loads(text)


def _feature_importance_from_lr(
    lr: LogisticRegression,
    preprocessor: ColumnTransformer,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Tổng hợp tầm quan trọng feature theo |coef|, gộp lại về tên feature gốc
    (do OHE tạo nhiều cột).
    """
    # Lấy tên cột sau preprocessor
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    num_cols = preprocessor.transformers_[0][2]  # numeric feature names
    cat_cols = preprocessor.transformers_[1][2]  # categorical feature names

    ohe_out = list(ohe.get_feature_names_out(cat_cols))
    all_out = list(num_cols) + ohe_out

    # Hệ số của LR (shape: [1, n_features])
    coefs = np.abs(lr.coef_.ravel())
    imp_map: Dict[str, float] = {c: 0.0 for c in feature_names}

    # Map numeric 1-1
    for i, col in enumerate(num_cols):
        if col in imp_map:
            imp_map[col] += float(coefs[i])

    # Map OHE: gộp theo prefix trước '__'
    idx = len(num_cols)
    for col in ohe_out:
        # ohe name kiểu: "<original>_<value>" trong sklearn>=1.3 với get_feature_names_out
        # Để an toàn, tách theo '_' từ phải sang nếu cần.
        base = col.split("_", 1)[0]
        if base in imp_map:
            imp_map[base] += float(coefs[idx])
        idx += 1

    # Chuẩn hoá [0,1]
    total = sum(imp_map.values()) or 1.0
    for k in imp_map:
        imp_map[k] = round(imp_map[k] / total, 6)
    return dict(sorted(imp_map.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------
# Agent 1: sinh prompt phân tích schema
# ---------------------------------------------------------------------
def agent1_build_prompt(df: pd.DataFrame, lang: str = "vi") -> Dict[str, Any]:
    """
    Phân tích schema và sinh prompt chuẩn (JSON) cho tác vụ chấm điểm.
    Luôn ưu tiên tiếng Việt ở system prompt.
    """
    cfg = load_config()
    target = _guess_target(df)
    num_cols, cat_cols = _split_types(df, target)

    schema = {
        "columns": df.columns.tolist(),
        "numeric": num_cols,
        "categorical": cat_cols,
        "target": target,
        "rows": int(len(df)),
    }

    sys_vi = (
        "Bạn là chuyên gia Marketing/Credit. "
        "Nhiệm vụ: tạo PROMPT CHUẨN HOÁ (JSON) để chấm điểm 'khách hàng tiềm năng' "
        "cho dữ liệu bảng (tabular). Luôn trả lời bằng TIẾNG VIỆT."
    )
    user_vi = (
        "Hãy tạo JSON gồm các khoá: {goal, features, target, method, explanation_style}. "
        "Yêu cầu: 'features' chỉ gồm các cột có ích; 'target' là nhị phân (nếu đoán được). "
        "Nếu có target → method='logistic_regression'; nếu không → method='gmm_2_clusters'. "
        f"Schema dữ liệu: {json.dumps(schema, ensure_ascii=False)}"
    )

    try:
        content = chat_completion(
            [
                {"role": "system", "content": sys_vi},
                {"role": "user", "content": user_vi},
            ],
            temperature=cfg.llm.temperature if hasattr(cfg, "llm") else 0.0,
        )
        j = _safe_json_from_text(content)
        # đảm bảo khoá tối thiểu
        j.setdefault("goal", "Ước lượng xác suất khách hàng tiềm năng (1)")
        j.setdefault("features", num_cols + cat_cols)
        j.setdefault("target", target)
        j.setdefault("method", "logistic_regression" if target else "gmm_2_clusters")
        j.setdefault("explanation_style", "Top-5 đặc trưng + giải thích ngưỡng")
        prompt = json.dumps(j, ensure_ascii=False)
    except Exception as e:
        log_event("agent1_prompt_fallback", payload={"error": str(e)})

        j = {
            "goal": "Ước lượng xác suất khách hàng tiềm năng (1)",
            "features": num_cols + cat_cols,
            "target": target,
            "method": "logistic_regression" if target else "gmm_2_clusters",
            "explanation_style": "Top-5 đặc trưng + Next Best Action",
        }
        prompt = json.dumps(j, ensure_ascii=False)

    return {"prompt": prompt, "schema": schema, "target": target}


# ---------------------------------------------------------------------
# Agent 2: huấn luyện & chấm điểm
# ---------------------------------------------------------------------
def agent2_fit_and_score(
    df: pd.DataFrame,
    prompt_info: Dict[str, Any],
    session_dir: Optional[str] = None,
    persist_model: bool = False,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Huấn luyện theo prompt & chấm điểm.
    - Có target: LogisticRegression (balanced).
    - Không target: GMM 2 cụm → lấy proba cụm 'tiềm năng' theo heuristic.
    Trả về: dataframe scored (nếu cần), metrics, feature_importance, model_path, csv_path.
    """
    meta = json.loads(prompt_info["prompt"]) if isinstance(prompt_info["prompt"], str) else prompt_info["prompt"]
    features: List[str] = meta.get("features") or [c for c in df.columns if c != meta.get("target")]
    target = meta.get("target")

    # Cắt còn các cột cần thiết
    used_cols = [c for c in features if c in df.columns] + ([target] if target in df.columns else [])
    work = df[used_cols].copy()

    # Split types
    num_cols, cat_cols = _split_types(work, target)

    model = None
    proba = None
    metrics: Dict[str, Any] = {}
    feature_importance: Dict[str, float] = {}

    if target and target in work.columns:
        # Chuẩn hoá target binary
        y = _normalize_boolish(work[target])
        if y.isna().any():
            # nếu normalize lỗi → cast nhị phân mặc định qua factorize
            y = pd.Series(pd.factorize(work[target].astype(str))[0]).astype(float)
        X = work.drop(columns=[target])

        # Pipeline: impute + scale(OE) + LR
        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
        )

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=None if hasattr(LogisticRegression, "n_jobs") else None,
        )

        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

        # train/valid
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if y.nunique() == 2 else None
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict_proba(X_valid)[:, 1]
        auc = float(roc_auc_score(y_valid, preds)) if y_valid.nunique() == 2 else None
        metrics["auc"] = auc

        # full proba
        proba = pipe.predict_proba(X)[:, 1]
        model = pipe

        # importance
        lr: LogisticRegression = pipe.named_steps["clf"]
        pre: ColumnTransformer = pipe.named_steps["prep"]
        feature_importance = _feature_importance_from_lr(lr, pre, features)

    else:
        # Unsupervised — GMM 2 cụm trên dữ liệu đã mã hoá
        X = work.copy()
        # Simple encoding cho GMM: OHE + scale
        cat_encoded = []
        if cat_cols:
            X_cat = X[cat_cols].astype("category").apply(lambda s: s.cat.codes)
            cat_encoded = [c + "_code" for c in cat_cols]
            X_cat.columns = cat_encoded
        else:
            X_cat = pd.DataFrame(index=X.index)

        X_num = X[num_cols].copy() if num_cols else pd.DataFrame(index=X.index)
        X_all = pd.concat([X_num, X_cat], axis=1).fillna(0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all.values)

        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=RANDOM_STATE)
        gmm.fit(X_scaled)
        probs = gmm.predict_proba(X_scaled)  # shape (n,2)

        # Chọn cụm 'tiềm năng':
        # Ưu tiên biến 'balance' nếu có; nếu không dùng trung bình điểm chuẩn hoá.
        potential_idx = 1
        if "balance" in X_all.columns:
            means = gmm.means_ @ np.ones((X_scaled.shape[1], 1))
            # tạm: cụm có mean lớn hơn coi là 'tiềm năng'
            potential_idx = int(np.argmax(means))
        else:
            potential_idx = int(np.argmax(gmm.means_.sum(axis=1)))
        proba = probs[:, potential_idx]
        model = {"scaler": scaler, "gmm": gmm, "columns": list(X_all.columns)}
        metrics["method"] = "gmm_2_clusters"

        # importance tạm thời: độ đóng góp biến theo phương sai nghịch
        var = gmm.covariances_[potential_idx].diagonal()
        invvar = 1.0 / (var + 1e-9)
        names = list(X_all.columns)
        s = invvar / invvar.sum()
        feature_importance = {names[i]: float(s[i]) for i in np.argsort(s)[::-1]}
        # map lại code -> feature gốc
        for c in cat_cols:
            if c + "_code" in feature_importance:
                feature_importance[c] = feature_importance.pop(c + "_code")

    # Gắn proba vào dataframe
    out = df.copy()
    out[SCORING_COL] = proba

    # Lưu CSV trong session nếu có
    csv_path = None
    if session_dir:
        csv_path = os.path.join(session_dir, f"scored_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv")
        os.makedirs(session_dir, exist_ok=True)
        out.to_csv(csv_path, index=False)

    # Lưu model/metadata vào warehouse (nếu bật)
    model_path = None
    if persist_model:
        os.makedirs(WAREHOUSE_MODELS_DIR, exist_ok=True)
        model_id = f"{dt.datetime.utcnow().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        model_path = os.path.join(WAREHOUSE_MODELS_DIR, f"scorer_{model_id}.joblib")
        joblib.dump(model, model_path)

        try:
            cfg = load_config()
            sqlite_fp = sqlite_path or getattr(getattr(cfg, "storage", object()), "sqlite_path", None)
            if sqlite_fp:
                os.makedirs(os.path.dirname(sqlite_fp), exist_ok=True)
                conn = sqlite3.connect(sqlite_fp)
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        id TEXT PRIMARY KEY,
                        created_utc TEXT,
                        rows INT,
                        has_target INT,
                        auc REAL,
                        model_path TEXT,
                        csv_path TEXT,
                        features_json TEXT,
                        importance_json TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO runs(id, created_utc, rows, has_target, auc, model_path, csv_path, features_json, importance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        dt.datetime.utcnow().isoformat(),
                        int(len(df)),
                        1 if (target and target in df.columns) else 0,
                        float(metrics.get("auc")) if metrics.get("auc") is not None else None,
                        model_path,
                        csv_path,
                        json.dumps(features, ensure_ascii=False),
                        json.dumps(feature_importance, ensure_ascii=False),
                    ),
                )
                conn.commit()
                conn.close()
        except Exception as e:
            log_event("warehouse_write_error", payload={"error": str(e)})


    return {
        "rows": int(len(df)),
        "scored_csv": csv_path,
        "model_path": model_path,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "scoring_column": SCORING_COL,
    }


# ---------------------------------------------------------------------
# API-level tiện ích cho /score_session
# ---------------------------------------------------------------------
def run_scoring_pipeline(
    df: pd.DataFrame,
    session_dir: Optional[str] = None,
    persist_model: bool = False,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Thực thi full pipeline: Agent1 -> Agent2.
    """
    a1 = agent1_build_prompt(df, lang="vi")
    result = agent2_fit_and_score(
        df=df,
        prompt_info=a1,
        session_dir=session_dir,
        persist_model=persist_model,
        sqlite_path=sqlite_path,
    )
    result["agent1"] = a1
    return result
