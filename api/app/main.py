# # app/main.py
# from __future__ import annotations

# import os
# import io
# import json
# import uuid
# import shutil
# import traceback
# from typing import Optional, Dict, Any

# import pandas as pd
# from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, StreamingResponse
# from pydantic import BaseModel

# # modules có sẵn trong source của bạn
# from .agents import run_scoring_pipeline
# from .llm import chat_completion
# from .config import load_config
# from .redis_store import set_json, get_json, log_event as redis_log_event

# # -------------------------------------------------------------------
# # App & CORS
# # -------------------------------------------------------------------
# app = FastAPI(title="Scoring API (Session Private)", version="1.0.0")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # UI chạy khác cổng
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------------------------------------------------
# # Helpers
# # -------------------------------------------------------------------
# DATA_ROOT = "/app/data"
# SESS_ROOT = os.path.join(DATA_ROOT, "sessions")
# os.makedirs(SESS_ROOT, exist_ok=True)

# def _sid_dir(sid: str) -> str:
#     d = os.path.join(SESS_ROOT, sid)
#     os.makedirs(d, exist_ok=True)
#     return d

# def _safe_read_table(path: str) -> pd.DataFrame:
#     """Đọc CSV/XLS/XLSX/Parquet; auto-detect delimiter; fillna hợp lý."""
#     ext = os.path.splitext(path)[1].lower()
#     if ext in [".xls", ".xlsx"]:
#         df = pd.read_excel(path)
#     elif ext in [".parquet"]:
#         df = pd.read_parquet(path)
#     else:
#         try:
#             df = pd.read_csv(path)
#         except Exception:
#             # auto delimiter
#             df = pd.read_csv(path, sep=None, engine="python")
#     return df

# def _get_sess(sid: str) -> Dict[str, Any]:
#     jpath = os.path.join(_sid_dir(sid), "session.json")
#     if os.path.exists(jpath):
#         with open(jpath, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {"sid": sid}

# def _save_sess(sid: str, data: Dict[str, Any]) -> None:
#     jpath = os.path.join(_sid_dir(sid), "session.json")
#     with open(jpath, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

# def safe_log_event(event: str, payload: Optional[Dict[str, Any]] = None, sid: Optional[str] = None) -> None:
#     """Bọc logger Redis: không làm gãy flow nếu Redis lỗi; hợp nhất sid vào payload."""
#     try:
#         payload = payload or {}
#         if sid:
#             payload = {"sid": sid, **payload}
#         redis_log_event(event, payload)
#     except Exception:
#         # im lặng – tránh crash
#         pass

# def _guard_message(text: str) -> bool:
#     """Chỉ cho phép chào hỏi & câu hỏi liên quan chấm điểm/khách hàng tiềm năng."""
#     t = text.strip().lower()
#     if any(k in t for k in ["xin chào", "chào", "alo", "hello", "hi"]):
#         return True
#     allow_kw = [
#         "chấm điểm", "scoring", "khách hàng tiềm năng", "tiềm năng",
#         "đặc trưng", "roc", "auc", "ngưỡng", "giải thích", "thuộc tính",
#         "vì sao", "vì sao điểm", "ai là khách hàng", "khách nào",
#         "kết quả chấm", "tải csv", "thống kê", "review", "đánh giá mô hình",
#     ]
#     if any(k in t for k in allow_kw):
#         return True
#     return False

# # -------------------------------------------------------------------
# # Schemas
# # -------------------------------------------------------------------
# class ChatRequest(BaseModel):
#     message: str
#     sid: Optional[str] = None

# # -------------------------------------------------------------------
# # Endpoints
# # -------------------------------------------------------------------
# @app.get("/health")
# def health():
#     return {"ok": True, "service": "scoring-api"}

# @app.post("/session/new")
# def new_session():
#     sid = f"{uuid.uuid4().hex[:6]}-{uuid.uuid4().hex[:6]}"
#     _sid_dir(sid)
#     _save_sess(sid, {"sid": sid})
#     safe_log_event("session_new", {"sid": sid}, sid=sid)
#     return {"sid": sid}

# @app.post("/session/clear")
# def clear_session(sid: str = Query(...)):
#     sdir = _sid_dir(sid)
#     try:
#         shutil.rmtree(sdir, ignore_errors=True)
#         os.makedirs(sdir, exist_ok=True)
#         _save_sess(sid, {"sid": sid})
#         safe_log_event("session_cleared", {"sid": sid}, sid=sid)
#         return {"sid": sid, "cleared": True}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# @app.post("/upload_session")
# async def upload_session(
#     file: UploadFile = File(...),
#     sid: Optional[str] = Form(None),                 # nếu UI gửi kèm vào FormData
#     sid_q: Optional[str] = Query(None, alias="sid")  # hoặc gửi qua query ?sid=
# ):
#     sid_final = sid or sid_q
#     if not sid_final:
#         raise HTTPException(status_code=400, detail="Missing sid")

#     sdir = _sid_dir(sid_final)
#     ext = os.path.splitext(file.filename)[1].lower()
#     raw_path = os.path.join(sdir, f"uploaded{ext or '.csv'}")

#     blob = await file.read()
#     with open(raw_path, "wb") as f:
#         f.write(blob)

#     df = _safe_read_table(raw_path)
#     sess = _get_sess(sid_final)
#     sess["uploaded_path"] = raw_path
#     sess["csv_preview"] = df.head(12).to_dict(orient="records")
#     sess.pop("scored_csv", None)
#     sess.pop("score_summary", None)
#     _save_sess(sid_final, sess)

#     safe_log_event("upload_ok", {"rows": int(len(df)), "cols": len(df.columns)}, sid=sid_final)
#     return {"sid": sid_final, "rows": int(len(df)), "cols": len(df.columns), "sample_head_12": sess["csv_preview"]}

# @app.post("/score_session")
# def score_session(sid: str = Query(...)):
#     sess = _get_sess(sid)
#     path = sess.get("uploaded_path")
#     if not path or not os.path.exists(path):
#         return JSONResponse(status_code=400, content={"error": "Chưa có dữ liệu trong phiên"})

#     try:
#         df = _safe_read_table(path)
#         sdir = _sid_dir(sid)
#         result = run_scoring_pipeline(
#             df=df,
#             session_dir=sdir,
#             persist_model=False,
#             sqlite_path=None,
#         )
#         # lưu summary vào session cho chat
#         sess["score_summary"] = {
#             "rows": result.get("rows"),
#             "metrics": result.get("metrics"),
#             "scoring_column": result.get("scoring_column"),
#             "feature_importance": result.get("feature_importance"),
#         }
#         sess["scored_csv"] = result.get("scored_csv")
#         _save_sess(sid, sess)

#         safe_log_event("score_done", {"rows": sess["score_summary"]["rows"]}, sid=sid)
#         return result
#     except Exception as e:
#         safe_log_event("score_failed", {"error": str(e)}, sid=sid)
#         return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

# @app.get("/download_scored_session")
# def download_scored_session(sid: str = Query(...)):
#     sess = _get_sess(sid)
#     csv_path = sess.get("scored_csv")
#     if not csv_path or not os.path.exists(csv_path):
#         return JSONResponse(status_code=404, content={"error": "Chưa có file kết quả"})
#     def _iter():
#         with open(csv_path, "rb") as f:
#             yield from f
#     filename = os.path.basename(csv_path)
#     return StreamingResponse(_iter(), media_type="text/csv",
#                              headers={"Content-Disposition": f'attachment; filename="{filename}"'})

# @app.post("/chat")
# def chat(req: ChatRequest):
#     text = (req.message or "").strip()
#     sid = req.sid
#     if not text:
#         return {"intent": "chat", "text": "Vui lòng nhập câu hỏi."}

#     # Lọc bảo mật phạm vi
#     if not _guard_message(text):
#         return {"intent": "chat",
#                 "text": "Xin lỗi, mình chỉ hỗ trợ **chào hỏi** và **các câu hỏi liên quan chấm điểm/khách hàng tiềm năng**. \
# Vui lòng tải dữ liệu và hỏi trong phạm vi này nhé!"}

#     # Nếu có sid, thử đọc summary
#     summary = None
#     if sid:
#         sess = _get_sess(sid)
#         summary = sess.get("score_summary")

#     # Chưa có dữ liệu
#     if not summary and "chấm điểm" in text.lower():
#         return {"intent": "chat", "text": "Hiện chưa có dữ liệu trong phiên. Hãy upload CSV/XLSX rồi bấm **Chấm điểm** nhé."}

#     # LLM prompt (tiếng Việt, phạm vi rõ ràng)
#     sys_prompt = (
#         "Bạn là trợ lý scoring tiếng Việt. Chỉ trả lời về việc **chấm điểm khách hàng tiềm năng**: "
#         "quy trình, đặc trưng, ROC/AUC, cách giải thích điểm, cách tải CSV đã chấm, v.v.\n"
#         "Tuyệt đối không trả lời về dữ liệu/bảo mật/hệ thống/DB của người khác.\n"
#         "Nếu người dùng hỏi ngoài phạm vi, hãy từ chối lịch sự.\n"
#     )
#     user_msg = text
#     if summary:
#         ctx = json.dumps(summary, ensure_ascii=False)
#         user_msg = f"Ngữ cảnh kết quả phiên: {ctx}\nCâu hỏi: {text}"

#     try:
#         ans = chat_completion(
#             [{"role": "system", "content": sys_prompt},
#              {"role": "user", "content": user_msg}],
#             temperature=getattr(load_config().llm, "temperature", 0.0),
#         )
#         safe_log_event("chat_ok", {"len": len(ans)}, sid=sid)
#         return {"intent": "chat", "text": ans, "payload": None}
#     except Exception as e:
#         safe_log_event("chat_llm_fail", {"error": str(e)}, sid=sid)
#         # fallback trả lời ngắn
#         return {"intent": "chat",
#                 "text": "Không gọi được LLM. Bạn có thể bấm **Chấm điểm** rồi hỏi các câu như: "
#                         "“Các đặc trưng quan trọng nhất?”, “AUC là bao nhiêu?”, "
#                         "hoặc “Tải CSV đã chấm” để tải kết quả.",
#                 "payload": None}

# @app.get("/logs")
# def get_logs():
#     # nếu bạn đã có redis_store.get_json thì dùng, ở đây trả stub
#     return {"ok": True}
# app/main.py
from __future__ import annotations

import os
import json
import uuid
import shutil
import traceback
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .agents import run_scoring_pipeline
from .llm import chat_completion
from .config import load_config
from .redis_store import log_event as redis_log_event

# ================================================================
# App & CORS
# ================================================================
app = FastAPI(title="Scoring API (Session Private)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Constants & Helpers
# ================================================================
DATA_ROOT = "/app/data"
SESS_ROOT = os.path.join(DATA_ROOT, "sessions")
os.makedirs(SESS_ROOT, exist_ok=True)

SCORING_COL_DEFAULT = "_potential_proba"
ID_CANDIDATES = ["ID", "Id", "id", "customer_id", "customerId", "client_id"]

def _sid_dir(sid: str) -> str:
    d = os.path.join(SESS_ROOT, sid)
    os.makedirs(d, exist_ok=True)
    return d

def _log(event: str, payload: Optional[Dict[str, Any]] = None, sid: Optional[str] = None) -> None:
    try:
        payload = payload or {}
        if sid:
            payload = {"sid": sid, **payload}
        redis_log_event(event, payload)
    except Exception:
        pass

def _new_sid() -> str:
    return f"{uuid.uuid4().hex[:6]}-{uuid.uuid4().hex[:6]}"

def _safe_read_table(path: str) -> pd.DataFrame:
    """Đọc CSV/XLS/XLSX/Parquet; auto-detect delimiter khi cần."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    elif ext in [".parquet"]:
        df = pd.read_parquet(path)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")
    return df

def _get_sess(sid: str) -> Dict[str, Any]:
    jpath = os.path.join(_sid_dir(sid), "session.json")
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sid": sid}

def _save_sess(sid: str, data: Dict[str, Any]) -> None:
    jpath = os.path.join(_sid_dir(sid), "session.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _is_timeout_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("timeout" in s) or ("timed out" in s) or ("read timed out" in s)

def _best_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _df_snapshot(df: pd.DataFrame, note: str = "", head_rows: int = 10) -> str:
    """Sinh chuỗi mô tả dữ liệu gọn cho LLM."""
    try:
        num_desc = df.select_dtypes("number").describe().round(3).to_string()
    except Exception:
        num_desc = "(không có cột số để thống kê)"
    head_csv = df.head(head_rows).to_csv(index=False)
    cols = ", ".join(map(str, df.columns.tolist()))
    return (
        (note + "\n" if note else "") +
        f"Số dòng: {len(df)}, số cột: {len(df.columns)}\n"
        f"Các cột: {cols}\n"
        f"Thống kê cột số:\n{num_desc}\n"
        f"{head_rows} dòng đầu (CSV):\n{head_csv}"
    )

def _load_session_dfs(sid: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    """Trả về (scored_df, uploaded_df, session-json)."""
    if not sid:
        return None, None, {"sid": None}
    sess = _get_sess(sid)
    scored_df = None
    uploaded_df = None
    try:
        spath = sess.get("scored_csv")
        if spath and os.path.exists(spath):
            scored_df = _safe_read_table(spath)
    except Exception:
        scored_df = None
    try:
        upath = sess.get("uploaded_path")
        if upath and os.path.exists(upath):
            uploaded_df = _safe_read_table(upath)
    except Exception:
        uploaded_df = None
    return scored_df, uploaded_df, sess

# ================================================================
# Schemas
# ================================================================
class ChatRequest(BaseModel):
    message: str
    sid: Optional[str] = None

# ================================================================
# Endpoints
# ================================================================
@app.get("/health")
def health():
    return {"ok": True, "service": "scoring-api"}

@app.post("/session/new")
def new_session():
    sid = _new_sid()
    _sid_dir(sid)
    _save_sess(sid, {"sid": sid})
    _log("session_new", sid=sid)
    return {"sid": sid}

@app.post("/session/clear")
def clear_session(sid: str = Query(...)):
    sdir = _sid_dir(sid)
    try:
        shutil.rmtree(sdir, ignore_errors=True)
        os.makedirs(sdir, exist_ok=True)
        _save_sess(sid, {"sid": sid})
        _log("session_cleared", sid=sid)
        return {"sid": sid, "cleared": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload_session")
async def upload_session(
    file: UploadFile = File(...),
    sid: Optional[str] = Form(None),
    sid_q: Optional[str] = Query(None, alias="sid"),
):
    sid_final = sid or sid_q
    if not sid_final:
        raise HTTPException(status_code=400, detail="Missing sid")

    sdir = _sid_dir(sid_final)
    ext = os.path.splitext(file.filename or "")[1].lower()
    raw_path = os.path.join(sdir, f"uploaded{ext or '.csv'}")

    blob = await file.read()
    with open(raw_path, "wb") as f:
        f.write(blob)

    df = _safe_read_table(raw_path)
    sess = _get_sess(sid_final)
    sess["uploaded_path"] = raw_path
    sess["csv_preview"] = df.head(12).to_dict(orient="records")
    # reset kết quả cũ nếu có
    sess.pop("scored_csv", None)
    sess.pop("score_summary", None)
    _save_sess(sid_final, sess)

    _log("upload_ok", {"rows": int(len(df)), "cols": len(df.columns)}, sid=sid_final)
    return {
        "sid": sid_final,
        "rows": int(len(df)),
        "cols": len(df.columns),
        "sample_head_12": sess["csv_preview"],
        "filename": os.path.basename(raw_path)
    }

@app.post("/score_session")
def score_session(sid: str = Query(...)):
    """
    Huấn luyện/chấm điểm và **lưu file đã chấm** vào session,
    đồng thời lưu **summary** để chat dùng làm ngữ cảnh.
    """
    sess = _get_sess(sid)
    path = sess.get("uploaded_path")
    if not path or not os.path.exists(path):
        return JSONResponse(status_code=400, content={"error": "Chưa có dữ liệu trong phiên"})

    try:
        df = _safe_read_table(path)
        sdir = _sid_dir(sid)

        result = run_scoring_pipeline(
            df=df,
            session_dir=sdir,
            persist_model=False,
            sqlite_path=None,
        )

        # Lưu thông tin vào session
        sess["score_summary"] = {
            "rows": result.get("rows"),
            "metrics": result.get("metrics"),
            "scoring_column": result.get("scoring_column"),
            "feature_importance": result.get("feature_importance"),
        }
        sess["scored_csv"] = result.get("scored_csv")

        # Tạo preview top 20 theo điểm (nếu có) để hỗ trợ LLM
        try:
            scoring_col = sess["score_summary"].get("scoring_column") or SCORING_COL_DEFAULT
            if sess.get("scored_csv") and os.path.exists(sess["scored_csv"]):
                sdf = _safe_read_table(sess["scored_csv"])
                if scoring_col in sdf.columns:
                    id_col = _best_id_col(sdf)
                    cols = [scoring_col] + ([id_col] if id_col else [])
                    top_preview = sdf.sort_values(scoring_col, ascending=False)[cols].head(20)
                    sess["scored_top_preview"] = top_preview.to_dict(orient="records")
        except Exception:
            sess["scored_top_preview"] = None

        _save_sess(sid, sess)

        _log("score_done", {"rows": sess["score_summary"]["rows"]}, sid=sid)
        return result
    except Exception as e:
        _log("score_failed", {"error": str(e)}, sid=sid)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

@app.get("/download_scored_session")
def download_scored_session(sid: str = Query(...)):
    sess = _get_sess(sid)
    csv_path = sess.get("scored_csv")
    if not csv_path or not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": "Chưa có file kết quả"})

    def _iter():
        with open(csv_path, "rb") as f:
            yield from f

    filename = os.path.basename(csv_path)
    return StreamingResponse(
        _iter(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

# ================================================================
# Chat
# ================================================================
@app.post("/chat")
def chat(req: ChatRequest):
    """
    - Không chặn loại câu hỏi nào.
    - Luôn ưu tiên **CSV đã chấm** (nếu có) làm ngữ cảnh cho LLM để trả lời.
    - LLM chỉ được phép dùng dữ liệu của **phiên hiện tại** (được nêu trong system prompt).
    - Nếu LLM timeout -> trả thông điệp chuẩn.
    """
    text = (req.message or "").strip()
    sid = req.sid
    if not text:
        return {"intent": "chat", "text": "Vui lòng nhập câu hỏi.", "payload": None}

    scored_df, uploaded_df, sess = _load_session_dfs(sid)
    summary = sess.get("score_summary") if sid else None
    scoring_col = (summary or {}).get("scoring_column") or SCORING_COL_DEFAULT

    # Chuẩn bị system prompt + ngữ cảnh dữ liệu phiên hiện tại
    sys = (
        "Bạn là trợ lý dữ liệu tiếng Việt. Hãy trả lời thẳng câu hỏi của người dùng.\n"
        "Chỉ sử dụng và suy luận từ **ngữ cảnh của phiên hiện tại** được cung cấp phía dưới "
        "(không sử dụng dữ liệu của phiên khác).\n"
        "Nếu có file đã chấm thì ưu tiên dùng nó (có cột điểm dự đoán).\n"
    )
    msgs: List[Dict[str, str]] = [{"role": "system", "content": sys}]

    # Ưu tiên CSV đã chấm
    if scored_df is not None:
        note = f"(CSV đã chấm; nếu có, cột điểm là `{scoring_col}`)"
        msgs.append({"role": "system", "content": _df_snapshot(scored_df, note=note, head_rows=10)})
        if summary is not None:
            msgs.append({"role": "system", "content": f"Tóm tắt kết quả chấm: {json.dumps(summary, ensure_ascii=False)}"})
        if sess.get("scored_top_preview"):
            msgs.append({"role": "system", "content": f"Preview top theo điểm (tối đa 20 bản ghi): {json.dumps(sess['scored_top_preview'], ensure_ascii=False)}"})
    # Nếu chưa có file đã chấm thì đưa CSV gốc
    elif uploaded_df is not None:
        msgs.append({"role": "system", "content": _df_snapshot(uploaded_df, note="(CSV gốc — chưa chấm)", head_rows=10)})
    else:
        msgs.append({"role": "system", "content": "Phiên hiện tại chưa có dữ liệu (không có CSV được tải lên)."})

    msgs.append({"role": "user", "content": text})

    try:
        ans = chat_completion(
            msgs,
            temperature=getattr(load_config().llm, "temperature", 0.0),
        )
        _log("chat_ok", {"len": len(ans)}, sid=sid)
        return {"intent": "chat", "text": ans, "payload": None}
    except Exception as e:
        _log("chat_llm_fail", {"error": str(e)}, sid=sid)
        if _is_timeout_error(e):
            return {"intent": "chat", "text": "Xin lỗi, tôi không thể trả lời lúc này (LLM bị timeout).", "payload": None}
        return {"intent": "chat", "text": "Xin lỗi, tôi không thể trả lời lúc này.", "payload": None}

@app.get("/logs")
def get_logs():
    return {"ok": True}
