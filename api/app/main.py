from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uuid, os, pandas as pd, json, shutil
from .ingest import load_any
from .agents import agent1_build_prompt
from .schemas import UploadResponse, ScoreRunResponse, ChatRequest, ChatResponse
from .redis_store import get_json, set_json, SAVE_LOGS, log_event, read_logs
from .llm import chat_completion

app = FastAPI(title="Scoring API (Session Private)", version="1.0.0")

# allow UI on another port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESS_DIR = "/app/data/sessions"
os.makedirs(SESS_DIR, exist_ok=True)

def _sid_from_req(req: Request) -> str | None:
    return req.headers.get("X-Session-ID") or req.cookies.get("sid")

@app.get("/health")
def health():
    return {"status":"ok","message":"API up","save_logs": SAVE_LOGS}

@app.post("/session/new")
def new_session(response: Response):
    sid = str(uuid.uuid4())[:12]
    os.makedirs(f"{SESS_DIR}/{sid}", exist_ok=True)
    response.set_cookie("sid", sid, httponly=True, samesite="lax")
    set_json(f"session:{sid}", {"created": True})
    return {"session_id": sid}

@app.post("/session/clear")
def clear_session(request: Request):
    sid = _sid_from_req(request)
    if not sid: raise HTTPException(status_code=400, detail="Thiếu Session ID")
    p = f"{SESS_DIR}/{sid}"
    if os.path.exists(p): shutil.rmtree(p, ignore_errors=True)
    try:
        from .redis_store import client
        r = client()
        for k in r.scan_iter(f"*:{sid}"):
            r.delete(k)
        r.delete(f"session:{sid}")
        r.delete(f"prompt:session:{sid}")
        r.delete(f"logs:{sid}")
    except Exception:
        pass
    return {"cleared": True}

@app.post("/upload_session", response_model=UploadResponse)
async def upload_session(request: Request, file: UploadFile = File(...)):
    sid = _sid_from_req(request)
    if not sid: raise HTTPException(status_code=400, detail="Thiếu Session ID (gọi /session/new trước).")
    content = await file.read()
    df, src = load_any(content)
    analysis = agent1_build_prompt(df)
    os.makedirs(f"{SESS_DIR}/{sid}", exist_ok=True)
    df.to_parquet(f"{SESS_DIR}/{sid}/data.parquet")
    set_json(f"prompt:session:{sid}", analysis)
    set_json(f"session:{sid}", {"src": src, "rows": len(df), "cols": df.columns.tolist()})
    log_event(sid, "upload_session", {"rows": len(df)})
    return {"dataset_id": sid, "filename": src, "rows": len(df), "cols": df.columns.tolist(), "prompt": analysis["prompt"], "target": analysis["target"]}

def _is_scoring_query(text: str) -> bool:
    t = (text or "").lower()
    # greetings
    greet = ["xin chào","chào","hello","hi","hey","good morning","good afternoon","good evening"]
    if any(g in t for g in greet): return True
    # scoring-related allowlist
    allow = [
        "chấm điểm","score","khách hàng tiềm năng","top","giải thích","vì sao","roc","độ chính xác","auc",
        "ngưỡng","threshold","đặc trưng","feature","next best action","nba","segment","phân khúc",
        "fairness","thiên vị","drift","chệch","calibration","huấn luyện","train","mô hình",
        "export","tải csv","api","kpi","pipeline","retrain","cập nhật mô hình","what-if","ảnh hưởng",
        "điểm","xác suất","probability","explain","top-k"
    ]
    return any(k in t for k in allow)

@app.post("/score_session", response_model=ScoreRunResponse)
def score_session(request: Request):
    sid = _sid_from_req(request)
    if not sid: raise HTTPException(status_code=400, detail="Thiếu Session ID")
    path = f"{SESS_DIR}/{sid}/data.parquet"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu trong phiên")
    df = pd.read_parquet(path)
    analysis = get_json(f"prompt:session:{sid}") or agent1_build_prompt(df)
    # build simple pipeline (private; not written to warehouse)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import roc_auc_score
    target = analysis.get("target")
    num_cols = df.select_dtypes(include=["number","bool"]).columns.tolist()
    if target in num_cols: num_cols.remove(target)
    cat_cols = [c for c in df.columns if c not in num_cols + ([target] if target else [])]
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), [c for c in num_cols if c in df.columns and c != target]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), [c for c in cat_cols if c in df.columns and c != target]),
    ], remainder="drop", sparse_threshold=0.0)
    X = df.drop(columns=[target], errors="ignore")
    if target and target in df.columns and df[target].nunique() <= 5:
        Xtr, Xte, ytr, yte = train_test_split(X, df[target], test_size=0.2, random_state=42, stratify=df[target] if df[target].nunique()==2 else None)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(X)[:,1]
        metrics = {"roc_auc": float(roc_auc_score(yte, pipe.predict_proba(Xte)[:,1])) if yte.nunique()==2 else None}
    else:
        Z = pre.fit_transform(X)
        gmm = GaussianMixture(n_components=2, random_state=42).fit(Z)
        P = gmm.predict_proba(Z)
        pos = int((P.mean(axis=0)).argmax())
        proba = P[:,pos]
        metrics = {"unsupervised": True}
    scored = df.copy()
    scored["_potential_proba"] = proba
    scored.to_parquet(f"{SESS_DIR}/{sid}/scored.parquet")
    log_event(sid, "score_session", {"rows": len(scored)})
    return {"rows": int(len(scored)), "metrics": metrics}

@app.get("/download_scored_session")
def download_scored_session(request: Request):
    sid = _sid_from_req(request)
    if not sid: raise HTTPException(status_code=400, detail="Thiếu Session ID")
    path = f"{SESS_DIR}/{sid}/scored.parquet"
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="Chưa có kết quả chấm")
    df = pd.read_parquet(path)
    import io
    s = io.StringIO(); df.to_csv(s, index=False); s.seek(0)
    return StreamingResponse(iter([s.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=scored_session.csv"})

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    # Enforce domain: only greeting or scoring questions allowed.
    if not _is_scoring_query(req.message):
        log_event("global", "blocked", {"q": req.message})
        return {"intent":"blocked","text":"Xin lỗi, tôi chỉ hỗ trợ chấm điểm khách hàng và các câu hỏi liên quan (top-k, giải thích, ROC, ngưỡng, đặc trưng, v.v.). Hãy đặt câu hỏi trong phạm vi đó."}
    sid = _sid_from_req(request)
    p = get_json(f"prompt:session:{sid}") if sid else None
    sys = "Bạn là chatbot scoring. Chỉ dùng dữ liệu của phiên hiện tại (nếu có). Không trả lời ngoài phạm vi scoring."
    user = req.message + (f"\\n\\nPROMPT_DATASET={json.dumps(p, ensure_ascii=False)}" if p else "")
    try:
        ans = chat_completion([{"role":"system","content": sys},{"role":"user","content": user}], 0.0)
    except Exception as e:
        ans = f"Không gọi được LLM: {e}"
    log_event(sid or "global", "chat", {"q": req.message, "session": bool(sid), "has_prompt": bool(p)})
    return {"intent":"chat","text": ans}

@app.get("/logs")
def logs(session_id: str = "global", last_n: int = 200):
    try:
        return {"logs": read_logs(session_id, last_n=last_n)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))