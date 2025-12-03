from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .routers.chat import router as chat_router
from .routers.score import router as score_router
from .routers.logs import router as logs_router
from .routers.nl_score import router as nl_score_router

app = FastAPI(title="VIP Scoring API")

app.include_router(chat_router)
app.include_router(score_router)
app.include_router(logs_router)
app.include_router(nl_score_router)

# --- Serve static UI ---
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Mount thư mục static (nếu sau này bạn có thêm JS/CSS riêng)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Trang chủ trả về file index.html
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

# Nếu vẫn muốn health check JSON:
@app.get("/health")
def health():
    return {"status": "ok", "service": "vip-scoring", "lang": "vi"}
