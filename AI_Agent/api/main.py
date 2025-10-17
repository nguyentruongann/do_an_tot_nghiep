from fastapi import FastAPI
from api.routers.score import router as score_router
from database.bootstrap import bootstrap_db

app = FastAPI(title="Lead Agent Pro (LLM-first)")
bootstrap_db()
app.include_router(score_router, prefix="/v1", tags=["scoring"])

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
