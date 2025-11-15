from __future__ import annotations
from fastapi import FastAPI
from .routers import chat, score, logs

app = FastAPI(title="VIP Scoring API — Priority Grounded")
app.include_router(score.router)
app.include_router(chat.router)
app.include_router(logs.router)

@app.get("/")
def root():
    return {"ok": True, "service": "vip-scoring", "lang": "vi"}
