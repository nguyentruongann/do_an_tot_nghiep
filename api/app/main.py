from fastapi import FastAPI

"""
Application entry point.  This file registers all routers for the API
including chat, scoring and logs.  Absolute imports are used throughout.
"""

from .routers.chat import router as chat_router
from .routers.score import router as score_router
from .routers.logs import router as logs_router
from .routers.nl_score import router as nl_score_router

app = FastAPI(title="VIP Scoring API")

# Register routers for chat, scoring and logs
app.include_router(chat_router)
app.include_router(score_router)
app.include_router(logs_router)
app.include_router(nl_score_router)


@app.get("/")
def root():
    return {"status": "ok", "service": "vip-scoring", "lang": "vi"}
