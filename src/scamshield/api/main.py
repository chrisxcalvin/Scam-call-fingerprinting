"""FastAPI app: REST + static web UI (Owner C)."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import WEB_DIR
from .pipeline import Models
from .routes import router
from .ws import router as ws_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index, script embeddings and Whisper once at startup; pre-cache warning audio."""
    from starlette.concurrency import run_in_threadpool
    app.state.models = await run_in_threadpool(Models().load, os.environ.get("SCAMSHIELD_WARM_STT", "1") == "1")
    if os.environ.get("SCAMSHIELD_WARM_TTS", "1") == "1":
        from ..agent.tts import warm_cache
        log.info("tts cache: %d warning phrases ready", await warm_cache())
    yield


app = FastAPI(title="ScamShield Voice", lifespan=lifespan)
app.include_router(router)
app.include_router(ws_router)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": {"code": "internal", "message": "Internal server error."}})


@app.middleware("http")
async def no_cache_for_ui(request: Request, call_next):
    """The UI is static files that change during development and demos: never serve a stale copy."""
    resp = await call_next(request)
    if not request.url.path.startswith(("/api", "/ws")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
