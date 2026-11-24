"""WebSocket /ws/live (Owner C): PCM16 frames in, analysis events out (docs/03)."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .session import LiveSession

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    """Client -> server: binary PCM16 LE 16 kHz mono frames (~100 ms); text {"type": start|stop|ping}.
    Server -> client: ready, viz, transcript, fp_match, risk, alert, final, error (JSON text).

    A receiver loop only appends audio; a separate worker runs analysis ticks in a thread so slow
    STT never blocks incoming frames. Ticks run in order, so `risk` events are monotone in t_s."""
    from .routes import LIVE, MAX_SESSIONS, _evidence, _remember
    await ws.accept()
    models = getattr(ws.app.state, "models", None)
    if models is None or models.index is None:
        await ws.send_json({"type": "error", "message": "Models are not loaded on the server."})
        await ws.close()
        return
    sid = ws.query_params.get("session_id", "")
    sess = LiveSession(models, session_id=sid) if sid.isalnum() and len(sid) <= 32 else LiveSession(models)
    LIVE[sess.session_id] = sess
    stop = asyncio.Event()
    wake = asyncio.Event()

    async def send(events: list[dict]) -> None:
        for e in events:
            await ws.send_text(json.dumps(e))

    async def worker() -> None:
        while not stop.is_set():
            await wake.wait()
            wake.clear()
            while sess.ready() and not stop.is_set():
                await send(await run_in_threadpool(sess.tick))

    await ws.send_json({"type": "ready", "session_id": sess.session_id})
    task = asyncio.create_task(worker())
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                sess.push(msg["bytes"])
                if sess.ready():
                    wake.set()
            elif msg.get("text"):
                cmd = json.loads(msg["text"]).get("type")
                if cmd == "ping":
                    await ws.send_json({"type": "pong"})
                elif cmd == "stop":
                    break
    except (WebSocketDisconnect, RuntimeError):
        stop.set(); task.cancel()
        LIVE.pop(sess.session_id, None)
        return
    stop.set(); wake.set()
    await asyncio.gather(task, return_exceptions=True)
    try:
        await send(await run_in_threadpool(sess.tick, True))          # transcribe the tail
        if sess.duration_s >= 0.5:
            from .pipeline import analyze_audio
            wav = _to_wav(sess)
            result, evidence = await run_in_threadpool(analyze_audio, wav, models, sess.session_id)
            # keep the live timeline (what the user actually saw) in the final result
            result.risk_timeline = sess.timeline or result.risk_timeline
            _remember(_evidence, sess.session_id, evidence, MAX_SESSIONS)
            await ws.send_json({"type": "final", "result": json.loads(result.model_dump_json())})
        await ws.close()
    except Exception:
        log.exception("live finalisation failed")
        try:
            await ws.send_json({"type": "error", "message": "Final analysis failed on the server."})
            await ws.close()
        except Exception:
            pass
    finally:
        LIVE.pop(sess.session_id, None)       # live snapshot no longer needed (final Evidence is kept)


def _to_wav(sess: LiveSession) -> bytes:
    """Session audio as WAV bytes (in memory only; nothing is written to disk)."""
    import io

    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, sess.signal(), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()
