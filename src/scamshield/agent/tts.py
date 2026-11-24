"""Text-to-speech with cached warnings (Owner C)."""
from __future__ import annotations

import logging

from ..config import TTS_CACHE_DIR, get_settings
from .prompts import WARNINGS

log = logging.getLogger(__name__)


def _use_os_trust_store() -> None:
    """Use the OS certificate store for TLS if `truststore` is installed (verification stays ON).
    Needed on machines where antivirus/proxy certificates are only in the Windows store."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


async def tts(text: str, voice: str | None = None) -> bytes:
    """edge-tts synthesis -> mp3 bytes. Raises on network failure; callers fall back to browser
    speechSynthesis (SpokenMessage.audio_url = None)."""
    import edge_tts

    _use_os_trust_store()
    chunks = []
    async for msg in edge_tts.Communicate(text, voice or get_settings().tts_voice).stream():
        if msg["type"] == "audio":
            chunks.append(msg["data"])
    return b"".join(chunks)


async def cached_warning(key: str) -> bytes | None:
    """Audio for a fixed warning phrase, synthesised once and cached in artifacts/tts_cache/<key>.mp3."""
    if key not in WARNINGS:
        return None
    p = TTS_CACHE_DIR / f"{key}.mp3"
    if not p.exists():
        TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p.write_bytes(await tts(WARNINGS[key]))
    return p.read_bytes()


async def warm_cache() -> int:
    """Pre-generate all warning phrases (zero-latency alerts). Returns how many are cached."""
    n = 0
    for k in WARNINGS:
        try:
            await cached_warning(k)
            n += 1
        except Exception as e:  # network down -> UI falls back to speechSynthesis
            log.warning("tts cache %s failed: %s", k, type(e).__name__)
    return n


async def stream_tts(text: str, voice: str | None = None):
    """Yield mp3 chunks as edge-tts produces them (free-form answers: playback starts on the first chunk)."""
    import edge_tts

    _use_os_trust_store()
    async for msg in edge_tts.Communicate(text, voice or get_settings().tts_voice).stream():
        if msg["type"] == "audio":
            yield msg["data"]
