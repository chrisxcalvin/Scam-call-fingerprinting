"""LLM provider abstraction (Owner C). Providers: gemini | anthropic | offline."""
from __future__ import annotations

import logging
from functools import lru_cache

from ..config import get_settings

log = logging.getLogger(__name__)
TIMEOUT_S = 12.0          # Gemini rejects deadlines < 10 s; normal replies take ~2 s
MAX_TOKENS = 200


class LLMUnavailable(RuntimeError):
    """Raised when no network LLM is configured/reachable; the agent then uses its offline templates."""


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai
    from google.genai import types

    from .tts import _use_os_trust_store
    _use_os_trust_store()
    return genai.Client(api_key=get_settings().gemini_api_key.strip(),
                        http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)))


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key.strip(), timeout=TIMEOUT_S)


def provider() -> str:
    """Active provider after checking keys: 'offline' if AGENT_OFFLINE=1 or the key is missing."""
    s = get_settings()
    if s.agent_offline or s.llm_provider == "offline":
        return "offline"
    if s.llm_provider == "gemini" and s.gemini_api_key.strip():
        return "gemini"
    if s.llm_provider == "anthropic" and s.anthropic_api_key.strip():
        return "anthropic"
    return "offline"


def complete(system: str, user: str, history: list[dict] | None = None) -> str:
    """One chat turn. history = [{"role": "user"|"assistant", "text": ...}] (oldest first).
    Raises LLMUnavailable in offline mode or on any provider error (never leaks the key)."""
    p = provider()
    turns = (history or []) + [{"role": "user", "text": user}]
    try:
        if p == "gemini":
            from google.genai import types
            r = _gemini_client().models.generate_content(
                model=get_settings().gemini_model,
                contents=[types.Content(role="model" if t["role"] == "assistant" else "user",
                                        parts=[types.Part(text=t["text"])]) for t in turns],
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=MAX_TOKENS,
                                                   temperature=0.2))
            return (r.text or "").strip()
        if p == "anthropic":
            r = _anthropic_client().messages.create(
                model=get_settings().anthropic_model, max_tokens=MAX_TOKENS, system=system,
                messages=[{"role": t["role"], "content": t["text"]} for t in turns])
            return "".join(b.text for b in r.content if b.type == "text").strip()
    except Exception as e:
        log.warning("LLM %s failed: %s", p, type(e).__name__)
        raise LLMUnavailable(type(e).__name__) from None
    raise LLMUnavailable("offline")
