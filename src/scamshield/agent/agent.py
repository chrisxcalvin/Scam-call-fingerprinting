"""Voice agent (Owner C): fixed warnings + grounded Q&A, with an offline template fallback."""
from __future__ import annotations

import logging
import re

from ..schemas import Evidence, SpokenMessage
from . import llm
from .prompts import ADVICE, TYPE_NAMES, WARNINGS, build_user_prompt, evidence_text, system_prompt, warning_key

log = logging.getLogger(__name__)
MAX_SENTENCES = 3
ALLOWED_NUMBERS = {"1930", "100"}        # helpline, and "/100" in scores


def limit_sentences(text: str, n: int = MAX_SENTENCES) -> str:
    """Keep the first n sentences and strip markdown symbols (replies are spoken aloud)."""
    t = re.sub(r"(?m)^\s*(?:[-*•]|\d+\.)\s+", "", text)        # list bullets at line start
    t = re.sub(r"\*\*|__|`|^#+\s*", "", t, flags=re.M)        # bold, code, headings
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(parts[:n]).strip()


def is_grounded(reply: str, evidence: Evidence) -> bool:
    """Cheap hallucination guard: every number in the reply must appear in the evidence text
    (or be the 1930 helpline). Invented statistics like '80% of such calls' fail this check, and so do
    percentages/statistic words spelled out ('eighty percent', 'most victims'), since Evidence never has any."""
    ev = evidence_text(evidence)
    for num in re.findall(r"\d+(?:\.\d+)?", reply):
        if num not in ev and num not in ALLOWED_NUMBERS:
            return False
    if re.search(r"%|\bper ?cent\b|\bstatistic|\b(most|many|thousands of|millions of) (victims|people|calls)\b",
                 reply, re.I):
        return False
    return True


_INTENTS = [   # (name, regex): first match wins; checked in this order
    ("greeting", r"^(hi|hello|hey|hola|namaste|good (morning|afternoon|evening))\b"),
    ("closing", r"\b(thank(s| you)|bye|goodbye|take care|see you|that'?s all|all good)\b"),
    ("identity", r"\b(who|what) (are|r) (you|u)\b|\byour name\b|\bwho (made|built|created) you\b|\bwhat is scamshield\b|\bare you (a )?(bot|human|robot|ai)\b"),
    ("capability", r"\b(what can you do|how do you work|how does (this|it) work|what do you do)\b"),
    ("why", r"\b(why|how do you know|reason|evidence|what makes)\b"),
    ("what_to_do", r"\b(what (should|do|can) i|what now|next steps?|report)\b"),
    ("is_it", r"\b(is (it|this|that)|scam|safe|real|genuine|fake|sure)\b"),
]


def _intent(q: str) -> str:
    q = q.lower().strip()
    for name, pattern in _INTENTS:
        if re.search(pattern, q) and (name != "greeting" or len(q.split()) <= 4):
            return name
    return "other"


def offline_answer(question: str, ev: Evidence) -> str:
    """Template reply built only from Evidence and fixed descriptions of ScamShield (AGENT_OFFLINE=1, no key, or LLM failure)."""
    r, intent = ev.risk, _intent(question)
    risky = r.level in ("high", "scam")
    kind = TYPE_NAMES.get(ev.scam_type_guess or "", "scam")
    if intent == "greeting":
        return "Hi, I'm ScamShield. Ask me why this call was flagged, or what you should do."
    if intent == "closing":
        return "You're welcome. Stay safe, and hang up if they ring back." if risky else "You're welcome. Take care."
    if intent == "identity":
        return "I'm ScamShield, a voice assistant that listens to calls and tells you whether they look like scams, and why."
    if intent == "capability":
        return ("I compare a call with known scam recordings, check what is being said, and explain my reasoning. "
                "Ask me why it was flagged or what to do.")
    if intent == "other":
        return "That's outside what I can help with, but I'm happy to explain this call."
    if not risky and intent in ("why", "is_it"):
        return (f"I found no strong scam signs in this call; the risk is {r.score:.0f} out of 100. "
                "If you are unsure, hang up and call the official number yourself.")
    if intent == "what_to_do":
        return ADVICE if risky else "No strong scam signs were found. If anything feels wrong, hang up and call the official number yourself."
    if intent == "is_it":
        return f"This call is likely a {kind} scam, with a risk of {r.score:.0f} out of 100. {ADVICE.split('. ')[0]}."
    reasons = "; ".join(r.top_reasons[:2]).rstrip(".") or "several warning signs"
    return f"It is likely a {kind} scam because: {reasons}. Do not share any OTP or PIN, and hang up."


class Agent:
    def warn(self, evidence: Evidence) -> SpokenMessage:
        """Fixed warning for the guessed scam type; audio is pre-cached at /api/tts/<key>."""
        if evidence.risk.level not in ("high", "scam"):
            return SpokenMessage(text="No strong scam signs were found in this call. If you are unsure, "
                                      "hang up and call the official number yourself.")
        key = warning_key(evidence.scam_type_guess)
        return SpokenMessage(text=WARNINGS[key], audio_url=f"/api/tts/{key}")

    def answer(self, question: str, evidence: Evidence, history: list[dict]) -> SpokenMessage:
        """LLM answer grounded on `evidence`; falls back to the offline template if the LLM is
        unavailable or the reply fails the grounding check. audio_url is set by the API layer."""
        try:
            reply = limit_sentences(llm.complete(system_prompt(), build_user_prompt(question, evidence), history[-6:]))
            if not reply or not is_grounded(reply, evidence):
                log.info("LLM reply rejected (empty or ungrounded number); using template")
                reply = offline_answer(question, evidence)
        except llm.LLMUnavailable:
            reply = offline_answer(question, evidence)
        return SpokenMessage(text=reply)
