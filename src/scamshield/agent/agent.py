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


def _intent(q: str) -> str:
    q = q.lower()
    if re.search(r"\b(why|how do you know|reason|evidence|what makes)\b", q):
        return "why"
    if re.search(r"\b(what (should|do|can) i|what now|help|next|steps?|report)\b", q):
        return "what_to_do"
    if re.search(r"\b(is (it|this|that)|scam|safe|real|genuine|fake|sure)\b", q):
        return "is_it"
    return "other"


def offline_answer(question: str, ev: Evidence) -> str:
    """Template reply built only from Evidence (AGENT_OFFLINE=1, no key, or LLM failure)."""
    r, intent = ev.risk, _intent(question)
    risky = r.level in ("high", "scam")
    kind = TYPE_NAMES.get(ev.scam_type_guess or "", "scam")
    if intent == "other":
        return "I can only help with this call. Ask me why it was flagged, or what you should do."
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
