"""Weighted red-flag lexicon with negation / safe-context handling (Owner B).

Patterns are deliberately generic (not copied sentences from scripts.json) so that paraphrased
holdout scripts are a fair test. Weights are starting values, to be tuned on the dev split only.
"""
from __future__ import annotations

import re

from ..schemas import RedFlag, TranscriptSegment

# category -> [(regex, weight)]   (English + romanised Hindi)
LEXICON: dict[str, list[tuple[str, float]]] = {
    "urgency": [(r"right now|immediately|turant|abhi(?! tak)", 0.5), (r"\burgent\w*", 0.5),
                (r"within (\w+ )?(hours?|minutes?)|(do ?n[o']?t|der mat) (delay|kijiye|karo)", 0.6),
                (r"\bhurry\b|jaldi( karo)?|aaj hi", 0.5)],
    "authority": [(r"\binspector\b|\bpolice\b|cyber ?crime|\bcbi\b|\brbi\b|customs|income tax", 0.6),
                  (r"head office|government|sarkari|technical support|tech support", 0.5)],
    "secrecy": [(r"(do ?n[o']?t|never) tell (anyone|anybody|mom|dad|your family|\w+)", 0.8),
                (r"keep (this|it) (a )?secret|secret rakh", 0.8),
                (r"(kisi|mummy|papa|ghar)( \w+)? ko (bhi )?(mat|na|nahi) bata", 0.8),
                (r"(do ?n[o']?t) disconnect|(call )?kaat(na)? mat|phone mat kaat", 0.7)],
    "credential_request": [(r"\botp\b|one[- ]?time password|\bpin\b|\bcvv\b|password", 0.6),
                           (r"card details|(read|tell) (me )?(out )?(the|your) (code|otp|number)|(six|6)[- ]digit", 0.8),
                           (r"\bkyc\b", 0.4)],
    "payment": [(r"\btransfer\w*", 0.7), (r"processing fee|redelivery fee|\bfee\b", 0.6),
                (r"(send|pay) (\w+ ){0,4}(rupees|money|amount)|bhej (do|dijiye)|bhar(ni|na|o)\b", 0.7),
                (r"secure (\w+ )?account|safe account", 0.8)],
    "threat": [(r"\barrest\w*|digital arrest", 0.8), (r"\bblock(ed)?\b|block ho", 0.5),
               (r"send a team|legal action|\billegal\b|\bwarna\b|jail", 0.6)],
    "remote_access": [(r"remote access|anydesk|teamviewer|screen ?shar\w*", 0.9),
                      (r"install (the |this |an? )?(\w+ )?(app|application)|click (the|this) link", 0.7),
                      (r"keep your bank app open", 0.7)],
}
# local negation just before the phrase cancels credential/payment/remote flags ("never ask for your PIN")
NEGATION = r"\b(never|not|do ?n[o']?t|won'?t|kabhi|nahi|mat)\b"
NEGATABLE = {"credential_request", "payment", "remote_access"}
NEG_WINDOW_WORDS = 7
# safe-context cues typical of genuine bank / OTP messages
SAFE_CONTEXT = [r"if (this|it) was not you", r"(number|helpline) (printed |given )?on the back of your card",
                r"we will never ask", r"(do ?n[o']?t|never) share (this|the|your) (code|otp|pin)",
                r"agar (ye|yeh) aapne nahi kiya", r"card ke peeche", r"kabhi (\w+ ){0,4}nahi maang"]
SAFE_CONTEXT_FACTOR = 0.5


def normalise(text: str) -> str:
    """Lower-case, join dotted acronyms (O.T.P. -> otp), collapse whitespace."""
    t = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), text.lower())
    return re.sub(r"\s+", " ", t)


def safe_context_hits(text: str) -> list[str]:
    """Safe-context phrases present in the text (used here and by fusion to cap red-flag weight)."""
    t = normalise(text)
    return [m.group(0) for p in SAFE_CONTEXT for m in re.finditer(p, t)]


def _negated(t: str, start: int) -> bool:
    before = t[:start].split()[-NEG_WINDOW_WORDS:]
    return re.search(NEGATION, " ".join(before)) is not None


def red_flags(segments: list[TranscriptSegment]) -> list[RedFlag]:
    """Scan each segment with the lexicon; one RedFlag per match (phrase as spoken).

    negated=True when a negation word occurs within the 7 words before a credential/payment/
    remote-access phrase. If the whole transcript contains safe-context cues ("if this was not you",
    "we will never ask..."), remaining credential/payment weights are multiplied by 0.5.
    """
    safe = bool(safe_context_hits(" ".join(s.text for s in segments)))
    out: list[RedFlag] = []
    for seg in segments:
        t = normalise(seg.text)
        taken: list[tuple[int, int]] = []
        for cat, pats in LEXICON.items():
            for pat, w in pats:
                for m in re.finditer(pat, t):
                    if any(a < m.end() and m.start() < b for a, b in taken):
                        continue  # overlapping phrase already flagged
                    taken.append((m.start(), m.end()))
                    neg = cat in NEGATABLE and _negated(t, m.start())
                    weight = w * (SAFE_CONTEXT_FACTOR if safe and cat in ("credential_request", "payment") else 1.0)
                    out.append(RedFlag(category=cat, phrase=m.group(0), weight=round(weight, 3),
                                       start_s=seg.start_s, end_s=seg.end_s, negated=neg))
    return out


def redflag_score(flags: list[RedFlag]) -> float:
    """0..1 red-flag strength: noisy-OR over categories of the strongest non-negated flag per category,
    1 - prod_c (1 - max_w_c). Many categories together (authority + secrecy + payment) push it up."""
    best: dict[str, float] = {}
    for f in flags:
        if not f.negated:
            best[f.category] = max(best.get(f.category, 0.0), f.weight)
    p = 1.0
    for w in best.values():
        p *= 1.0 - w
    return round(1.0 - p, 4)
