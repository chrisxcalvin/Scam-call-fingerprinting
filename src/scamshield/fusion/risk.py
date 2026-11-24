"""Noisy-OR risk fusion (Owner B)."""
from __future__ import annotations

import json

from ..config import (FP_OVERRIDE_MIN_SCORE, FUSION_CONFIG_PATH, FUSION_WEIGHTS, LEVEL_BOUNDS,
                      SAFE_CONTEXT_REDFLAG_CAP, SCRIPT_SIM_RANGE)
from ..schemas import Level, RiskComponents, RiskState

CATEGORY_TEXT = {"urgency": "pressures you to act immediately", "authority": "claims to be police/bank/official",
                 "secrecy": "asks you to keep the call secret", "credential_request": "asks for OTP/PIN/card details",
                 "payment": "asks you to pay or transfer money", "threat": "threatens arrest or account block",
                 "remote_access": "asks you to install an app or click a link"}


def load_config() -> dict:
    """Fusion parameters: artifacts/fusion_config.json if run_eval wrote one, else provisional defaults."""
    cfg = {"weights": dict(FUSION_WEIGHTS), "level_bounds": list(LEVEL_BOUNDS), "fp_override": FP_OVERRIDE_MIN_SCORE,
           "script_sim_range": list(SCRIPT_SIM_RANGE), "safe_context_cap": SAFE_CONTEXT_REDFLAG_CAP,
           "source": "provisional defaults (config.py)"}
    if FUSION_CONFIG_PATH.exists():
        cfg.update(json.loads(FUSION_CONFIG_PATH.read_text(encoding="utf-8")))
    return cfg


def script_component(similarity: float, cfg: dict | None = None) -> float:
    """Map cosine similarity to 0..1: s = clip((sim - lo) / (hi - lo), 0, 1). Legit speech already has
    sim ~0.3-0.6 against scam scripts (shared words like bank/card), so raw cosine is not a probability."""
    lo, hi = (cfg or load_config())["script_sim_range"]
    return round(min(1.0, max(0.0, (similarity - lo) / (hi - lo))), 4)


def level_of(score: float, bounds: list[float] | tuple[float, ...] = LEVEL_BOUNDS) -> Level:
    """<30 safe | 30-60 suspicious | 60-85 high | >=85 scam."""
    if score >= bounds[2]:
        return "scam"
    if score >= bounds[1]:
        return "high"
    if score >= bounds[0]:
        return "suspicious"
    return "safe"


def fuse(components: RiskComponents, fp_hit: bool, *, t_s: float = 0.0, details: dict | None = None,
         safe_context: bool = False, cfg: dict | None = None) -> RiskState:
    """score = 100 * (1 - prod_i (1 - w_i * s_i)),  i in {fingerprint, script, redflags, voice}.

    Noisy-OR: each layer is an independent chance of 'this is a scam'; any one strong layer is enough,
    weak layers add up without exceeding 100. Rules: a fingerprint hit forces score >= 90; safe-context
    cues cap the red-flag component. `details` (scam_type, similarity, flag categories, ...) only feed
    the human-readable `top_reasons`, ordered by each layer's contribution w_i * s_i.
    """
    cfg = cfg or load_config()
    w = cfg["weights"]
    c = components.model_copy()
    if safe_context:
        c.redflags = min(c.redflags, cfg["safe_context_cap"])
    s = {"fingerprint": c.fingerprint, "script": c.script, "redflags": c.redflags, "voice": c.voice}
    p = 1.0
    for k, v in s.items():
        p *= 1.0 - w[k] * v
    score = 100.0 * (1.0 - p)
    if fp_hit:
        score = max(score, cfg["fp_override"])
    d = details or {}
    reasons: list[tuple[float, str]] = []
    if fp_hit:
        reasons.append((9.0, f"Audio matches a known {d.get('fp_scam_type', 'scam')} scam recording"))
    if s["script"] > 0:
        how = "closely match" if s["script"] >= 0.5 else "partly resemble"
        reasons.append((w["script"] * s["script"],
                        f"Words {how} a known {d.get('script_scam_type', 'scam')} script (similarity {d.get('similarity', 0):.2f})"))
    for cat in d.get("flag_categories", [])[:3]:
        reasons.append((w["redflags"] * s["redflags"] * 0.99, f"Caller {CATEGORY_TEXT.get(cat, cat)}"))
    if safe_context:
        reasons.append((0.01, "Contains safe-context phrases typical of genuine bank alerts"))
    reasons.sort(key=lambda r: -r[0])
    return RiskState(t_s=round(t_s, 2), score=round(score, 1), level=level_of(score, cfg["level_bounds"]),
                     components=c, top_reasons=[r[1] for r in reasons[:4]])
