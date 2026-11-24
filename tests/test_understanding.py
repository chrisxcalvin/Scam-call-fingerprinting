import pytest

from scamshield.fusion.risk import fuse, level_of, script_component
from scamshield.schemas import RiskComponents, TranscriptSegment
from scamshield.understanding.redflags import red_flags, redflag_score, safe_context_hits


def seg(text, a=0.0, b=1.0):
    return TranscriptSegment(start_s=a, end_s=b, text=text)


def test_scam_phrases_flagged():
    fl = red_flags([seg("This is the cyber crime police. You are under digital arrest. Do not tell anyone. "
                        "Transfer your savings to a secure account right now.")])
    cats = {f.category for f in fl if not f.negated}
    assert {"authority", "threat", "secrecy", "payment", "urgency"} <= cats
    assert redflag_score(fl) > 0.9


def test_negation_cancels_credential_request():
    fl = red_flags([seg("We will never ask for your PIN or OTP.")])
    assert fl and all(f.negated for f in fl if f.category == "credential_request")
    assert redflag_score(fl) == 0.0


def test_request_for_otp_not_negated():
    fl = red_flags([seg("Please read out the six digit OTP right now.")])
    assert any(f.category == "credential_request" and not f.negated for f in fl)


def test_safe_context_and_acronyms():
    assert safe_context_hits("If this was not you, please call the number printed on the back of your card.")
    assert any(f.phrase == "otp" for f in red_flags([seg("Share the O.T.P. now")]))


def test_romanised_hindi():
    cats = {f.category for f in red_flags([seg("kisi ko mat batana, abhi paise bhej do")])}
    assert {"secrecy", "urgency", "payment"} <= cats


def test_fusion_noisy_or_and_levels():
    z = RiskComponents(fingerprint=0, script=0, redflags=0, voice=0)
    assert fuse(z, False).score == 0 and fuse(z, False).level == "safe"
    st = fuse(RiskComponents(fingerprint=0, script=1.0, redflags=0.5, voice=0), False)
    assert st.score == pytest.approx(100 * (1 - (1 - 0.8) * (1 - 0.3)), abs=0.1)
    assert fuse(z, True).score >= 90 and fuse(z, True).level == "scam"
    assert [level_of(s) for s in (10, 30, 60, 85)] == ["safe", "suspicious", "high", "scam"]
    capped = fuse(RiskComponents(fingerprint=0, script=0, redflags=0.9, voice=0), False, safe_context=True)
    assert capped.components.redflags == 0.3


def test_script_component_mapping():
    assert script_component(0.3) == 0 and script_component(0.95) == 1 and 0 < script_component(0.7) < 1


def test_every_scam_type_has_a_warning():
    import json

    from scamshield.agent.prompts import WARNINGS, warning_key
    from scamshield.config import DATA_DIR
    p = DATA_DIR / "scripts.json"
    if not p.exists():
        pytest.skip("no data")
    types = {s["type"] for s in json.load(open(p, encoding="utf-8")) if s["label"] == "scam"}
    assert all(warning_key(t) != "warn_generic" for t in types), types - {k[5:] for k in WARNINGS}
