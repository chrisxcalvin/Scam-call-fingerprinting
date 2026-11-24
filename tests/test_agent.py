"""Agent tests run in offline mode (no network, no API quota)."""
import pytest

from scamshield.agent import agent as agent_mod
from scamshield.agent import llm
from scamshield.agent.agent import Agent, is_grounded, limit_sentences, offline_answer
from scamshield.agent.prompts import WARNINGS, build_user_prompt, system_prompt
from scamshield.schemas import Evidence, RedFlag, RiskComponents, RiskState, ScriptMatch


def ev(score=92.0, level="scam", guess="digital_arrest"):
    rs = RiskState(t_s=10, score=score, level=level, components=RiskComponents(fingerprint=0, script=1, redflags=0.9, voice=0),
                   top_reasons=["Words closely match a known digital_arrest script (similarity 0.93)", "Caller threatens arrest or account block"])
    return Evidence(risk=rs, fingerprint=[], red_flags=[RedFlag(category="threat", phrase="digital arrest", weight=0.8)],
                    script=[ScriptMatch(script_id="s", scam_type="digital_arrest", similarity=0.93, matched_text="m", query_text="q")],
                    voice=None, transcript_tail="you are under digital arrest", scam_type_guess=guess)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(llm, "provider", lambda: "offline")


def test_prompt_contains_rules_and_evidence():
    assert "ONLY" in system_prompt() and "3 short sentences" in system_prompt()
    p = build_user_prompt("why?", ev())
    assert "92/100" in p and "digital-arrest" in p and "QUESTION: why?" in p


def test_offline_answers_are_grounded_and_short():
    a = Agent()
    for q in ["Why do you think it's a scam?", "What should I do?", "Is this real?"]:
        r = a.answer(q, ev(), [])
        assert r.text and len([s for s in r.text.split(". ") if s]) <= 3 and is_grounded(r.text, ev())
    assert "only help with this call" in a.answer("What's the weather?", ev(), []).text


def test_safe_call_does_not_alarm():
    e = ev(11.0, "safe", None)
    assert "no strong scam signs" in offline_answer("Is this a scam?", e).lower()
    w = Agent().warn(e)
    assert w.audio_url is None and "No strong scam signs" in w.text


def test_warning_uses_cached_phrase():
    w = Agent().warn(ev())
    assert w.audio_url == "/api/tts/warn_digital_arrest" and w.text == WARNINGS["warn_digital_arrest"]


def test_grounding_guard_rejects_invented_numbers(monkeypatch):
    assert not is_grounded("About 80% of such calls steal money.", ev())
    assert is_grounded("Risk is 92 out of 100; report at 1930.", ev())
    assert not is_grounded("Eighty percent of such calls are fraud.", ev())
    assert not is_grounded("Thousands of victims lose money this way.", ev())
    assert is_grounded("This is likely a digital-arrest scam with a risk of ninety-two out of one hundred.", ev())
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Studies show 73% of victims lose money. Hang up.")
    r = Agent().answer("why?", ev(), [])
    assert "73" not in r.text                       # fell back to the template


def test_llm_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise llm.LLMUnavailable("x")
    monkeypatch.setattr(llm, "complete", boom)
    assert "digital-arrest" in Agent().answer("why?", ev(), []).text


def test_limit_sentences_keeps_hyphens_and_strips_markdown():
    assert limit_sentences("**Likely** a digital-arrest scam.\n- Hang up.\n- Call cybercrime.gov.in now. Four.") == \
        "Likely a digital-arrest scam. Hang up. Call cybercrime.gov.in now."
