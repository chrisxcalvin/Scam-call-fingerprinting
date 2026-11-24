"""Uses the real multilingual embedder (downloaded on first use)."""
from scamshield.understanding.script_match import ScriptLibrary, script_match, windows

SCRIPTS = [
    {"id": "s_arrest", "label": "scam", "type": "digital_arrest",
     "text": "This is the cyber crime police. You are under digital arrest. Do not tell anyone. Transfer your savings now."},
    {"id": "s_lottery", "label": "scam", "type": "lottery",
     "text": "Congratulations, you won a lucky draw. Pay a small processing fee to release the prize."},
    {"id": "l_friend", "label": "legit", "type": "friend", "text": "Are you free for lunch on Saturday?"},
]


def test_windows():
    assert windows(["a.", "b.", "c."]) == ["a. b.", "b. c."] and windows(["a."]) == ["a."] and windows([]) == []


def test_paraphrase_scores_above_unrelated():
    lib = ScriptLibrary.build(SCRIPTS)
    assert {c["script_id"] for c in lib.chunks} == {"s_arrest", "s_lottery"}     # legit not in library
    para = script_match("Police cyber cell here. You are digitally arrested and must not inform your family.", lib)
    other = script_match("Hi, the plumber will come at ten tomorrow to fix the kitchen tap.", lib)
    assert para[0].script_id == "s_arrest" and para[0].similarity > other[0].similarity + 0.2
