import numpy as np
import pytest

from scamshield.understanding.text_model import TextModel, to_component

SCAM = ["You must transfer your savings to the safe account right now and do not tell anyone.",
        "Read out the code that came on your phone or your account will be blocked today.",
        "Pay the processing fee immediately to receive your prize, it is a secret until then.",
        "Your son is in trouble with the police, send money right now and do not tell your wife."] * 4
LEGIT = ["Your bill is due on the fifteenth, you can pay on our official app or at the counter.",
         "We will never ask for your PIN. If you did not make this payment call the number on your card.",
         "Hey are you free for lunch on Saturday, let me know and I will book the table.",
         "I am outside with your order, please tell me the delivery code from the app."] * 4


def _model():
    return TextModel.fit(SCAM + LEGIT, [True] * len(SCAM) + [False] * len(LEGIT), C=10)


def test_scam_scores_above_legit_on_unseen_text():
    m = _model()
    d = m.decision(["Transfer the money to this account immediately and keep this call secret.",
                    "Can you tell me which floor to come to, I have your delivery."])
    assert d[0] > 0 > d[1]


def test_empty_text_is_not_scam_and_roundtrip(tmp_path):
    m = _model()
    assert m.decision([""])[0] < 0
    m.save(tmp_path / "m.pkl")
    np.testing.assert_allclose(TextModel.load(tmp_path / "m.pkl").decision(["right now transfer"]), m.decision(["right now transfer"]))
    scam_f, legit_f = m.top_features(5)
    assert len(scam_f) == 5 and scam_f[0][1] > 0 > legit_f[0][1]


def test_component_mapping():
    assert to_component(-5, (0, 2)) == 0 and to_component(5, (0, 2)) == 1 and to_component(1, (0, 2)) == 0.5
