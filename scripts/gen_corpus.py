"""Generate a diverse, fictional scam / legit script corpus with Gemini for training and evaluating Layer 2.

Splits (by script, each from its own API calls):
  train : library + classifier training          model A (gemini-3.5-flash-lite), prompt style A
  dev   : calibration + model selection           model A, different calls / seeds
  test2 : LOCKED final test, run once             model B (gemini-3.5-flash), prompt style B, different seeds

Everything is fictional and for training a DEFENSIVE detector. Output: scripts/corpus/{train,dev,test2}.json
(lists of {id,label,type,lang,text,split,model}); the SHA-256 of test2.json is printed so it can be recorded
before any model is trained.   Usage: python scripts/gen_corpus.py [--only train|dev|test2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "scripts" / "corpus"
MODEL_A, MODEL_B = "gemini-3.5-flash-lite", "gemini-3.5-flash"

SCAM = {
    "digital_arrest": "'digital arrest' scam: caller poses as police / CBI / customs / narcotics officer, claims the victim is linked to a crime (parcel, money laundering, SIM misuse), orders them to stay on the call or video, forbids telling anyone, and demands money be moved to a 'safe' or 'verification' account",
    "kyc_otp": "KYC / OTP / card scam: caller poses as a bank, card company, telecom or wallet employee, says an account, card or SIM will be blocked or KYC/PAN/Aadhaar is pending, and tries to get the victim to read out a code, PIN, CVV or card details, or to confirm account details",
    "lottery_prize": "lottery / prize / gift scam: caller says the victim won a lucky draw, car, cash prize, gift or refund and must first pay a fee, tax, GST, insurance or courier charge, often by UPI, gift cards or bank transfer",
    "relative_emergency": "relative-in-trouble scam: caller pretends to be a grandchild, nephew, son, friend of a relative or a doctor/police officer, claims an accident, arrest, hospital bill or other emergency, asks for urgent money, and asks the victim not to tell other family members",
    "fake_courier": "fake courier / customs scam: caller or automated message says a parcel is held, undeliverable or has unpaid duty, asks the victim to press a key, click a link, pay a small fee or enter card details to release it",
    "tech_support": "tech support / remote access scam: caller poses as Microsoft, an antivirus company, an ISP or a bank security team, says the computer or account is hacked, and asks the victim to install a remote-access app, share the screen, open net banking or buy gift cards",
}
SCAM_HI = ["digital_arrest", "kyc_otp", "lottery_prize", "relative_emergency"]
LEGIT = {
    "bank_alert": "automated or human bank message about a transaction, balance or card use that says to call the number on the card if it was not the customer, and that the bank never asks for PIN/OTP",
    "bank_kyc_reminder": "genuine bank or wallet reminder to update KYC documents at a branch or in the official app, warning never to share OTP or passwords",
    "otp_notice": "an automated message delivering a one-time password for a login or transaction, with the standard warning not to share it with anyone",
    "bank_fraud_check": "a genuine fraud-monitoring call or message asking the customer to confirm or deny a recent payment by pressing a key, stating the bank will never ask for card numbers or OTP",
    "delivery": "a real delivery person, courier or grocery/food partner calling about an order at the gate, asking for the delivery code shown in the app or a landmark",
    "hospital": "a hospital, clinic or school nurse informing a parent or relative about a minor injury, a test result being ready, or an appointment, calmly and without asking for money on the phone",
    "police_notice": "a genuine police, traffic or municipal helpdesk notice about a challan, lost-and-found item or verification, directing the person to an official website or station and warning against paying anyone over the phone",
    "utility_bill": "electricity, gas, water or broadband provider reminding about a bill due date and official payment channels, telling the customer to ignore threatening messages",
    "telecom": "mobile operator or broadband company offering a plan, announcing maintenance, or confirming a recharge, without asking for secrets",
    "appointment": "a clinic, salon, bank branch, passport office or service centre confirming or rescheduling an appointment",
    "family": "a family member calling to say they are running late, asking what to buy, or sharing news, casual and warm",
    "friend": "a friend or colleague calling to plan lunch, a weekend trip, a meeting or to borrow notes",
    "workplace": "a manager, HR or IT helpdesk of the person's own employer calling about a meeting, a laptop update, payroll date or a ticket, without asking for passwords",
    "ecommerce": "an online shop or marketplace confirming an order, a refund to the original payment method or a return pickup, and warning never to share OTP or click unknown links",
    "government_service": "a genuine government or insurance helpline calling about a scheme registration, pension certificate or Aadhaar centre appointment, directing to the official portal",
    "customer_survey": "a company calling for a short, polite customer-satisfaction survey after a service visit, asking nothing sensitive",
}
LEGIT_HI = ["bank_alert", "otp_notice", "delivery", "family", "friend", "appointment", "utility_bill", "workplace"]
TONES = ["very polite and formal", "aggressive and pushy", "friendly and casual", "hurried and anxious", "bureaucratic and official",
         "soft-spoken and reassuring", "using technical jargon", "using very simple words for an elderly listener", "rambling with small talk first",
         "terse and to the point", "sympathetic", "authoritative"]
LANG_NOTE = {"en": "English (Indian context: rupees, UPI, Aadhaar, PAN, lakh; Indian names).",
             "hi": "Hinglish: Hindi written in the ROMAN alphabet mixed with English words, as Indians speak and type it (for example 'aapka account block ho jayega'). Never use Devanagari or Urdu script."}


def prompt_a(label: str, desc: str, n: int, lang: str, rng: random.Random) -> str:
    tones = ", ".join(rng.sample(TONES, 4))
    if label == "scam":
        task = f"scripts of what a SCAMMER says in the first 20-40 seconds of a phone call. Scam type: {desc}."
        extra = "Vary the persona, organisation, story, order of tactics and wording. Do NOT always use the same keywords (for example do not always say OTP or transfer): real scammers also say 'code', 'verification number', 'send', 'deposit', 'sort it out', 'settle', 'fee', etc."
    else:
        task = f"scripts of a GENUINE, harmless phone call or automated message. Kind: {desc}."
        extra = "Make them natural and realistic. Where it is natural for this kind of call, include words that scam calls also use (bank, account, OTP, code, police, payment, KYC, urgent), exactly as honest organisations and people use them."
    return (f"You are helping build a research dataset to train a DEFENSIVE scam-call detector. All names, numbers and organisations are fictional.\n"
            f"Write {n} distinct, realistic {task}\nLanguage: {LANG_NOTE[lang]}\nTones to mix across the scripts: {tones}.\n{extra}\n"
            f"Each script is 3-6 sentences of plain spoken text (no stage directions, no quotes, no numbering).\nReturn ONLY a JSON array of {n} strings.")


def prompt_b(label: str, desc: str, n: int, lang: str, rng: random.Random) -> str:
    """Different wording and structure than prompt A on purpose (test2 should not mirror the training prompt)."""
    role = ("calls that are scams, written as a speech-to-text transcript of the caller's opening turn" if label == "scam"
            else "perfectly legitimate calls or automated voice messages, written as a speech-to-text transcript of the opening turn")
    return (f"Dataset task (defensive research, fictional details only). Produce {n} varied transcripts of {role}.\n"
            f"Category: {desc}.\nLanguage/style: {LANG_NOTE[lang]}\n"
            f"Requirements: different speakers and situations in every item; length between 35 and 90 words; spoken register with contractions and occasional hesitation; "
            f"{'avoid repeating the same trigger words; paraphrase heavily' if label == 'scam' else 'include honest uses of risky-sounding words where natural'}.\n"
            f"Output format: a JSON array of exactly {n} strings and nothing else.")


def prompt_direct(label: str, desc: str, n: int, lang: str, rng: random.Random) -> str:
    """Plain, explicit scam wording (the common words real scammers do use). Added to train because prompt A
    told the model to avoid keywords, which taught the classifier that 'OTP' means legit."""
    tones = ", ".join(rng.sample(TONES, 3))
    lines = [
        "You are helping build a research dataset to train a DEFENSIVE scam-call detector. All names, numbers and organisations are fictional.",
        f"Write {n} distinct realistic scripts of what a SCAMMER says in the first 20-40 seconds of a phone call. Scam type: {desc}.",
        f"Language: {LANG_NOTE[lang]}",
        f"Tones to mix: {tones}.",
        "Use the plain, DIRECT wording that real scam calls commonly use, including the explicit words where they fit the scam: "
        "'OTP', 'PIN', 'CVV', 'transfer', 'send the money', 'UPI', 'gift card', 'account will be blocked', 'do not tell anyone', 'right now', 'immediately', 'arrest'.",
        "Each script is 3-5 sentences of plain spoken text (no stage directions, no numbering).",
        f"Return ONLY a JSON array of {n} strings.",
    ]
    return chr(10).join(lines)


def call(client, types, model: str, prompt: str, tries: int = 4) -> list[str]:
    for k in range(tries):
        try:
            r = client.models.generate_content(model=model, contents=prompt,
                                               config=types.GenerateContentConfig(temperature=1.0, response_mime_type="application/json"))
            arr = json.loads(r.text)
            out = [re.sub(r"\s+", " ", a).strip() for a in arr if isinstance(a, str) and len(a.split()) >= 12]
            if out and not any(re.search(r"[\u0900-\u097F\u0600-\u06FF]", a) for a in out):   # no Devanagari/Urdu script
                return out
        except Exception as e:  # rate limit, malformed JSON, transient network
            time.sleep(2 * (k + 1))
            last = type(e).__name__
    return []


def build(split: str) -> list[dict]:
    from google import genai
    from google.genai import types

    from scamshield.agent.tts import _use_os_trust_store
    from scamshield.config import get_settings
    _use_os_trust_store()
    client = genai.Client(api_key=get_settings().gemini_api_key.strip(), http_options=types.HttpOptions(timeout=120000))
    model, pf = (MODEL_B, prompt_b) if split == "test2" else (MODEL_A, prompt_direct if split == "train_direct" else prompt_a)
    # (label, type, desc, lang, n_per_call, calls)
    plan = []
    if split == "train":
        plan += [("scam", t, d, "en", 10, 4) for t, d in SCAM.items()] + [("scam", t, SCAM[t], "hi", 9, 3) for t in SCAM_HI]
        plan += [("legit", t, d, "en", 5, 1) for t, d in LEGIT.items()] + [("legit", t, LEGIT[t], "hi", 4, 1) for t in LEGIT_HI]
    elif split == "train_direct":
        plan += [("scam", t, d, "en", 10, 2) for t, d in SCAM.items()] + [("scam", t, SCAM[t], "hi", 8, 1) for t in SCAM_HI]
    elif split == "dev":
        plan += [("scam", t, d, "en", 12, 1) for t, d in SCAM.items()] + [("scam", t, SCAM[t], "hi", 8, 1) for t in SCAM_HI]
        plan += [("legit", t, d, "en", 2, 1) for t, d in LEGIT.items()] + [("legit", t, LEGIT[t], "hi", 3, 1) for t in LEGIT_HI]
    else:  # test2
        plan += [("scam", t, d, "en", 12, 1) for t, d in SCAM.items()] + [("scam", t, SCAM[t], "hi", 8, 1) for t in SCAM_HI]
        plan += [("legit", t, d, "en", 2, 1) for t, d in LEGIT.items()] + [("legit", t, LEGIT[t], "hi", 3, 1) for t in LEGIT_HI]
    seed = {"train": 11, "train_direct": 44, "dev": 22, "test2": 33}[split]
    jobs = []
    for i, (label, t, d, lang, n, calls) in enumerate(plan):
        for c in range(calls):
            rng = random.Random(seed * 1000 + i * 10 + c)
            jobs.append((label, t, lang, pf(label, d, n, lang, rng)))
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda j: call(client, types, model, j[3]), jobs))
    items, seen = [], set()
    for (label, t, lang, _), texts in zip(jobs, results):
        for text in texts:
            key = re.sub(r"\W+", " ", text.lower())[:120]
            if key in seen:
                continue
            seen.add(key)
            items.append({"label": label, "type": t, "lang": lang, "text": text, "split": split, "model": model})
    for i, it in enumerate(items):
        it["id"] = f"{split}_{it['label']}_{it['type']}_{it['lang']}_{i:03d}"
    failed = sum(1 for r in results if not r)
    print(f"{split}: {len(items)} scripts from {len(jobs)} calls ({failed} failed) with {model}", flush=True)
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["train", "train_direct", "dev", "test2"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ([args.only] if args.only else ["train", "dev", "test2"]):
        p = OUT / f"{split}.json"
        if p.exists():
            print(f"{split}: exists, skipping (delete the file to regenerate)")
            continue
        items = build(split)
        p.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {p}  sha256={hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
