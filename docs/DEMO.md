# Demo runbook (5 minutes)

Follows `docs/09_VIVA_AND_DEMO.md`. Clip choices below are based on the eval results (docs/RESULTS.md) so that each step shows a behaviour the system reliably has, plus one honest failure.

## Before the session
- [ ] `python scripts/run_demo.py` on the demo laptop; status shows **ready** and all models loaded.
- [ ] Results tab shows the dev and test splits.
- [ ] Headphones on; volume and mic level tested in the room.
- [ ] Network test: one agent question answered by the LLM. If the venue network is bad, restart with `python scripts/run_demo.py --offline` (cached spoken warnings + grounded template answers + browser speech still work; Whisper runs locally).
- [ ] Fallback video recorded (see below) and on the desktop.

## Script
1. **(30 s) Problem + architecture.** `docs/architecture.svg`.
2. **(60 s) Upload / replayed scam.** Untick *as live call*, pick **Scam: KYC/OTP over a GSM phone line**. Gauge jumps to SCAM; green match box on the spectrogram. Pipeline tab: **offset-histogram spike** ("hundreds of hashes agree on one time offset: this is a known recording"), stage timings.
3. **(45 s) Legit call that says OTP.** **Legit: bank debit alert that mentions PIN/OTP** → SAFE; transcript shows *PIN* / *OTP* struck through as safe context ("we will never ask for your PIN or OTP").
4. **(90 s) Live.** Tick *as live call*, pick **Scam: digital arrest, different voice** (not in the fingerprint library): risk rises safe → suspicious → SCAM around 10 s, warning is spoken. Then (if the room allows) **● Live mic** with a teammate reading a scam in their own words.
5. **(45 s) Agent.** Hold 🎤: "Why do you think it's a scam?", then "What should I do?". Optional: "What's the weather?" → it stays in scope.
6. **(30 s) Results + honesty.** Results tab: test-split numbers, including the Hinglish failure (British-accent Hinglish defeats Whisper: misses and false alarms), and live words-only latency (median 8.2 s). Limitations + deployment path.

## Recording the fallback video
Windows: `Win + Alt + R` (Xbox Game Bar) records the browser window with sound; run steps 2–5 once. Keep it under 3 minutes.

## If something fails during the demo
| Problem | Do this |
|---|---|
| Mic permission denied / no sound | use *as live call* with a sample (same live code path) |
| Agent replies slowly or errors | it falls back automatically; or restart with `--offline` |
| Warning audio does not play | browser speech is used automatically; check the 🔊 toggle |
| Server crashed | `python scripts/run_demo.py` again (models reload in ~20 s) |
| Everything | play the fallback video |
