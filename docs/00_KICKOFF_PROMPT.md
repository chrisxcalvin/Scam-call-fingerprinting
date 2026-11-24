# 00 — Kickoff prompt (paste into Claude Code at the repo root)

```
You are building "ScamShield Voice" for a university speech-processing project. Before writing any code:
1. Read CLAUDE.md, then docs/01_PROBLEM_AND_SCOPE.md through docs/07_PHASES.md in order, and docs/09_VIVA_AND_DEMO.md.
2. Summarise back to me in ≤ 15 lines: the goal, the 3 detection layers, the module ownership, and the Phase 0 acceptance criteria. List any ambiguity or conflict you found in the docs — do not silently resolve it.

Then execute Phase 0 ONLY:
- Scaffold the repo exactly per docs/02_ARCHITECTURE.md (src/scamshield/... packages with __init__.py, stubs with full type hints and docstrings stating the math/behaviour, NotImplementedError bodies).
- Implement config.py (all DSP constants + paths + env settings via pydantic-settings) and schemas.py exactly from docs/03_API_CONTRACT.md.
- Write scripts/check_data.py per docs/06_DATA_AND_EVAL.md (validate data/ layout, manifests, 16 kHz mono, label counts; clear summary output).
- Create tests/ with one xfail-marked test per stub plus a passing test for schemas round-trip.
- FastAPI app with GET /api/health and static serving of web/ (web/index.html can be a placeholder).
- Update .gitignore if needed. Do not implement any DSP/ML logic yet.

Rules: follow CLAUDE.md strictly (from-scratch DSP, no fake metrics, no data in git, keep it simple). When Phase 0 acceptance criteria pass, run pytest -q and check_data.py, show me the output, and STOP. Wait for my go-ahead before Phase 1.
```

## How to use with three people
Each person opens the same repo in Claude Code and pastes **their** prompt from `docs/08_MODULE_PROMPTS.md` after Phase 0 is merged. Work on separate branches; only the owner edits their module.
