# Vishwas: Backend Handoff (full context for a new Claude session)

**Project:** Smart India Hackathon 2026, Problem Statement **26079**, "AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts" (Ministry of Earth Sciences / NCMRWF). Team: PowerPuff Squad. Product name: **Vishwas**.
**Repo:** https://github.com/Tanish-Shitanshu/sih2026-forecast-bust-detection, branch **`main`** (everything below is merged there).
**Date of this handoff:** 2026-09-29.
**Your job:** build the **backend** (`backend/` folder), then wire the **frontend** to it. The data and ML workstreams are finished, trained, tested and merged.

---

## 0. Instructions to Claude (read first)

- Read this whole file, then read these in the repo: `ml/BACKEND_INTEGRATION.md` (the precise ML ↔ backend contract), `ml/README.md` (ML details, metrics, limitations), `ml/serve.py` (a working reference API server), and the **API page inside `frontend/index.html`** (search for `var API=[`). **That API page is the authoritative contract** for routes, roles and response shapes.
- **Critically evaluate this plan before building.** If you see a better approach, say so explicitly and explain why. If something is ambiguous, ask rather than guessing. This is a hackathon with judges who ask hard questions, so prefer honest and defensible over flashy.
- **Don't modify `ml/` or `data/` internals.** Import and call them. If you truly need an ML change (a new field, a bug), describe it and ask first; the ML owner (Mohit) will make it. Don't change the bust definition, the model, or the response shapes the ML service returns.
- Work on a new branch (e.g. `backend-v1`), not directly on `main`. Open a PR when done.
- The machine is **Windows**. Use the existing Python venv at `ml/.venv` or create `backend/.venv`; `ml/requirements.txt` has the pinned ML dependencies.

---

## 1. What the product does (in one paragraph)

NCMRWF issues medium-range rainfall forecasts (Day 1–10 ahead). Some of them "bust" (turn out badly wrong). Vishwas takes an already-issued forecast and, for each of India's **33 meteorological subdivisions** × **lead day 1–10**, gives:
- a **calibrated bust probability** (a stated 40% really means about 40%);
- a **warning tier** (Green/Yellow/Orange/Red) with a recommended action;
- a **plain-language explanation** (SHAP-based);
- the **closest historical analog**;
- a **model self-confidence** score (how much precedent the model has).

All of this comes *before* the outcome is known. It does not forecast weather; it judges how far to trust an existing forecast. Duty forecasters record what actually happened, seniors approve those entries, and approved entries recalibrate the model.

**Locked bust definition (do not change):**
```
bust = |forecast_rain − observed_rain| ≥ max(25 mm, subdivision's 95th-percentile error
        for that lead day, fit on training years only)
       OR rain/no-rain category call was wrong (2.5 mm threshold, IMD rainy-day line)
trigger_reason ∈ {magnitude, category, both}
```
Thresholds are fit **separately per forecast source**, never shared.

---

## 2. Repository layout (`main`)

```
frontend/index.html      Single-file vanilla-JS portal (v6), hash routing. Currently runs on BUILT-IN FAKE DATA.
                         Its "API" page (var API=[...]) documents all 13 endpoints, roles and sample responses.
prototypes/              Old UI prototypes (ignore)
data/                    Data engineering (Tanish): GEFS+ERA5+IMD pipeline, backup track
  processed/bust_dataset.parquet   602,580 rows, 2015–2019 (GEFS comparison dataset)
  README.md
ml/                      ML (Mohit): complete
  vishwas_ml/            The package. service.py = what the backend calls
  serve.py               Reference FastAPI server on the frontend's /api/v1 paths
  BACKEND_INTEGRATION.md Endpoint-to-method map, roles, errors, recalibration, backend responsibilities
  README.md              Full ML write-up, metrics, limitations, demo event
  config.json            Forecast-source profiles (ncmrwf default, gefs, synthetic)
  data/ncmrwf_pairs.parquet   91,080 rows, real NCMRWF forecasts vs IMD, 1993–2015 (committed)
  data/subdivisions.json      The 33 subdivisions, tiers, trust levels, events (synced from frontend)
  models/ncmrwf/ | gefs/ | synthetic/   Trained bundles (committed)
  scripts/               train.py, evaluate.py, demo_api.py, recalibrate.py, build_s2s_pairs.py, ...
  tests/                 44 tests (pytest)
backend/                 DOES NOT EXIST YET: this is what you build
```

---

## 3. Data and models (what powers the answers)

### Forecast sources (two independent models)
| | **NCMRWF (primary, default)** | GEFS (backup/comparison) |
|---|---|---|
| Forecast | NCMRWF's own **S2S reforecast** (Unified Model coupled system, ~60 km), from rds.ncmrwf.gov.in | NOAA GEFSv12 reforecast (control member) |
| Years | **1993–2015**, 12 runs/year (init on the 1st of each month) | 2015–2019, daily runs |
| Rows | 91,080 (276 runs × 33 subdivisions × 10 leads) | 602,580 |
| Truth | IMD gridded rainfall 0.25°, averaged to subdivisions | IMD, same |
| Weather features | NCMRWF's own MSLP, surface pressure, 10 m winds + IMD previous-day rain | ERA5 (previous day, leak-free) |
| Test years | 2013–2015 | 2019 |
| **Test PR-AUC** | **0.534** [95% CI 0.499–0.578] | 0.509 [0.495–0.523] |
| Forecast-amount-only baseline | 0.452 | 0.467 |
| Base rate (no-skill PR-AUC) | 0.166 | 0.167 |
| Model folder | `ml/models/ncmrwf/` | `ml/models/gefs/` |

- **NCMRWF is the product default.** The problem statement is about trusting *NCMRWF's* forecasts. GEFS exists only as a comparison.
- **Honest weaknesses** (the UI copy mustn't overclaim):
  - no gain over the raw forecast at Days 1–3;
  - catches only ~30% of big-miss (magnitude) busts;
  - weakest on narrow coastal subdivisions (Coastal Karnataka, Kerala);
  - the NCMRWF archive ends in 2015.

### Critical consequence for the backend: the demo is a historical replay
There is **no live 2026 NCMRWF feed**. The archive ends 2015-12-01. So:
- Every ML call takes a **`cycle`** (forecast issue date). The backend and frontend must let the user **pick a cycle** from `GET /api/v1/cycles` rather than assuming "today".
- **Suggested demo default: `2015-12-01`** (Chennai floods), with the storyline "here's what Vishwas would have flagged before the event".
  - Honest result: Tamil Nadu Day 1 was **missed** (11%, Green; the forecast said 12 mm and 41 mm fell subdivision-wide).
  - Day 4 was flagged **Orange (40%)** and Day 5 **Yellow**; both did bust.
  - Present it as it is.
- The frontend currently hardcodes the text "00Z cycle of 28 September 2026". That must become the selected cycle.

---

## 4. How the backend calls the ML (the contract)

```python
import sys; sys.path.insert(0, "<repo>/ml")
from vishwas_ml.config import load_config
from vishwas_ml.service import VishwasService

svc = VishwasService(cfg=load_config(source="ncmrwf"))   # load ONCE at startup (~3 s); keep one per process
```
Each method returns a dict **already in the exact shape of the frontend API page**. Pass it through unchanged.

| Frontend route | Allowed roles | ML call |
|---|---|---|
| `GET /api/v1/confidence-map?cycle&lead_day` | all | `svc.confidence_map(lead_day, cycle)` |
| `GET /api/v1/bust-probability?subdivision&lead_day` (+`cycle`) | all | `svc.bust_probability(sub, lead_day, cycle)` |
| `GET /api/v1/subdivisions/{code}/explain?lead_day` (+`cycle`) | all | `svc.explain(code, lead_day, cycle)` |
| `GET /api/v1/model-trust?lead_day` (+`cycle`) | all | `svc.model_trust(lead_day=..., cycle=...)` |
| `POST /api/v1/outcomes` | duty, senior | `vishwas_ml.feedback.validate_outcome(body)` + **your DB**; fill `predicted_bust_probability` from `svc.bust_probability(...)` |
| `GET /api/v1/outcomes?status&subdivision` | all | **your DB** |
| `POST /api/v1/outcomes/{id}/review` | senior (not own entry) | **your DB** |
| `GET/POST /api/v1/users`, `PATCH /api/v1/users/{id}` | admin (can't change/deactivate self) | **your DB** |
| `PUT /api/v1/settings/bust-definition` | admin | **your DB**; applied at next retrain |
| `POST /api/v1/retraining` | admin | background job: `vishwas_ml.pipeline.train_pipeline(cfg)` with `cfg["bust"]` overridden from settings; then reload `VishwasService` |
| `GET /api/v1/audit?limit` | admin | **your DB** |

Useful extras, not in the frontend yet:
- `GET /api/v1/cycles`: `svc.cycles()`
- `GET /api/v1/analogs`: `svc.analogs(sub, lead_day, cycle, k)`
- `GET /api/v1/action`: `svc.action(...)`
- `GET /api/v1/model-trust?subdivision=`: `svc.model_trust(subdivision=...)`

**Details you'll hit:**
- **Errors:** `ValueError` → 422 (bad subdivision, lead day outside 1–10); `LookupError` → 404 (unknown cycle). See the exception handlers in `ml/serve.py`.
- **Subdivision codes contain `/` and `&`** (`TN/PY`, `HR/DL`, `J&K`, `SAU/KCH`, `KNK/GA`, `SIK/NWB`). Use a `{code:path}` route and URL-encode on the client (`TN%2FPY`, `J%26K`). The service also accepts full subdivision names.
- `cycle` accepts `2015-12-01` or `2015-12-01T00Z`; omitted = latest available (2015-12-01 for NCMRWF).
- **Performance:** first call for a cycle ~1 s (scores all 330 rows), then cached. Retraining takes ~30 s (NCMRWF) or ~2 min (GEFS). Never run it on the request thread.
- `weather_event` can be `null` even for flagged tiers. The UI should show "no specific event".
- **Recalibration** from approved outcomes: `vishwas_ml.feedback.recalibrate(...)`; the exact snippet is in `ml/BACKEND_INTEGRATION.md` §3.
  - Outcome semantics: `incorrect` = the forecast busted, `correct` = it held, `partial` = ignored.
  - Nothing changes below 50 approved outcomes.
- **Do NOT use `svc.record_outcome`** in production. It writes a dev-only JSONL file.

---

## 5. What the backend must own (ML deliberately doesn't)

1. **Auth + roles.** Four roles, defined in the frontend's `ROLES` object:
   - **duty** (Duty forecaster): records outcomes, can't approve, can't manage users.
   - **senior** (Senior forecaster): records outcomes, approves or rejects *others'* entries, never their own.
   - **admin** (Administrator): manages users and settings, starts retraining, doesn't record or approve outcomes.
   - **observer**: view only.

   The prototype's demo accounts are ids `forecaster` (R. Nair, duty), `senior` (S. Iyer), `admin` (V. Rao) and `observer` (Guest observer), with the shared password `vishwas123` (`var PW`). Seed these for the demo, but hash passwords and issue tokens (the API page says requests carry a token issued at sign-in, and the role in the token decides access).
2. **Outcomes table + review workflow:** pending → approved/rejected; 409 if not pending; 403 on self-review. The frontend's `FB0` array has 16 sample log entries you can seed.
3. **Users** (add, change role, deactivate, reactivate; admins can't deactivate themselves; user id format: 3–20 lowercase letters/digits, starting with a letter), **audit log** of admin actions, **settings** (`error_threshold_mm` 1–500, default 25; `regional_percentile` 50–99, default 95).
4. **Retraining job** status (the frontend shows stages: "Preparing data", "Training the model", "Calibrating probabilities", "Validating on held-out years").
5. **CORS** for the frontend (it's a static file; `ml/serve.py` allows all origins for dev).

Storage choice is yours. SQLite is plenty for the hackathon demo. Justify the choice if you pick something heavier.

---

## 6. Frontend wiring (after the backend works)

`frontend/index.html` computes everything client-side from a fake `calc(day)` function (a seeded pseudo-random generator) and fake arrays (`FACTORS`, `ANALOGS`, `REASONS`, `FB0`, `mkUsers()`). To wire it up:
1. Replace `calc(day)` consumers with `GET /api/v1/confidence-map` (map, tables, region summaries), `bust-probability`, `explain` (factors + closest analog) and `model-trust`.
2. Add a **cycle picker** fed by `/api/v1/cycles`, defaulting to `2015-12-01`, and replace the hardcoded "28 September 2026" copy.
3. Replace sign-in, the feedback log, approvals, users, settings, retraining and audit with real API calls.
4. Keep the look and accessibility. Keep the API page, but its samples can now come from the live API.
5. Response field names already match what the frontend's API page documents: `bust_probability`, `forecast_confidence`, `level`, `band`, `weather_event`, `recommended_action`, `model_self_confidence`, `factors[{feature, contribution_percent, direction}]`, `closest_analog{date, description, outcome}`.

---

## 7. Run and verify what exists

```bash
cd ml
.venv/Scripts/python -m pip install -r requirements.txt          # if the venv is fresh
.venv/Scripts/python -m pytest -q tests                          # 44 tests, ~2 min
.venv/Scripts/python scripts/demo_api.py --cycle 2015-12-01 --subdivision TN/PY --lead 4
.venv/Scripts/python -m uvicorn serve:app --port 8000            # then open http://localhost:8000/docs
```
`VISHWAS_SOURCE=gefs` (env var) or `load_config(source="gefs")` switches to the comparison model.

---

## 8. Known gotchas (already solved; don't undo them)

- **`ml/.gitattributes` marks model files `-text`.** On Windows, git's CRLF conversion corrupts LightGBM's `booster.txt` ("Model format error"). If you ever see that error, re-checkout the models from git.
- **Each forecast source has its own thresholds and model folder.** Training refuses to overwrite another source's bundle.
- **GEFS data already has ERA5 shifted to the previous day** at source (leak fix), so `era5_shift_days` stays 0 for GEFS.
- Every ML feature is known at forecast issue time, and tests guard against leakage (`ml/tests/test_leakage.py`).

---

## 9. Team

- **Mohit:** ML owner (this handoff). Questions about ML behaviour or fields go to him.
- **Tanish:** data engineering (`data/`, GEFS track).
- Backend + frontend integration: **you**, starting now.

**Report back with:** your backend plan (stack, DB, auth approach, folder layout) *before* writing much code, anything in this plan you'd change, and questions.
