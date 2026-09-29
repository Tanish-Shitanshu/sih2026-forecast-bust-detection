# Vishwas backend

FastAPI + SQLite, sitting directly on top of `ml/vishwas_ml/service.py`. It owns
everything `ml/BACKEND_INTEGRATION.md` asked it to own: auth, the 4 roles,
outcomes + review workflow, users, audit log, settings, and retraining as a
background job. It does not modify `ml/` or `data/` -- it imports and calls
`VishwasService` and `train_pipeline` unchanged.

## Run it

Reuses `ml/.venv` (it already has the full pandas/lightgbm/shap stack that
`VishwasService` needs -- no separate `backend/.venv`).

```bash
cd ml && ../ml/.venv/bin/pip install -r ../backend/requirements.txt   # pyjwt, bcrypt
cd ../backend
../ml/.venv/bin/python -m uvicorn app:app --port 8000
```

First run creates `backend/vishwas.db` (gitignored) and seeds it from
`ml/data/subdivisions.json` and the frontend's own demo users/outcomes. Delete
the file to reseed from scratch.

`GET /health` should show `"latest_cycle":"2015-12-01T00Z"`.

## Serve the frontend against it

```bash
cd frontend && python3 -m http.server 5500
```

Open `http://localhost:5500/index.html`. The backend's CORS is permissive
(`allow_origins=["*"]`, token-based auth, no cookies), so this works from any
origin/port -- verified in an actual browser session with the frontend on
:5500 and the backend on :8000, not just a same-origin dev setup.

Sign in with any of the demo accounts on the login page (`forecaster`,
`senior`, `admin`, `observer`), password `vishwas123` for all of them (plus 4
more duty-role accounts seeded from the frontend's `mkUsers()`).

## Tests

```bash
cd backend && ../ml/.venv/bin/python -m pytest -q tests/test_backend.py
```

37 tests, real `VishwasService` + real SQLite (temp file) via FastAPI's
`TestClient`, nothing mocked. Covers every route, every role boundary
(including the negative cases: duty can't approve, admin can't deactivate
itself, senior can't review their own entry, observer can't POST anything),
the two URL-encoded subdivision-code edge cases (`TN%2FPY`, `J%26K`), 422/404
error semantics, retraining running off the request thread with working
status polling, and CORS preflight.

`backend/tests/verify_live.py` is the same checklist run manually against a
live server instead of TestClient -- useful for a from-scratch sanity check,
not part of the committed suite.

**Retraining never touches `ml/models/`.** A run writes its bundle to
`backend/runtime_models/<source>/` (gitignored; override with
`VISHWAS_RUNTIME_MODELS`), staged in a temporary folder and swapped in only
when complete. The service then loads that bundle in preference to the
committed baseline in `ml/models/<source>/`. Delete the runtime folder to go
back to the baseline. The tests point `VISHWAS_RUNTIME_MODELS` at a temp
directory and assert with `git status` that `ml/models/` is unchanged.
(Earlier versions retrained in place, so every test run rewrote committed
model files.)

**What a retraining run does:**
1. `train_pipeline` with the admin's saved bust definition.
2. Recalibration from **approved** forecaster outcomes (`vishwas_ml.feedback.recalibrate`). It's applied only when there are at least `feedback.min_outcomes` (50) usable approved outcomes; otherwise the job result says `"status": "skipped: not enough approved outcomes"`.

The job's `metrics.recalibration` field reports which happened. (Earlier
versions reported `approved_outcomes_used` without passing the outcomes to
the model at all.)

## Design choices

- **SQLite**, per the ML handoff's own suggestion ("SQLite is plenty for the
  hackathon demo"). Five small tables (users, outcomes, audit, settings,
  retraining_jobs), no concurrent-write load beyond a single-process demo.
- **JWT bearer tokens** (`PyJWT`) + **bcrypt** password hashes. The frontend's
  API page already assumed "requests carry a token issued at sign-in" but
  didn't document a login route (its original prototype checked the password
  client-side against a hardcoded string) -- `POST /api/v1/auth/login` is the
  minimal real addition that makes that assumption true.
- **Retraining runs on a daemon `threading.Thread`**, not the request thread.
  `POST /api/v1/retraining` returns in under 20ms in practice (measured both
  via pytest and in the browser); `GET /api/v1/retraining/{job_id}` polls a
  DB-backed status row (`queued` -> `running` -> `done`/`failed`). A
  process-global lock returns 409 on a concurrent second attempt.
- **`service_registry.py`** holds one `VishwasService` per process and
  reloads it after a successful retraining run, so the next prediction call
  uses the freshly trained model without a restart.

## Two additions beyond the original 13-route contract

Both were necessary for the frontend to actually work, not scope creep:

- `POST /api/v1/auth/login` -- see above.
- `GET /api/v1/retraining/{job_id}` -- status polling for the background job;
  the original contract only specified the `POST` that starts one.
- `GET /api/v1/settings/bust-definition` -- the contract only specified the
  `PUT`. Added so the admin settings panel shows the value actually persisted
  in the database instead of a hardcoded guess that could drift from it
  across sessions or other admins' changes.

All three are purely additive, same auth model as their siblings, and don't
touch the locked bust definition, the ML response shapes, or anything in
`ml/`.

## What "wired to the frontend" covers

The signed-in workspace (overview, dashboard, regions, Model Trust, feedback
log, administration) is driven entirely by real backend calls: sign-in,
the cycle picker (`GET /api/v1/cycles`, defaulting to `2015-12-01`, the last
cycle in the NCMRWF archive and the Chennai floods date used throughout
`ml/README.md`), the bust-risk/confidence maps, the subdivision drawer
(factors + closest analog via `/explain`), Model Trust, the outcomes log
(record/list/approve/reject), and admin (users, settings, audit, retraining).

The **public landing page** (pre-login) and the **API reference page**
(`#/api`) intentionally keep using the original illustrative `calc(day)`
sample generator -- both already self-label as illustrative in their own
copy ("Sample view", "Sample bust-risk map ... with illustrative data",
"Illustrative data, not connected to a live feed", "Sample response"), so
this isn't a gap, it's the same honesty the rest of the UI already commits
to: real data only where the UI already claims it, and clearly-labelled
illustrative data everywhere else.

## Known limitations (stated here, not smoothed over)

- The model itself: at lead days 1-3 it is no better than the forecast amount
  alone (it still beats climatology at every lead), it catches roughly 30% of
  magnitude busts, and it is weakest on Coastal Karnataka and Kerala.
  See `ml/README.md` for the full honest accounting -- the backend passes
  every prediction through unchanged, it doesn't add or remove caveats.
  Demo copy in the frontend states this plainly rather than smoothing it
  over.
- User roles are read from the database on every request, not from the
  token, so an admin's role change or deactivation applies immediately to
  tokens already issued.
- `vishwas.db` is a single SQLite file with no backup/migration story --
  fine for a hackathon demo, not for production.
- Login tokens are 12-hour JWTs signed with a hardcoded demo secret
  (`VISHWAS_JWT_SECRET` env var overrides it) -- not meant to survive contact
  with a real deployment.
