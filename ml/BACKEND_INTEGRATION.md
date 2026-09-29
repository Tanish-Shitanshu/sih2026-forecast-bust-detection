# Backend integration guide (ML side)

This is for whoever builds `backend/` (FastAPI + DB). The ML side is finished and tested: 44 tests, real NCMRWF model trained. This page says how to call it, what it returns, and what the backend itself must own.

## 1. What the ML package gives you

`ml/vishwas_ml/service.py` → `VishwasService`: **one method per ML endpoint**. Each returns a dict already in the exact shape of the frontend's API page (`frontend/index.html`, `var API`). The backend adds auth, persistence and the admin endpoints around it.

`ml/serve.py` is a **working reference server** with the same routes. You can copy its route functions into `backend/`, or run it as-is during frontend development:

```bash
cd ml
.venv/Scripts/python -m uvicorn serve:app --port 8000     # http://localhost:8000/docs
```

### Setup inside the backend
```python
import sys; sys.path.insert(0, "<repo>/ml")          # or pip install -e later
from vishwas_ml.config import load_config
from vishwas_ml.service import VishwasService

svc = VishwasService(cfg=load_config(source="ncmrwf"))   # load once at startup (~3 s)
```
- Dependencies: `ml/requirements.txt`.
- The **NCMRWF** model (`ml/models/ncmrwf/`) and its data (`ml/data/ncmrwf_pairs.parquet`) are committed, so nothing needs downloading.
- `source="gefs"` loads the comparison model. It needs `data/processed/bust_dataset.parquet` from the data branch (PR #2) to be merged.
- Keep one `VishwasService` per process. It caches each scored cycle (first call ~1 s, then instant).

## 2. Endpoint map

| Frontend route | Roles (from the API page) | Call | Notes |
|---|---|---|---|
| `GET /api/v1/confidence-map?cycle&lead_day` | all | `svc.confidence_map(lead_day, cycle)` | 33 items, frontend's subdivision order |
| `GET /api/v1/bust-probability?subdivision&lead_day` | all | `svc.bust_probability(sub, lead_day, cycle)` | |
| `GET /api/v1/subdivisions/{code}/explain?lead_day` | all | `svc.explain(code, lead_day, cycle)` | codes contain `/` and `&` (`TN/PY`, `J&K`); use a `{code:path}` route and URL-encode |
| `GET /api/v1/model-trust?lead_day` | all | `svc.model_trust(lead_day=..., cycle=...)` | least confident first |
| `POST /api/v1/outcomes` | duty, senior | `vishwas_ml.feedback.validate_outcome(body)` + **your DB insert** | `predicted_bust_probability` comes from `svc.bust_probability(...)` |
| `GET /api/v1/outcomes` | all | your DB | |
| `POST /api/v1/outcomes/{id}/review` | senior | your DB | only approved outcomes feed recalibration |
| `POST /api/v1/retraining` | admin | run `vishwas_ml.pipeline.train_pipeline(cfg)` **as a background job** | ~30 s NCMRWF, ~2 min GEFS; returns metrics. Then reload `VishwasService` |
| `PUT /api/v1/settings/bust-definition` | admin | store it; at the next retrain pass `cfg["bust"]["min_error_mm"]` and `cfg["bust"]["percentile"]` | don't edit `config.json` at runtime; override the dict you pass to `train_pipeline` |
| `GET/POST/PATCH /api/v1/users`, `GET /api/v1/audit` | admin | backend only | no ML involvement |

Extras the frontend doesn't call yet, but that are useful for the demo:

| Route | Call |
|---|---|
| `GET /api/v1/cycles` | `svc.cycles()`, the forecast dates available for replay |
| `GET /api/v1/analogs?subdivision&lead_day&cycle&k` | `svc.analogs(...)` |
| `GET /api/v1/action?subdivision&lead_day&cycle` | `svc.action(...)` |
| `GET /api/v1/model-trust?subdivision=` | `svc.model_trust(subdivision=...)` (all 10 lead days + reliability flags) |

**Errors:** service methods raise `ValueError` for bad input (→ **422**) and `LookupError` for an unknown cycle (→ **404**). `serve.py` shows the exception handlers.

**`cycle`** accepts `"2015-12-01"` or `"2015-12-01T00Z"`; omitted means the latest available run. With NCMRWF the latest is **2015-12-01**. The archive ends in 2015, so the demo **replays historical cycles**, and the frontend should get a cycle picker from `/api/v1/cycles` rather than assuming "today".

## 3. Recalibration from forecaster feedback
```python
from vishwas_ml.feedback import recalibrate
from vishwas_ml.model import BustModel
model = BustModel.load("ml/models/ncmrwf"); model.recalibrator = None
rc, summary = recalibrate(model, approved_outcomes_df, pd.read_parquet("ml/models/ncmrwf/calibration_set.parquet"),
                          cfg["feedback"])
if rc is not None: model.recalibrator = rc; model.save("ml/models/ncmrwf")   # then reload VishwasService
```
- `approved_outcomes_df` needs the columns `predicted_bust_probability, outcome, status`.
- `incorrect` = the forecast busted, `correct` = it held, `partial` is ignored.
- Nothing changes below 50 approved outcomes.

## 4. What the backend must own (ML deliberately does not)
- Sign-in, tokens and the four roles (`duty`, `senior`, `admin`, `observer`) with the permissions in the API page. `serve.py` only reads an `X-Role` header as a stand-in.
- The outcomes table and review flow (pending → approved/rejected; a senior can't review their own entry). `svc.record_outcome` writes a **dev-only** JSONL file; don't use it in production.
- Users, audit log, settings storage.
- Running retraining off the request thread, then reloading the model.

## 5. Frontend wiring checklist
1. Replace the frontend's `calc(day)` illustrative data with `GET /api/v1/confidence-map` (and the other routes above).
2. Add a cycle (date) picker fed by `/api/v1/cycles`. Suggested default for the demo: `2015-12-01` (Chennai floods; see `ml/README.md` for what the model got right and wrong).
3. `weather_event` can be `null` even for Yellow/Orange/Red, when no event rule fits. Show "no specific event" in that case.
4. `model_self_confidence` levels are the frontend's own VIO labels, lowercased (`typical`, `slightly reduced`, `reduced`, `substantially reduced`).

## 6. Sanity check before wiring
```bash
cd ml && .venv/Scripts/python -m pytest -q tests            # 44 tests
.venv/Scripts/python scripts/demo_api.py --cycle 2015-12-01 --subdivision TN/PY --lead 4
```
