# Vishwas ML pipeline

Predicts, for each of the 33 IMD subdivisions and each lead day 1–10, the calibrated probability that an issued NCMRWF/GFS rainfall forecast will bust. It also explains the probability in plain language, finds historical analogs, and scores the model's own reliability.

**Status:** complete pipeline, trained on **synthetic data** that matches the real schema. Swapping in the real table takes one line (see below).

## Quick start

```bash
cd ml
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python scripts/make_synthetic.py     # regenerate ml/data/synthetic_pairs.parquet (deterministic)
.venv/Scripts/python scripts/train.py              # train + calibrate + evaluate + save ml/models/ (~25 s)
.venv/Scripts/python scripts/evaluate.py           # per-lead / per-subdivision tables from the saved bundle
.venv/Scripts/python scripts/demo_api.py --cycle 2023-08-13   # every endpoint's JSON for one cycle
.venv/Scripts/python -m pytest -q tests            # 25 tests (~45 s)
```

## Swapping in the real dataset

1. Put the `data/` workstream's table (`.parquet` or `.csv`, same columns) anywhere, then change **one line** in `ml/config.json`:
   ```json
   "data_path": "../data/processed/pairs.parquet"
   ```
   (or pass `--data path` to `train.py` without editing the config).
2. Run `scripts/train.py`. It validates the table before training:
   - unknown or missing subdivision codes, `lead_day` outside 1–10, duplicate keys and missing columns stop the run;
   - `error ≠ forecast − observed`, inconsistent `observed_rain` for the same valid date, and `trigger_reason` / `is_bust` mismatches are reported as warnings.
3. It recomputes `is_bust` with the locked formula and training-year thresholds, and logs its agreement with the provided `is_bust` column. On the synthetic data the agreement is 1.0000.

If the frontend's subdivision list ever changes, run `scripts/sync_subdivisions.py`; a test fails until you do.

## What is implemented

| Component | Where | Notes |
|---|---|---|
| Schema, loader, validator | `vishwas_ml/schema.py` | Exact 15-column schema from the brief |
| Locked bust label | `vishwas_ml/labels.py` | `|f−o| ≥ max(25 mm, P95 of |error| per subdivision × lead day, training years only)` **OR** rain/no-rain call wrong (2.5 mm, IMD rainy-day line). `trigger_reason` = magnitude / category / both |
| Split | `vishwas_ml/splits.py` | By year: last = test, second-last = calibration, rest = training |
| Features (29) | `vishwas_ml/features.py` | All known at issue time: forecast amount vs climatology, distance to the rain line, run-to-run change, consistency across leads, regional contrast, ERA5 anomalies (z-scored per subdivision × month, so ERA5 units don't matter), pressure tendency, historical bust rate (**out-of-fold** for training rows) |
| Classifier | `vishwas_ml/model.py` | LightGBM, early stopping on the calibration year |
| Calibration | `vishwas_ml/model.py` | Isotonic regression fit on the calibration year only; output clipped to 1–99 % |
| Explainability | `vishwas_ml/explain.py` | Exact TreeSHAP (LightGBM `pred_contrib`, checked against `shap` in tests), grouped into 12 factors and phrased as sentences, e.g. "Forecast of 2.8 mm sits near the 2.5 mm rain/no-rain line, so a small error flips the call" |
| Analog matcher | `vishwas_ml/analogs.py` | k-NN per lead day in a SHAP-weighted feature space; prefers the same region; **only returns cases whose outcome was known before the query's issue date** |
| Model trust | `vishwas_ml/trust.py` | Four flags: sparse history, unfamiliar pattern (analog-distance percentile), inputs outside training range, model vs analog disagreement. Levels use the frontend's thresholds |
| Weather-event tag | `vishwas_ml/events.py` | Rules engine over the frontend's event names and per-subdivision candidate sets. Heuristic; `null` when nothing fits |
| Action tier | `vishwas_ml/actions.py` | Rules engine: probability band → frontend `TIERS` level, band and action text |
| Feedback → recalibration | `vishwas_ml/feedback.py`, `scripts/recalibrate.py` | Approved outcomes only; a second isotonic layer anchored on the calibration year so a few entries can't swing it; skipped below 50 outcomes |
| Service layer | `vishwas_ml/service.py` | One method per endpoint, returning the frontend's response shapes |

## API mapping (for backend)

The frontend's API page (`frontend/index.html`, `var API`) is authoritative. Its paths differ from the handoff brief, so both are covered:

| Frontend endpoint | Brief's name | Service method | Shape |
|---|---|---|---|
| `GET /api/v1/confidence-map?cycle&lead_day` | `/confidence-map` | `confidence_map(lead_day, cycle)` | exact frontend keys |
| `GET /api/v1/bust-probability?subdivision&lead_day` | `/bust-probability` | `bust_probability(sub, lead_day, cycle)` | exact |
| `GET /api/v1/subdivisions/{code}/explain?lead_day` | `/explain` | `explain(code, lead_day, cycle)` | exact (`factors` + `closest_analog`) |
| `GET /api/v1/model-trust?lead_day` | `/model-trust` | `model_trust(lead_day=…)` | exact, least confident first |
| — | `/model-trust?region=` | `model_trust(subdivision=…)` | same item fields + `lead_day`, `flags` |
| — | `/analogs` | `analogs(sub, lead_day, cycle)` | new; fields named like `closest_analog` |
| — | `/action` | `action(sub, lead_day, cycle)` | new; tier fields + `trust_note` |
| `POST /api/v1/outcomes` | `/feedback` | `record_outcome(...)` (dev store) + `feedback.validate_outcome` | exact |
| `POST /api/v1/retraining` | — | `pipeline.train_pipeline()` | returns metrics |
| `PUT /api/v1/settings/bust-definition` | — | write `bust.min_error_mm` / `bust.percentile` in `config.json`, then retrain | — |

`cycle` accepts `2026-09-28` or `2026-09-28T00Z`; omitted = latest issue date in the table. Bad inputs raise `ValueError` (→ 422) and an unknown cycle raises `LookupError` (→ 404). Scoring one cycle takes about 1 s and is cached per cycle.

```python
from vishwas_ml.service import VishwasService
svc = VishwasService()                      # loads ml/models and the table in config.json
svc.confidence_map(lead_day=3)
```

For live use, construct `VishwasService(history=df)` with the latest rows. The table needs no outcome columns, but it must include the previous ~10 days of issues, which the run-to-run features use.

## Current metrics (synthetic data, test year 2023, 39,930 rows)

Bust base rate is 18.9%, rising from 9.9% on Day 1 to 25.7% on Day 10. "Alert" means Orange or Red (p ≥ 0.38).

| | PR-AUC | ROC-AUC | Brier skill vs clim. | ECE | Precision @ alert | Recall @ alert | Accuracy @ 0.5 |
|---|---|---|---|---|---|---|---|
| **Model** | **0.611** | 0.889 | **0.267** | **0.007** | 0.590 | **0.664** | 0.854 |
| Forecast amount + lead only | 0.578 | 0.858 | 0.218 | 0.008 | 0.608 | 0.571 | 0.851 |
| Climatology (historical rate) | 0.381 | 0.741 | 0 | 0.035 | 0.467 | 0.133 | 0.811 |

The accuracy column shows why accuracy isn't the headline metric: climatology scores 81% while catching 13% of busts. Per-lead and per-subdivision precision, recall and PR-AUC are in `models/evaluation/per_lead.csv` and `per_subdivision.csv`. Reliability and PR-AUC by lead are plotted in `evaluation.png`. The model beats both baselines at every lead day. Calibration error stays at or below 0.02 at every lead.

**How to read these numbers:** they measure the pipeline, not real-world skill. In the synthetic data most busts (≈90%) are pure rain/no-rain flips, so the forecast amount carries most of the signal. ERA5 moisture and pressure anomalies add a real but modest lift. The real data will decide the actual numbers and feature ranking.

## Open questions for the data team

1. **ERA5 timing.** Are the ERA5 columns the state at the issue date (the same for all 10 lead days) or at the valid date? Valid-date reanalysis isn't available when the forecast is issued, so using it as-is would leak the outcome. The pipeline auto-detects which it is (`era5_timing: "auto"`). For valid-date data, it reads the issue-date state from earlier rows whose valid date equals the issue date, so both formats work. Issue-date values are preferred.
2. **`total_precipitation` window.** The 24 h accumulation should end at issue time. A calendar-day total for the issue date includes hours after a 00Z issue.
3. **Rain/no-rain line.** 2.5 mm (IMD rainy day) is assumed; change it in `config.json → bust.rain_threshold_mm`. With a hard line, 2.4 mm vs 2.6 mm counts as a bust. This is expected to dominate the real bust rate, as it does here.
4. **Thresholds need training years.** `is_bust` in the shared table needs P95 thresholds from "training years". ML recomputes labels with its own split anyway, so the column is only cross-checked.

## Assumptions to confirm

- **Feedback outcome meaning:** `incorrect` = the forecast busted (label 1), `correct` = it held, `partial` = excluded. This is inferred from the frontend's sample log, where high predicted probabilities pair with "incorrect".
- The weather-event tag and model-trust scores are transparent heuristics, not trained models. The trust `min_busts` (20) is set for the 2-year synthetic training set and should be revisited with real data.
- 33 subdivisions exactly as the frontend lists them (IMD's official list has 36; the frontend merges some, e.g. Uttarakhand isn't listed).

## Not yet done / stubbed

- `record_outcome` writes to a local dev file (`data/feedback_dev.jsonl`); persistence and the pending → approved review flow belong in the backend.
- No live NCMRWF ingestion. The service scores whatever rows it's given.
- Hyperparameters are defaults, not tuned (training takes seconds; tune once real data lands).
- The production bundle is the evaluated one (trained on the training years only). Once real data is in, consider refitting on all years except the calibration year.

## Layout

```
ml/
  config.json            data path, bust definition, split, model params
  data/                  subdivisions.json (synced from frontend), synthetic_pairs.parquet
  models/                saved bundle + evaluation/ (metrics.json, per_lead.csv, per_subdivision.csv, evaluation.png)
  scripts/               sync_subdivisions, make_synthetic, train, evaluate, recalibrate, demo_api
  vishwas_ml/            the package
  tests/                 labels, leakage, contract, model tests
```
