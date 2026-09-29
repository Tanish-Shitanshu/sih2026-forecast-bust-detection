# Vishwas ML pipeline

Predicts, for each of the 33 IMD subdivisions and each lead day 1–10, the calibrated probability that an issued medium-range rainfall forecast (NCMRWF, or GEFS as a stand-in) will bust. It also explains the probability in plain language, finds historical analogs, and scores the model's own reliability.

**Status:** complete pipeline, trained on **synthetic data** that matches the real schema. Swapping in the real table takes one line (see below).

## Quick start

```bash
cd ml
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python scripts/make_synthetic.py     # regenerate ml/data/synthetic_pairs.parquet (deterministic)
.venv/Scripts/python scripts/train.py              # train + calibrate + evaluate + save ml/models/ (~25 s)
.venv/Scripts/python scripts/evaluate.py           # per-lead / per-subdivision tables from the saved bundle
.venv/Scripts/python scripts/demo_api.py --cycle 2023-08-13   # every endpoint's JSON for one cycle
.venv/Scripts/python -m pytest -q tests            # 29 tests (~90 s)
```

## Swapping in the real dataset

1. Put the `data/` workstream's table (`.parquet` or `.csv`) anywhere, then change **one line** in `ml/config.json`:
   ```json
   "data_path": "../data/processed/pairs.parquet"
   ```
   (or pass `--data path` to `train.py` without editing the config).
   The loader accepts the data team's column names as they are (`forecast_rain_mm`, `observed_rain_mm`, `error_mm`, `era5_msl_pa`, `era5_sp_pa`, `era5_d2m_k`, `era5_t2m_k`; extra `year` / `subdivision_name` columns are ignored). See `ALIASES` in `vishwas_ml/schema.py`. A test trains on exactly that format.
   - **Required:** `date, subdivision_code, lead_day, forecast_rain, observed_rain`.
   - **Optional:** wind, ERA5 precipitation, `is_bust`, `trigger_reason`, and `forecast_rain_spread` (ensemble std across members). Missing ERA5 variables become NaN features, and labels are always recomputed.
2. Run `scripts/train.py`. It validates the table before training:
   - unknown subdivision codes, `lead_day` outside 1–10, duplicate keys and missing required columns stop the run;
   - `error ≠ forecast − observed`, inconsistent `observed_rain` for the same valid date, and `trigger_reason` / `is_bust` mismatches and absent optional columns are reported as warnings.
3. It recomputes `is_bust` with the locked formula and training-year thresholds, and logs its agreement with the provided `is_bust` column if there is one. On the synthetic data the agreement is 1.0000.

If the frontend's subdivision list ever changes, run `scripts/sync_subdivisions.py`; a test fails until you do.

## What is implemented

| Component | Where | Notes |
|---|---|---|
| Schema, loader, validator | `vishwas_ml/schema.py` | 15-column schema from the brief, plus the data team's column names and an optional ensemble-spread column |
| Locked bust label | `vishwas_ml/labels.py` | `|f−o| ≥ max(25 mm, P95 of |error| per subdivision × lead day, training years only)` **OR** rain/no-rain call wrong (2.5 mm, IMD rainy-day line). `trigger_reason` = magnitude / category / both |
| Split | `vishwas_ml/splits.py` | By year: last = test, second-last = calibration, rest = training |
| Features (31) | `vishwas_ml/features.py` | All known at issue time: forecast amount vs climatology, distance to the rain line, run-to-run change, consistency across leads, ensemble spread (when available), regional contrast, ERA5 anomalies (z-scored per subdivision × month, so ERA5 units don't matter), pressure tendency, historical bust rate (**out-of-fold** for training rows) |
| Classifier | `vishwas_ml/model.py` | LightGBM, early stopping on the calibration year |
| Calibration | `vishwas_ml/model.py` | Isotonic regression fit on the calibration year only; output clipped to 1–99 % |
| Explainability | `vishwas_ml/explain.py` | Exact TreeSHAP (LightGBM `pred_contrib`, checked against `shap` in tests), grouped into 13 factors and phrased as sentences, e.g. "Forecast of 2.8 mm sits near the 2.5 mm rain/no-rain line, so a small error flips the call" |
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

## Forecast source

- **Now:** NOAA GEFS reforecast, 2015–2019, as a stand-in. It's not NCMRWF's model, and the write-up should say so.
- **NCMRWF option found (2026-09-29):** the RDS portal's **"S2S global sub-seasonal to seasonal re forecast, 1993–2015"** (`rds.ncmrwf.gov.in/datasets/s2s-daily`) is NCMRWF's own Unified Model-based coupled system (GC2, N216).
  - Hindcasts are initialized on the 1st, 9th, 17th and 25th of every month, with 6 ensemble members.
  - The download form offers Forecast Day T to T+36, "Total Precipitation Amount", MSLP, surface pressure, 10 m winds, a lat/lon box and NetCDF4.
  - It has both rainfall and lead days, so it fits this pipeline. The 6 members also give the `forecast_rain_spread` feature.
  - Caveat: it's NCMRWF's extended-range system, not the operational deterministic NCUM-G.
- **NGFS datasets on the same portal are reanalyses** (analysis fields, no lead times), so they can't serve as the forecast source.

**S2S is now working end to end (2026-09-29).**
- `vishwas_ml/ncmrwf_s2s.py` reads the portal's downloads.
- `vishwas_ml/subdivision_weights.py` builds grid-to-subdivision weights from IMD's subdivision shapefile (`indian_met_zones.v2`, github.com/India-Meteorological-Department/Indian_met_zones). The N216 table is committed as `data/weights_ncmrwf_s2s_n216.csv`.
- `scripts/build_s2s_pairs.py` writes the pairs table (S2S + IMD, optionally merging the data team's ERA5 parquet).
- **Lead-day alignment verified against IMD 2015:** file `day00` correlates 0.77 with IMD date init+1, against 0.62 and 0.54 for the neighbouring days, so `dayNN` = `lead_day` NN+1.
- Forecast skill falls from 0.77 (Day 1) to 0.38 (Day 10).

The pipeline handles either source unchanged: an issue cadence of every 8 days (S2S) is fine, and run-to-run change is simply computed against the previous init.

## Known limitations of the NCMRWF S2S source

State these plainly in the write-up and demo.
- **Which NCMRWF model it is.** The S2S reforecast: NCMRWF's Unified Model coupled system (GC2/GA6, N216 ≈ 60 km), initialized from ERA-Interim, with 4 runs per month (1st, 9th, 17th, 25th), 1993–2015. It is not the operational high-resolution NCUM-G deterministic run forecasters use day to day. It is the same model family, and it is real NCMRWF output.
- **Resolution vs subdivision averaging.** Averaging over a subdivision removes much of the resolution gap for large inland subdivisions (e.g. W.RJ, J&K, W.MP: 50–60 grid cells each). It removes much less for narrow coastal and hilly ones (Coastal Karnataka: 12 cells, 2 mostly inside; Kerala; Konkan & Goa), where terrain-driven rain is resolution-sensitive. Expect the model's bust estimates to be least transferable there.
- **Not just resolution.** The model configuration (coupled ocean, model version, ERA-Interim start states) differs from today's operational system, so the error behaviour learned here is a proxy for the operational system's, though a much closer one than GEFS.
- **Run frequency.** 4 runs a month rather than daily, so fewer rows per year (~4,000) and a weaker run-to-run change feature (only Day 1–2 forecasts have an earlier run covering the same day).
- **Years.** 1993–2015 only. Live demo rows for 2026 would need the operational NCUM feed, which isn't on the portal.

## Data questions

1. **ERA5 timing (confirmed by the data team: issue date).** Valid-date reanalysis would leak the outcome. The pipeline still auto-detects the format (`era5_timing: "auto"`); for valid-date data it reads the issue-date state from earlier rows.
2. **`total_precipitation` window.** If it's added, the 24 h accumulation should end at issue time. A calendar-day total for the issue date includes hours after a 00Z issue. It's optional; the current data table doesn't include it.
3. **Rain/no-rain line: 2.5 mm (IMD rainy day), confirmed.** It's configurable in `config.json → bust.rain_threshold_mm`. With a hard line, 2.4 mm vs 2.6 mm counts as a bust, which is expected to dominate the real bust rate.
4. **Lead-day alignment for S2S.** Before relying on it, check whether "T+1" is the 24 h ending 00Z on init+1 or covering init+1, against IMD's 08:30 IST day.

## Assumptions (confirmed by the team)

- **Feedback outcome meaning:** `incorrect` = the forecast busted (label 1), `correct` = it held, `partial` = excluded.
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
