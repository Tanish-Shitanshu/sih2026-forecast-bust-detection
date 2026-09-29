# Vishwas ML pipeline

Predicts, for each of the 33 IMD subdivisions and each lead day 1–10, the calibrated probability that an issued medium-range rainfall forecast (NCMRWF, or GEFS as a stand-in) will bust. It also explains the probability in plain language, finds historical analogs, and scores the model's own reliability.

**Status:** trained on **real NCMRWF forecasts** (S2S reforecast, 1993–2015, 91,080 rows) against IMD observations. It's evaluated on held-out years 2013–2015 with confidence intervals. A synthetic-data model is kept for pipeline tests, and a GEFS model will be trained separately when that dataset arrives.

## Quick start

```bash
cd ml
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python scripts/train.py --source ncmrwf        # real NCMRWF model -> models/ncmrwf (~30 s)
.venv/Scripts/python scripts/evaluate.py --source ncmrwf     # metrics + CIs, per lead day / subdivision
.venv/Scripts/python scripts/demo_api.py --source ncmrwf --cycle 2015-12-01 --subdivision TN/PY --lead 1
.venv/Scripts/python -m uvicorn serve:app --port 8000        # live API on the frontend's /api/v1 paths
.venv/Scripts/python -m pytest -q tests                      # 44 tests (~2 min)
```

`--source` picks the forecast source: `ncmrwf` (default), `gefs` (when Tanish's data lands) or `synthetic`. Each has its own data file, bust thresholds and bundle under `models/<source>/`.

**Rebuilding the NCMRWF table from the raw downloads** (23 zips from rds.ncmrwf.gov.in, IMD via imdlib, IMD subdivision shapefile):
```bash
.venv/Scripts/python scripts/build_s2s_pairs.py --s2s <SIH>/s2s_raw/years/*.zip --fc <SIH>/s2s_raw/ncmrwf_fc_1993_2015.parquet     --imd-dir <SIH>/geo_data/imd --shapefile <SIH>/geo_data/indian_met_zones.v2 --out data/ncmrwf_pairs.parquet
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

Full guide for the backend team: [BACKEND_INTEGRATION.md](BACKEND_INTEGRATION.md).

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

## Current metrics: NCMRWF (real data)

**Setup:**
- NCMRWF S2S rain + MSLP / surface pressure / 10 m winds, 1993–2015, 12 runs a year.
- IMD 0.25° observations averaged to the 33 subdivisions.
- Bust thresholds fit on the training years only (1993–2011).
- Calibration year 2012; **test years 2013–2015** (36 runs, 11,880 rows).
- Base rate is 16.6%. 95% CIs come from a block bootstrap over the 36 issue dates.

| | PR-AUC [95% CI] | ROC-AUC | Brier skill vs clim. | ECE | Precision @ alert | Recall @ alert | Accuracy @ 0.5 |
|---|---|---|---|---|---|---|---|
| **Model** | **0.534** [0.499, 0.578] | 0.864 | **0.230** | 0.021 | 0.546 | **0.518** | 0.857 |
| Forecast amount + lead only | 0.452 [0.423, 0.489] | 0.835 | 0.171 | 0.012 | 0.522 | 0.395 | 0.847 |
| Climatology (historical rate) | 0.272 [0.243, 0.296] | 0.682 | 0 | 0.018 | 0.392 | 0.010 | 0.834 |

**Model minus forecast-only: +0.084 PR-AUC, 95% CI [+0.062, +0.109]**, better in 100% of resamples.

**What the numbers say:**
- **Lead day.** At Days 1–3 the model is no better than the forecast amount alone (Day 1: 0.478 vs 0.479). The gain appears from Day 5 onward (Day 5: 0.570 vs 0.456; Day 7: 0.491 vs 0.373).
- **Trigger type.** At Orange+ it catches **53% of rain/no-rain busts but only 30% of magnitude busts**. Large misses are the hard case.
- **Where it works.**
  - Best on the large NW plains: Haryana/Delhi 0.76, Punjab 0.76, West UP 0.73.
  - Weakest where the resolution limitation predicts: Coastal Karnataka (alert recall 13%), Kerala 0.42, Konkan & Goa 0.45.
  - Gujarat is also low (0.26), but its base rate is only 6.7%.
- **Inputs.** The NCMRWF weather variables add little: mean |SHAP| is 0.05 for pressure and 0.03 for wind, against 1.22 for forecast amount. Skill comes mainly from the forecast itself, how it varies across the run, regional context and each subdivision's bust history.
- **Accuracy is not the headline.** Climatology scores 83% accuracy while catching 1% of busts.

Per-lead and per-subdivision tables are in `models/ncmrwf/evaluation/`. Retrain with `python scripts/train.py --source ncmrwf`.

### GEFS + ERA5 comparison model (Tanish's PR #2, backup track)

**Setup:** GEFSv12 control reforecast, 2015–2019, daily runs, 602,580 rows; ERA5 features; IMD observations. It has its own thresholds and its own bundle in `models/gefs/`, with nothing shared with NCMRWF. Split: train 2015–2017, calibration 2018, test 2019. The labels match the data team's `is_bust` column 100%.

| | NCMRWF (primary) | GEFS (backup) |
|---|---|---|
| Test years / rows | 2013–2015 / 11,880 | 2019 / 120,450 |
| Base rate | 16.6% | 16.7% |
| **Model PR-AUC [95% CI]** | **0.534** [0.499, 0.578] | **0.509** [0.495, 0.523] |
| Forecast amount only | 0.452 | 0.467 |
| Model minus forecast-only | +0.084 [+0.062, +0.109] | +0.042 [+0.035, +0.050] |
| Recall @ Orange+, magnitude busts | 30% | 41% |

The test years differ, so this is a side-by-side, not a head-to-head.

**ERA5 leakage: found, fixed at source, verified.**
- The first version of the table had ERA5 as daily means/sums over the *issue date*, which includes the 24 h after a 00Z issue. ERA5 rain on the issue date correlated 0.81 with the Day-1 observed rain, and the leak inflated Day-1 PR-AUC from 0.635 to 0.676.
- The data team fixed it at source (data branch commit `56a65ec`): every ERA5 column is now the previous day's.
- Checked here: new ERA5 = old ERA5 of the previous day exactly; forecast/observed/labels are byte-identical; ERA5 rain scale is fixed (mean 3.89 mm vs IMD 3.49).
- Because the shift is now in the data, the `gefs` profile uses `era5_shift_days: 0`. Setting it to 1 would shift twice.
- Results on the fixed data: PR-AUC 0.509 [0.495, 0.523], Day 1 0.633, the same as the ML-side workaround gave.
- 1 Jan 2015 (330 rows) has no ERA5 (it would need 31 Dec 2014); the model treats it as missing.
- `gefs` reads `../data/processed/bust_dataset.parquet`, which exists once PR #2 is merged. Before that, train with `--data <copy>`.

### Synthetic data (pipeline check only, not a skill claim)
Test year 2023, 39,930 rows: PR-AUC 0.611 [0.594, 0.625] vs forecast-only 0.578 and climatology 0.381. Bundle in `models/synthetic/`.

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
- **Run frequency: 12 runs a year.** The documentation lists inits on the 1st, 9th, 17th and 25th, but the portal only offers **init day 01** for every year (09/17/25 are disabled on the download form, checked 2026-09-29 for 1993–2015). That gives 12 runs × 33 subdivisions × 10 lead days = 3,960 rows per year, **~91,000 rows over 1993–2015**. Runs are a month apart, so the run-to-run change feature is always empty for this source; the model simply doesn't use it.
- **Weather state is the day-01 forecast, not the analysis.** The portal has no "T" field for MSLP, surface pressure or winds (they start at day01). The pipeline uses the model's day-01 values as the issue-time state. They are NCMRWF output that exists when the forecast is issued, so there's no leakage, but they describe the atmosphere 24 h ahead, not the observed state at issue.
- **Staggered grid.** UM 10 m winds sit half a grid cell east-west of rain and pressure. Each variable is averaged over its own cells (the weights table carries both grids).
- **Years.** 1993–2015 only. Live demo rows for 2026 would need the operational NCUM feed, which isn't on the portal.

## Demo event: Chennai floods, December 2015

From the real NCMRWF run of 1 Dec 2015 against IMD, subdivision Tamil Nadu & Puducherry:
- **Day 1 (2 Dec):** forecast **12.4 mm**, observed **41.0 mm** subdivision-wide. That's a magnitude bust (error −28.6 mm, over the 25 mm floor).
- **5–11 Dec:** the forecast called almost no rain (0.7–4.6 mm) while 3–21 mm fell, so there are rain/no-rain busts on Days 4, 5, 8, 9 and 10.
- **Neighbouring subdivisions** (Coastal AP, Rayalaseema) were forecast well.
- **Caveat to say out loud:** 41 mm is the average over all of Tamil Nadu; Chennai itself got several hundred mm. Subdivision averaging dilutes local extremes.

**What the trained NCMRWF model says** (2015 is a test year, so this is out-of-sample):
- **Day 1 (2 Dec):** 11%, Green. **Missed.** The Day-1 lead time pulled the risk down.
- **Day 4:** 40%, Orange. Correct: that day busted (rain/no-rain).
- **Day 5:** 20%, Yellow. Correct.

Present it as it is: the model flagged the continuing-rain phase and missed the first extreme day. That matches its measured weakness on magnitude busts.

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
