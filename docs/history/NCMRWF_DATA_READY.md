# NCMRWF S2S: full archive downloaded
**From:** ML (Mohit) · **Date:** 2026-09-29 · Replies to `ML_NCMRWF_PIPELINE_PROMPT.md`

## Plan review (as the prompt asked)
The plan is sound. I checked it against the real files, not the labels. What changed or was added:
1. **Per-source bundles.** Thresholds were already fit per dataset, but every model saved to one `ml/models/` folder, so training GEFS after NCMRWF would have overwritten it. Now there are `models/ncmrwf/`, `models/gefs/` and `models/synthetic/`, selected with `--source`. Training refuses to overwrite another source's bundle.
2. **Only init day 01 exists on the portal.** The docs say 1st/9th/17th/25th, but the form disables 09/17/25 for every year 1993–2015. That means 12 runs per year, not 48.
3. **We took all 23 years instead of 5.** Requests complete in about 1 minute, not hours. That gives **1993–2015, 276 runs, ≈91,000 rows**, far more than planned.
4. **Weather state comes from day01, not T.** The portal has no "T" field for MSLP, surface pressure or winds. The day01 values are still NCMRWF output available at issue time, so there's no leakage, but they're the forecast state 24 h ahead.
5. **The winds sit on a staggered grid**, half a cell east-west of rain and pressure. Each variable is averaged over its own grid. The weights table covers both grids, 33 subdivisions each.
6. **Recent observed rain** = IMD on (issue date − 1). IMD day D ends 03Z on D, so day D would be 3 h in the future at a 00Z issue.
7. **Dewpoint and 2 m temperature are not offered** at single level. Those columns are empty for this source, and the model handles that.
8. **Block-bootstrap CIs** (resampling whole forecast runs) are now reported, since test sets are small.

## Downloaded (all verified on the portal's request log)
- 23 requests: years 1993–2015, init 01, all 12 months, T..T+9, India box N38/S6/E98/W68.
- Variables: Total Precipitation Amount, MSLP, Surface Pressure, 10 m U, 10 m V.
- Each zip is ~20 MB and holds 552 files: rain day00–09, the other 4 variables day01–09.
- IMD gridded rainfall for 1992–2016 (imdlib), for observations and previous-day rain.

## Still needed from you
Nothing for the NCMRWF track. When GEFS + ERA5 is done, share it and I'll train it as a separate `gefs` model with its own thresholds, then compare the two side by side.

## Results: NCMRWF model (branch `ml-pipeline-v1`, `ml/models/ncmrwf/`)

**Setup:** 91,080 rows (1993–2015, 276 runs × 33 subdivisions × 10 leads), 0 missing IMD. Thresholds fit on 1993–2011, calibration on 2012, **test on 2013–2015**.

| | PR-AUC [95% CI] | Recall @ Orange+ |
|---|---|---|
| **Model** | **0.534** [0.499, 0.578] | 0.52 |
| Forecast amount only | 0.452 [0.423, 0.489] | 0.40 |
| Climatology | 0.272 [0.243, 0.296] | 0.01 |

- **Base rate** is 16.6%, and **model minus forecast-only is +0.084 [+0.062, +0.109]**.
- **Calibration error** is 0.021.
- **Lead day:** no gain at Days 1–3; the gain is at Days 5–10.
- **Trigger type:** catches 53% of rain/no-rain busts but only 30% of magnitude busts.
- **Where it works:** best on the NW plains (HR/DL, PB, W.UP: 0.73–0.76), weakest on the coast (Coastal Karnataka recall 13%, Kerala 0.42).
- **Chennai 2015 replay** (out-of-sample): Day 1 missed (11%); Day 4 flagged Orange and Day 5 Yellow, and both did bust.

The API runs with `cd ml && .venv/Scripts/python -m uvicorn serve:app --port 8000` (frontend paths, `/docs` for the list).
