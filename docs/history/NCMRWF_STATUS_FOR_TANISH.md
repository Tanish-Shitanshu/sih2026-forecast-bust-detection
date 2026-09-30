# NCMRWF forecast data: solved, with Day 1–10 verified
**From:** ML (Mohit) · **Date:** 2026-09-29 · Replaces `NCMRWF_FINDINGS.md` / `NCMRWF_TEST_DOWNLOAD.md`

## 1. "Isn't S2S extended-range only (16–45 days)?" No. Verified on real files.
S2S systems *extend to* week 6, but they output **every day from the init onward**. What we downloaded from rds.ncmrwf.gov.in:
- The portal's Forecast Day selector runs **T, T+1, T+2 … T+36**.
- Files are `APCP-sfc_IC20150701_day00.nc … day09.nc`: one real rainfall field per day, starting at the init date.
- **Checked against IMD observations (2015, all 33 subdivisions, 12 inits):**

| NCMRWF file dayNN vs IMD date | init+NN | **init+NN+1** | init+NN+2 |
|---|---|---|---|
| Correlation, lead 1 | 0.62 | **0.77** | 0.54 |
| Correlation, all leads | 0.58 | **0.61** | 0.47 |

- **Skill by lead (correlation with IMD):** Day 1 0.77 · Day 2 0.79 · Day 3 0.70 · Day 4 0.74 · Day 5 0.51 · Day 6 0.70 · Day 7 0.63 · Day 8 0.54 · Day 9 0.51 · Day 10 0.38. That's a normal medium-range decay.
- **Mean forecast 3.24 mm/day vs observed 2.99.** Little bias.

**Mapping:** `dayNN` = **lead_day NN+1**, valid on IMD date init + lead_day. Lead days 1–10 need **T..T+9**.

**Honest caveat for judges:** this is NCMRWF's S2S reforecast (their Unified Model coupled system, ~60 km, 1993–2015, 4 inits/month), not the operational high-res NCUM-G deterministic run. It is NCMRWF's model, though, with rainfall per lead day, which GEFS never was.

## 2. What's built (branch `ml-pipeline-v1`, folder `ml/`)
- **Weights table for NCMRWF's grid:** `ml/data/weights_ncmrwf_s2s_n216.csv` (1,163 rows, all 33 subdivisions).
  - Built from IMD's subdivision shapefile (`indian_met_zones.v2`, github.com/India-Meteorological-Department/Indian_met_zones).
  - Weight = overlap area × cos(lat). The shapefile has 36 subdivisions; A&N, Lakshadweep and Uttarakhand are dropped to match the frontend's 33.
  - Builder: `ml/vishwas_ml/subdivision_weights.py`. It works for any regular grid.
- **Reader:** `ml/vishwas_ml/ncmrwf_s2s.py`. It handles:
  - values that are already mm/day, despite the "kg m-2 s-1" label;
  - float32 coordinates;
  - refusing downloads whose EAST/WEST boxes are swapped.
- **Pairs builder:** `ml/scripts/build_s2s_pairs.py`. Takes S2S downloads + IMD (via imdlib) + optional ERA5 parquet, and writes the pairs table in **your column format** (`forecast_rain_mm`, `observed_rain_mm`, `error_mm`, `era5_*`, `year`).
- **2015 test output:** 3,960 rows (12 inits × 33 × 10), 0 rows missing IMD.

## 3. What I need from you
1. **ERA5 parquet(s)** with `date, subdivision_code, era5_msl_pa, era5_sp_pa, era5_d2m_k, era5_t2m_k` at the **issue date** (one row per date × subdivision), for the years we pull from S2S. The builder merges them on (date, subdivision_code).
   - Note: S2S covers **1993–2015**, and your ERA5 so far is 2015–2019. We need ERA5 for the S2S years we use (suggest 2011–2015 first).
2. **IMD:** not needed, the builder pulls it with imdlib. Send yours if it's already aggregated, and I'll cross-check.

## 4. Download plan (on our side)
- One request = **1 year × 1 init day × all 12 months**, Forecast Day T..T+9, Total Precipitation Amount, **N 38 · S 6 · E 98 · W 68**.
- **Done:** 2015 init day 01.
- **Next:** 2015 init days 09/17/25, then 2011–2014 (4 init days each). That's 19 more requests, giving 5 years for the train/calibrate/test split.

## 5. One finding for the team
Real bust rate on 2015 is **~18%**, but **94% of busts are rain/no-rain flips** (the 2.5 mm line). Magnitude busts (≥ 25 mm or P95) are rare. That's our locked definition working as written, but the demo should show both triggers, and we may want to present them separately.
