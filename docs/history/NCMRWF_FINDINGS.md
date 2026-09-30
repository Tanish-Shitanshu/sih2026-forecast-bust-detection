# NCMRWF Data Access: Findings
**Project:** Vishwas, SIH 2026 PS 26079
**From:** ML (Mohit), in reply to `NCMRWF_ACCESS_TASK.md`
**Date:** 2026-09-29

---

## TL;DR

**NCMRWF's own forecast archive with rainfall and lead days exists on the portal.** It's the **"S2S global sub-seasonal to seasonal re forecast since 1993 to 2015"** dataset. The earlier search looked only at the NGFS datasets, and those turn out to be reanalyses, so they could never have worked no matter what variables they had.

| Question from the task doc | Answer |
|---|---|
| Rainfall variable? | **Yes.** "Total Precipitation Amount" (Accumulated, Single Level) |
| Separate values per lead day? | **Yes.** A "Forecast Day" selector from T to T+36, with each day selectable |
| Whose model? | **NCMRWF's own**: the NCMRWF implementation of the Met Office Unified Model coupled system (GC2) |
| Years | **1993–2015** (23 years) |
| Format | NetCDF4 in a Zip, CC-BY licence |
| Access | Sign in, then download. No approval step, but each request is a **queued job** (the portal dashboard shows ~29 min processing plus ~2 h queue wait) |

---

## 1. The dataset

- **Name:** S2S global sub-seasonal to seasonal re forecast since 1993 to 2015
- **URL:** https://rds.ncmrwf.gov.in/datasets/s2s-daily (Download tab: `?tab=download`)
- **DOI:** 10.64349/nmrf.rds.s2s.50521 (cite this)
- **System:** Global Atmosphere 6.0 (Unified Model, ENDGame core) at **N216** (~60 km), 85 levels, coupled to NEMO ocean and sea ice. Atmosphere initial conditions come from ERA-Interim.
- **Initializations:** the **1st, 9th, 17th and 25th of every month**, which is 48 init dates per year and ~1,100 over 1993–2015
- **Ensemble:** **6 members** (lagged initial conditions plus SKEB2 perturbations)
- **Forecast length:** 36 days, daily output
- **Docs quote:** "The dataset provides daily outputs for all ensemble members and forecast lead times."

### Download form (exact options)
- **Year:** 1993 … 2015 · **Month:** Jan … Dec · **Initialization Day:** 01 / 09 / 17 / 25
- **Forecast Day:** T, T+1 … T+36 (select all / clear all)
- **Level → Single Level:**
  - Instantaneous: 10 Metre U/V Wind Component, **Mean Sea Level Pressure**, **Surface Pressure**, Orography, Land Mask
  - Accumulated: **Total Precipitation Amount**
  - Averaged: surface SW/LW/latent heat fluxes
- **Level → Pressure Level:** U, V, T, Geopotential, Specific Humidity at 10–925 hPa
- **Geographical Area:** Whole Area or **Select Coordinates** (use India: lat 6–38 N, lon 68–98 E)
- **Format:** NetCDF4, Zip · **Terms:** "I accept the CC-BY licence" · Login to Download

## 2. What the other datasets are (so nobody re-checks them)

| Dataset | What it actually is | Usable as forecast source? |
|---|---|---|
| NGFS daily / 6-hourly single levels / 6-hourly pressure levels (1999–2018) | **Reanalysis.** Docs: "gridded atmospheric analysis fields … NGFS reanalysis", T574L64 | **No.** No lead times |
| IMDAA (1979–2020) | Regional reanalysis | No (could substitute for ERA5, but we don't need that) |
| MERA (2020–2025) | Hourly merged satellite+radar **rainfall analysis** over India | No (observations, not a forecast) |
| Ocean analysis (2016–2026) | NEMO ocean analysis | No |
| **S2S (1993–2015)** | **NCMRWF reforecast** | **Yes** |

The top-nav "Download" link only goes to your own request queue (`/my-requests`), and "Live" is a server-stats dashboard. There's no separate bulk archive.

## 3. Trade-offs vs GEFS, stated honestly

**Better than GEFS:**
- It is **NCMRWF's model**, which is what PS 26079 is about, so "is this actually NCMRWF's forecast?" gets a yes.
- 23 years instead of 5 means far more busts to learn from, and per-subdivision thresholds become more stable.
- The 6 ensemble members give **ensemble spread** as a feature. In my earlier research, a learned model on ensemble features was a strong bust predictor.

**Weaker than GEFS / caveats to state in the write-up:**
- It's NCMRWF's **extended-range (S2S) coupled system**, not the operational deterministic **NCUM-G** medium-range model forecasters use day to day. Same model family (Unified Model), coarser (N216 ~60 km).
- There are only 4 inits per month, not daily, so ~1,100 init dates × 33 subdivisions × 10 leads ≈ **364k rows** (GEFS 2015–2019 gives ~600k). That's still plenty.
- The years (1993–2015) don't overlap GEFS's 2015–2019 except 2015. That's fine: ERA5 (1940+) and IMD (1901+) cover them.
- **Lead-day alignment must be checked** on the first file: is "T+1" the 24 h ending 00Z on init+1, or the calendar day init+1? That then has to line up with IMD's day (24 h ending 08:30 IST).

## 4. ML side is ready for it

On branch `ml-pipeline-v1` (pushed):
- The loader accepts **the data team's exact column names** (`forecast_rain_mm`, `era5_msl_pa`, …). Missing wind or ERA5 precipitation and missing `is_bust` are fine; a test trains on exactly the format in `DATA_PIPELINE_STATUS.md`.
- It takes an optional **`forecast_rain_spread_mm`** column (std of the 6 members' daily rain) and uses it as a feature when present. `forecast_rain_mm` should be the **ensemble mean**.
- An 8-day init cadence is handled; nothing else changes.

## 5. Proposed next steps

1. **Test request (small; do this first).** Sign in, open the S2S Download tab, and select:
   - Year 2015, Month Jul, Init Day 01
   - Forecast Day T to T+10
   - Single Level → Total Precipitation Amount
   - Select Coordinates: 6–38 N, 68–98 E; NetCDF4

   Then accept the licence and submit. Expect ~2–3 h in the queue.
2. **Inspect the file.** Check the member dimension, how lead time and time are encoded, units (kg m⁻² = mm, or m?), and whether accumulation is per day or running. This settles question 4 in section 3.
3. **If it checks out**, script the remaining requests: one request per (year, month) with all 4 init days, T..T+10, precipitation only, India box. That's 276 requests, so ask the portal whether multi-month requests are allowed, or queue them in batches.
4. **Data pipeline:** add an S2S reader that outputs ensemble-mean `forecast_rain_mm` plus `forecast_rain_spread_mm` per (init date, subdivision, lead day), reusing the existing 33-subdivision aggregation. IMD and ERA5 extraction for 1993–2014 are the other new pieces.
5. **Keep GEFS running** as the fallback until S2S rows actually exist. The ML pipeline can train on either, or be compared on both.
