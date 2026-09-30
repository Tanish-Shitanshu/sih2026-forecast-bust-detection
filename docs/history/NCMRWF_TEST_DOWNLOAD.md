# NCMRWF S2S: first test download results
**From:** ML (Mohit) · **Date:** 2026-09-29 · Follows `NCMRWF_FINDINGS.md`

## What came back (init 2015-07-01, Forecast Day T..T+10, Total Precipitation Amount)
- The request **went through** and took ~2 h in the queue. The zip is 1.2 MB.
- **One NetCDF per forecast day:** `APCP-sfc_IC20150701_day00.nc` … `day10.nc`
- **Variable** `apcp` on `(t, latitude, longitude)`; `t` = init date + NN days
- **Grid:** N216, 0.833° lon × 0.556° lat (coarser than GEFS 0.5° / IMD 0.25°)
- **Units:** the attribute says `kg m-2 s-1`, but **the values are already daily totals in mm**. NE India in July averages 10–40 mm/day; multiplying by 86400 would give ~10⁶. Don't convert.
- **No ensemble-member dimension**, and the form has no member selector. It's one field per day (likely the ensemble mean or control), so there's no spread feature.

## Problem found: the area was wrong. Must re-request
The form's coordinate boxes are ordered **NORTH, SOUTH, EAST, WEST** (not W-then-E). Our request reached the server as longitude 90→68, so it returned **everything except 68–90°E, i.e. most of India**. Correct entries: **NORTH 38 · SOUTH 6 · EAST 98 · WEST 68**. The ML reader now refuses a file with the box swapped.

## Lead-day mapping (please match this in the data pipeline)
- `day00` = first 24 h after the 00Z init (it can't be before the init), i.e. the 24 h ending 00Z on init+1.
- IMD's date D is the 24 h ending 08:30 IST (03Z) on D. So `dayNN` pairs with **IMD date init+NN+1**, which is our **`lead_day = NN+1`**.
- Lead days 1–10 therefore need **T..T+9**; T+10 isn't used.
- **Check:** is GEFS lead day 1 in `_features_*.parquet` also the first 24 h (forecast hour 0–24)? Both sources should use the same convention.
- I haven't yet verified this against IMD data. Once a correct-box file arrives, the check is: the correlation of `day00` with IMD date init+1 should beat that with IMD date init.

## Ready on the ML side (branch `ml-pipeline-v1`)
`ml/vishwas_ml/ncmrwf_s2s.py`:
- `read_request(zip_or_folder)`: returns a tidy grid with init_date, lead_day, lat, lon, forecast_rain_mm. It applies the mapping above, normalizes longitude, and guards units and the box.
- `to_subdivisions(grid, weights)`: area-weighted means per subdivision. It needs **your cell → subdivision weights table built on the N216 grid** (columns latitude, longitude, subdivision_code, weight), the same way you built them for GEFS/ERA5.
- `request_plan()`: the 92 requests (see below).

## Full-dataset size and how to request it
- On the form, **Year** and **Initialization Day** are single choice, but **Month is multi-select**. So one request = one year × one init day × all 12 months.
- The full archive is **23 × 4 = 92 requests**, each selecting all 12 months, Forecast Day T..T+9, Total Precipitation Amount, and N 38 / S 6 / E 98 / W 68.
- That's ~1,100 inits × 10 leads ≈ 11,000 small files, about 1 GB.
- Suggested order: 2011–2015 first (5 years, 20 requests), so there's a usable train/calibration/test set quickly, then work backwards.
