# Rainfall bust-detection dataset — SIH 2026 PS 26079 (Vishwas)

**This is the GEFS+ERA5 secondary/comparison track, not the primary one.**
NCMRWF's real S2S data (Mohit's `ml-pipeline-v1` branch) is the primary
forecast source — its `day00`–`day09` files give exactly the Day 1–10
horizon this project needs. This dataset instead pairs NOAA's public
GEFSv12 reforecast against IMD ground truth, as an independent second
model to validate against. **The two tracks are never merged**: each has
its own independently-fit bust magnitude threshold (see "Locked bust
definition" below), computed separately on its own training years, so
neither track's numbers leak into or get diluted by the other's.

**What's in it.** One row per (forecast issue date, subdivision, lead day),
pairing a GEFS rainfall forecast against what IMD's gridded rainfall says
actually happened, with a locked bust definition applied and a handful of
ERA5 atmospheric features attached at the forecast's issue date.
`data/processed/bust_dataset.parquet` — **602,580 rows**, zero missing
(subdivision, lead_day) combinations across all 5 years.

**Train/test split.** 5 years, 2015–2019, full calendar year (not
monsoon-only). 2015–2018 → `train` (482,130 rows), 2019 → `test` (120,450
rows). The bust magnitude threshold is fit on `train` only, then applied
unchanged to both splits — see "Locked bust definition" below.

**Bust rate.** **15.3%** of rows are flagged `is_bust`. Broken down by
`trigger_reason`: `none` 510,438 (84.7%), `category` 84,724 (14.1%),
`magnitude` 6,192 (1.0%), `both` 1,226 (0.2%) — category mismatches
dominate, which is expected, since the magnitude threshold is deliberately
calibrated to only catch the most extreme ~5% of errors per subdivision
and lead day.

## Sources

| Source | Used for | Access |
|---|---|---|
| [IMD Gridded Rainfall](https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html) | Ground truth (`observed_rain`) | Direct download, no login. Daily, 0.25°, India. |
| [ERA5](https://cds.climate.copernicus.eu) (ECMWF, via Copernicus CDS) | Atmospheric features | Free account + personal token in `~/.cdsapirc`, via the `cdsapi` package. |
| **NOAA GEFSv12 reforecast** (`s3://noaa-gefs-retrospective/`) | Forecast (`forecast_rain`) | Public S3, no login. |

**Why this track exists alongside NCMRWF's own data.** `rds.ncmrwf.gov.in`
was checked first, per the task's instructions, and its reanalysis
products — IMDAA (1979–2020) and NGFS (1999–2019) — don't fit a daily
Day 1–10 medium-range forecast need. At the time this track was built, its
S2S product (1993–2015) looked like it wouldn't fit either; **that read
was wrong and has since been corrected** — S2S's `day00`–`day09` files do
give exactly Day 1–10, and that's exactly what Mohit's `ml-pipeline-v1`
branch now uses as the primary track. This GEFS+ERA5 track was kept as-is
rather than retired, since an independent second forecast source (GEFS
control member `c00`, total precipitation `apcp_sfc`) is genuinely useful
for comparison, not a stand-in for a missing primary source.

## Scope

Full calendar year, 2015–2019, all 12 months. ERA5 itself was downloaded
for the fuller 2015–2024, 12 months (sitting in `data/raw/era5/`), but the
paired table is capped at 2019 because **GEFS reforecast data only goes up
to 2019** (verified directly against the bucket: there's a `2020/` prefix,
but it holds exactly one misfiled 2010 date, not real 2020 coverage).
Extending past 2019 needs a different forecast archive.

Scaling up further (once/if a post-2019 forecast source exists) is a
matter of re-running the extraction and `compute_bust.py` for the extra
years — the date range isn't hardcoded into the bust logic.

## The 33 subdivisions

`data/scripts/subdivisions.csv` holds the same 33 meteorological
subdivisions (id, code, name) as `frontend/index.html`'s `SUBS` array, so
codes match exactly end to end.

Subdivision **boundaries** come from IMD's own official file
(`mausam.imd.gov.in/imd_latest/contents/district_shapefiles/sd_boundary.json`),
which has 36 subdivisions (ours are 36 minus Lakshadweep, A&N Islands, and
Uttarakhand — none of which are in the frontend's list). That raw file
ships with **no CRS and unlabeled projected coordinates** — not lon/lat.
Rather than guess the projection, we used a community re-projection of the
same official dataset to WGS84
([gist](https://gist.github.com/planemad/1604109e8057cb6d6822e8909468f16c)),
verified independently before trusting it: all 36 names match the raw file
exactly, the declared CRS is CRS84, the overall bounding box lands at the
expected lon 68–98° / lat 6–38° for India, and small/compact subdivisions
(Kerala, Tamil Nadu) land within ~15km of their known geographic centroids.
`data/scripts/build_subdivision_boundaries.py` maps the 36 down to our 33
and relabels with our codes; the result is
`data/processed/subdivision_boundaries.geojson`.

Every 0.25° grid cell (IMD, GEFS, and ERA5 each have their own grid) is
assigned to a subdivision by point-in-polygon on the cell center
(`data/scripts/grid_utils.py`). A subdivision's daily value is the mean
over every grid cell assigned to it (a sum, for the accumulated
precipitation fields). Cells outside all 33 subdivisions (ocean, other
countries) are simply not counted.

## Locked bust definition

```
bust = (|forecast_rain - observed_rain| >= max(25mm, subdivision's 95th-
        percentile rain error FOR THAT LEAD DAY, fit on training years only))
       OR (rain/no-rain category call was wrong)
```

- **Magnitude threshold**: for each (subdivision, lead_day), the 95th
  percentile of `|forecast_rain - observed_rain|` is computed using
  **only rows where `year` is a training year** (2015–2018 here), then
  floored at 25mm. That fixed threshold is then applied to every row,
  train and test alike — see `magnitude_threshold_mm`.
- **Category call**: a day is "rain" if rainfall ≥ 2.5mm (IMD's standard
  rainy-day cutoff), applied to both `forecast_rain` and `observed_rain`
  independently. The category call is wrong if the two disagree (forecast
  said rain, observed said no-rain, or vice versa).
- **`trigger_reason`** records which condition(s) actually fired:
  `magnitude`, `category`, `both`, or `none` (not a bust).

Train/test split: `year in {2015,2016,2017,2018}` → `train`,
`year == 2019` → `test`. This is what makes the threshold "fit on training
years only" — the 2019 rows never influence their own bust threshold.

Implemented in `data/scripts/compute_bust.py`.

## Data dictionary — `data/processed/bust_dataset.parquet`

| Column | Meaning | Units / values |
|---|---|---|
| `date` | Forecast issue (init) date | `YYYY-MM-DD` |
| `subdivision_code` | IMD subdivision code, matches `frontend/index.html` | e.g. `KL`, `W.RJ` |
| `lead_day` | Forecast lead time | integer, 1–10 |
| `forecast_rain` | GEFSv12 reforecast (control member) 24h precip total for this lead day, subdivision-mean | mm |
| `observed_rain` | IMD gridded rainfall, subdivision-mean, on `date + lead_day` | mm |
| `error` | `forecast_rain - observed_rain` | mm (signed; positive = forecast over-predicted) |
| `is_bust` | Locked bust definition applied | boolean |
| `trigger_reason` | Which part of the bust definition fired | `magnitude`, `category`, `both`, `none` |
| `mslp` | ERA5 mean sea level pressure, subdivision-mean, daily mean of the 4 synoptic hours, from **`date - 1`** | Pa |
| `surface_pressure` | ERA5 surface pressure, same aggregation, from `date - 1` | Pa |
| `dewpoint_2m` | ERA5 2m dewpoint temperature, same aggregation, from `date - 1` (moisture proxy) | K |
| `temp_2m` | ERA5 2m temperature, same aggregation, from `date - 1` | K |
| `wind_u10` | ERA5 10m eastward wind component, same aggregation, from `date - 1` | m/s |
| `wind_v10` | ERA5 10m northward wind component, same aggregation, from `date - 1` | m/s |
| `total_precipitation` | ERA5's own precipitation estimate (reanalysis, independent of both `forecast_rain` and `observed_rain`), subdivision-mean, true daily **sum of all 24 hourly values**, from `date - 1` | mm |

Plus 4 columns beyond the requested handoff schema, kept because they're
useful and don't conflict with it — flag if a strict schema match is
needed instead and these should be dropped:

| Column | Meaning |
|---|---|
| `subdivision_name` | IMD subdivision name, matches `frontend/index.html` |
| `magnitude_threshold_mm` | The actual per-(subdivision, lead_day) magnitude threshold applied to this row (mm) |
| `year` | Calendar year of `date` |
| `split` | Whether this row's year was used to fit `magnitude_threshold_mm` — `train` or `test` |

All ERA5 features describe atmospheric conditions **as of the day before
the forecast's issue date** — see "Known issues, found and fixed" below
for why — and are constant across a given date+subdivision's 10 lead-day
rows (they're a feature of the forecast run itself, not of any one lead
day).

`bust_dataset.csv` (same data, gitignored — 135MB, over GitHub's 100MB
hard limit) can be regenerated locally: `data/.venv/bin/python3 -c
"import pandas as pd; pd.read_parquet('data/processed/bust_dataset.parquet').to_csv('data/processed/bust_dataset.csv', index=False)"`.

## Known issues, found and fixed

Mohit found two real bugs in the ERA5 side features after this dataset was
first built. Neither affects `forecast_rain`, `observed_rain`, `error`,
`is_bust`, or `trigger_reason` — only the 7 ERA5 columns were wrong and
have been rebuilt (`data/scripts/fix_era5_leakage_and_precip.py`).

**Leakage.** ERA5 features were originally attached at the forecast's own
issue date, averaged/summed over all 4 of that day's synoptic hours
(00/06/12/18Z). For a forecast issued at 00Z, three of those four
snapshots happen *after* the forecast was made — the "feature" partly
described the future. This is why Day-1 model performance looked inflated
(0.635) before being caught: the issue-date ERA5 rainfall feature
correlated 0.81 with next-day's actual observed rain, implausibly high for
a same-day weather-state feature. Fixed by shifting every ERA5 feature to
`issue_date - 1` (the most recent full day entirely in the past when a
00Z forecast was made) — enforced centrally by
`era5_utils.era5_features_for_issue_date`, so no future call site can
reintroduce the leak by indexing the raw per-day dict directly.

*Re-checked after the fix*: the same correlation (corrected
`total_precipitation` vs. next-day `observed_rain`, full dataset, n=60,225)
is now **0.653** — a real, sharp drop from 0.81, not just a smaller number.
To confirm it's not still-leaky rather than just "lower," it was checked
against an independent, ERA5-free baseline: IMD's own ground-truth rain
autocorrelated with itself 2 days later is **0.565** (n=59,928, no
ERA5/GEFS involved at all). 0.653 sits right next to that pure-persistence
baseline — the fixed feature now behaves like real weather persistence,
not like something that can see the future.

**`total_precipitation` scale.** It averaged 0.61mm against IMD's 3.29mm
on matched same-calendar-days (reproduced directly, not just taken on
faith) — a ~5.4x undercount, matching Mohit's diagnosis almost exactly.
Root cause: CDS delivers ERA5 `total_precipitation` as clean 1-hour
increments even at hourly resolution, so the original 4-times-daily sample
(00/06/12/18Z) only ever captured ~4 of the day's 24 hours (4/24 ≈ 0.167,
close to the observed 0.185 ratio — the tell). Fixed by downloading all 24
hourly values per day separately (`download_era5_hourly_tp.py`,
`era5_utils.hourly_tp_subdivision_daily`) and summing those instead of the
original 4. The old 4-sample summation was removed from
`era5_subdivision_daily` entirely — not just stopped being called — so the
wrong quantity can no longer be pulled from that function by mistake.

*Re-checked after the fix*: corrected `total_precipitation` now averages
3.891mm against IMD's 3.488mm on the same matched days (full dataset,
n=60,225) — **ratio 1.116**, up from 0.185, and squarely in normal
reanalysis-vs-gauge range rather than a 5.4x undercount. The
same-day correlation was essentially unchanged (0.746 vs. 0.740 before) —
expected, since this bug was a magnitude/scale error, not a shape error;
fixing it corrected the mean without needing to touch the correlation.

**Known gap**: fixing the leakage shift means `2015-01-01` needs ERA5 data
from `2014-12-31`, which was never downloaded (the original pull started
at 2015). Those specific rows (33 subdivisions × 10 lead days = 330 of
602,580, 0.05%) have `NaN` ERA5 columns rather than a silently wrong value.

## Layout

```
data/
  raw/
    imd/     IMD yearly NetCDF (ind{year}_rfp25.nc), + the raw and validated subdivision boundary GeoJSONs
    era5/    era5_{year}.nc (4x-daily instant fields; actually zip archives — see era5_utils.py), extracted per-year copies under _extracted/
    era5_hourly_tp/  era5_tp_hourly_{year}.nc, 24 hourly total_precipitation values/day — the Bug 2 fix, download_era5_hourly_tp.py
    gefs/    GEFSv12 reforecast apcp_sfc GRIB2, one file per init date
  processed/
    subdivision_boundaries.geojson   33 subdivisions, our codes/names, WGS84
    _features_{year}.parquet         per-year raw pairing table (pre-bust-formula), gitignored
    _parts/{year}/{YYYYMMDD}.parquet per-date intermediates from the parallel extractor, gitignored
    bust_dataset.parquet             final output table (committed)
    bust_dataset.csv                 same data, gitignored (135MB, exceeds GitHub's limit)
  scripts/
    subdivisions.csv                 33 subdivisions: id, code, name (source of truth, matches frontend)
    grid_utils.py                    grid-cell -> subdivision assignment (shared)
    build_subdivision_boundaries.py  36 raw IMD subdivisions -> our 33
    atomic_io.py                     write-then-rename parquet helper (no truncated files on a kill)
    download_imd.py / download_gefs.py   download_gefs retries transient network errors with backoff
    download_era5_hourly_tp.py       24-hourly total_precipitation pull, the Bug 2 fix
    imd_utils.py / era5_utils.py / gefs_utils.py   per-source loading + aggregation
    extract_features.py              sequential, per-year, resumable (checkpoints every 20 dates)
    extract_features_parallel.py     process-pool version (see below) — this is what was actually used
    compute_bust.py                  combines years, applies the locked bust formula
    patch_schema.py                  one-time: added wind_u10/wind_v10/total_precipitation, renamed columns to the handoff schema
    fix_era5_leakage_and_precip.py   one-time: rebuilt all 7 ERA5 columns to fix Bugs 1 and 2 above
  .venv/     (gitignored) Python environment for these scripts
```

## Parallel extraction

`extract_features_parallel.py` is what actually produced this dataset —
the sequential `extract_features.py` remains as a simpler fallback (and is
what 2015 and part of 2016 were originally run with; both produce identical
output). Each date is one unit of work — download the GEFS file, decode,
aggregate, delete the raw file, write one atomic per-date part file to
`_parts/{year}/{date}.parquet` — handed to a pool of worker **processes**
(not threads: cfgrib/eccodes has real thread-safety caveats for concurrent
C-library calls within one process). No worker ever reads-modifies-writes
a file another worker might also be touching — that pattern silently loses
rows under concurrency (last writer wins, no error). A separate merge step
concatenates all part files into `_features_{year}.parquet`.

Worker count is chosen from currently-available RAM (`psutil`), not
hardcoded — override with `--workers N` if bandwidth, not RAM, is the
binding constraint (see below). Reference data (IMD, ERA5, the GEFS grid
map) is loaded once in the parent process and handed to every worker via
the pool initializer, rather than reloaded redundantly by each one.

A date whose download fails outlasts `download_gefs.py`'s own retries
(4 attempts, exponential backoff) is reported by name at the end and
skipped — it will not silently appear as a false zero in the output. Retry
just the failed dates, then re-run the plain per-year command to trigger
the merge:
```
data/.venv/bin/python3 data/scripts/extract_features_parallel.py 2018 --dates 20181125,20181127
data/.venv/bin/python3 data/scripts/extract_features_parallel.py 2018   # sees them done now, merges
```

**On network variability**: actual throughput swung enormously while
building this dataset — from ~3 sec/date (7 workers, good conditions) to
~45 sec/date (3 workers, degraded connection) over the course of one
afternoon, confirmed by direct speed tests against both the S3 bucket and
an unrelated host (not an S3-specific throttling issue). Fewer workers were
used deliberately during the slow stretch, since once bandwidth — not
RAM or CPU — is the bottleneck, more concurrent connections mostly add
contention rather than throughput. 4 dates failed even after retries
during the worst stretch (`20181125`, `20181127`, `20181128`, `20181230`)
and were retried individually once conditions improved.

## Reproducing / extending

```
data/.venv/bin/python3 data/scripts/build_subdivision_boundaries.py
data/.venv/bin/python3 data/scripts/download_imd.py 2015 2016 2017 2018 2019
data/.venv/bin/python3 data/scripts/download_era5_hourly_tp.py 2015 2016 2017 2018 2019
data/.venv/bin/python3 data/scripts/extract_features_parallel.py 2015   # repeat per year; downloads GEFS as needed
data/.venv/bin/python3 data/scripts/compute_bust.py 2015 2016 2017 2018 2019
data/.venv/bin/python3 data/scripts/patch_schema.py 2015 2016 2017 2018 2019           # adds wind/tp, renames to handoff schema
data/.venv/bin/python3 data/scripts/fix_era5_leakage_and_precip.py 2015 2016 2017 2018 2019   # fixes Bugs 1 and 2, see "Known issues" above
```

A from-scratch run of `extract_features_parallel.py` no longer needs the
`patch_schema.py` / `fix_era5_leakage_and_precip.py` follow-up steps at
all — both fixes now live in `era5_utils.py` and are applied automatically
during extraction. They're listed here because that's how this dataset's
history actually went: built, then patched twice as bugs were found. A
fresh rebuild only needs the first four commands.
