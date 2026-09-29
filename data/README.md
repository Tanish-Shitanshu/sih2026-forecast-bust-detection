# Rainfall bust-detection dataset — SIH 2026 PS 26079 (Vishwas)

**This is the GEFS+ERA5 backup/comparison track, not the primary NCMRWF
track.** NCMRWF's own RDS portal was checked first and found to offer only
reanalysis products (no Day 1–10 medium-range reforecast archive — see
"Sources" below), so this dataset pairs NOAA's public GEFSv12 reforecast
against IMD ground truth instead, as a working, fully-real alternative to
train and validate against while a primary NCMRWF-sourced dataset is
pursued separately.

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

**Why GEFS and not NCMRWF RDS.** `rds.ncmrwf.gov.in` was checked first, per
the task's instructions. It offers only *reanalysis* products — IMDAA
(1979–2020) and NGFS (1999–2019) — plus one sub-seasonal-to-seasonal
reforecast (1993–2015) that doesn't match our daily Day 1–10 medium-range
horizon. This was confirmed from the site's own public dataset catalog,
without ever hitting a registration wall. Per the task's fallback
instruction, NOAA's public GEFSv12 reforecast was used instead: the
control member (`c00`), total precipitation (`apcp_sfc`), for each
forecast issue date.

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
| `mslp` | ERA5 mean sea level pressure, subdivision-mean, daily mean of the 4 synoptic hours, at the **init date** | Pa |
| `surface_pressure` | ERA5 surface pressure, same aggregation | Pa |
| `dewpoint_2m` | ERA5 2m dewpoint temperature, same aggregation (moisture proxy) | K |
| `temp_2m` | ERA5 2m temperature, same aggregation | K |
| `wind_u10` | ERA5 10m eastward wind component, same aggregation | m/s |
| `wind_v10` | ERA5 10m northward wind component, same aggregation | m/s |
| `total_precipitation` | ERA5's own precipitation estimate (reanalysis, independent of both `forecast_rain` and `observed_rain`), subdivision-mean, daily **sum** of the 4 non-overlapping 6h accumulations | mm |

Plus 4 columns beyond the requested handoff schema, kept because they're
useful and don't conflict with it — flag if a strict schema match is
needed instead and these should be dropped:

| Column | Meaning |
|---|---|
| `subdivision_name` | IMD subdivision name, matches `frontend/index.html` |
| `magnitude_threshold_mm` | The actual per-(subdivision, lead_day) magnitude threshold applied to this row (mm) |
| `year` | Calendar year of `date` |
| `split` | Whether this row's year was used to fit `magnitude_threshold_mm` — `train` or `test` |

All ERA5 features describe atmospheric conditions **at forecast issue
time** (not per lead day) — they're a feature of the forecast run itself,
constant across a given date+subdivision's 10 lead-day rows.

`bust_dataset.csv` (same data, gitignored — 135MB, over GitHub's 100MB
hard limit) can be regenerated locally: `data/.venv/bin/python3 -c
"import pandas as pd; pd.read_parquet('data/processed/bust_dataset.parquet').to_csv('data/processed/bust_dataset.csv', index=False)"`.

## Layout

```
data/
  raw/
    imd/     IMD yearly NetCDF (ind{year}_rfp25.nc), + the raw and validated subdivision boundary GeoJSONs
    era5/    era5_{year}.nc (actually zip archives — see era5_utils.py), extracted per-year copies under _extracted/
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
    imd_utils.py / era5_utils.py / gefs_utils.py   per-source loading + aggregation
    extract_features.py              sequential, per-year, resumable (checkpoints every 20 dates)
    extract_features_parallel.py     process-pool version (see below) — this is what was actually used
    compute_bust.py                  combines years, applies the locked bust formula
    patch_schema.py                  one-time: added wind_u10/wind_v10/total_precipitation, renamed columns to the handoff schema
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
data/.venv/bin/python3 data/scripts/extract_features_parallel.py 2015   # repeat per year; downloads GEFS as needed
data/.venv/bin/python3 data/scripts/compute_bust.py 2015 2016 2017 2018 2019
data/.venv/bin/python3 data/scripts/patch_schema.py 2015 2016 2017 2018 2019   # adds wind/tp, renames to handoff schema
```
