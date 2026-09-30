# External reference data (committed so nothing depends on one person's machine)

| File | What | Built by |
|---|---|---|
| `imd_subdivision_daily_1992_2025.parquet` | IMD 0.25° gridded daily rainfall averaged over the 33 subdivisions (`date` = IMD date, `subdivision_code`, `observed_rain_mm`), 409,827 rows | `scripts/build_s2s_pairs.imd_subdivision_daily` on imdlib files (`scripts/download_imd.py`) |
| `ncmrwf_fc_1993_2015.parquet` | NCMRWF S2S per-lead subdivision fields (rain, MSLP, surface pressure, 10 m winds), 1993–2015 | `scripts/build_s2s_pairs.py` from the RDS portal zips |
| `indian_met_zones/indian_met_zones.v2.*` | IMD meteorological subdivision shapefile | github.com/India-Meteorological-Department/Indian_met_zones |

Used by `scripts/build_neps_pairs.py` (IMD observations) and `experiments/ncmrwf_improve.py`.
