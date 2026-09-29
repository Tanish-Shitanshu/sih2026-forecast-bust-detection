"""The pairs table schema shared with the data/ workstream, plus a loader and validator.

One row = one (issue date, subdivision, lead day). `observed_rain` is the IMD value
for the forecast's valid date (date + lead_day).
"""
import warnings

import numpy as np
import pandas as pd

KEYS = ["date", "subdivision_code", "lead_day"]
ERA5 = ["mslp", "surface_pressure", "dewpoint_2m", "temp_2m", "wind_u10", "wind_v10", "total_precipitation"]
COLUMNS = KEYS + ["forecast_rain", "observed_rain", "error", "is_bust", "trigger_reason"] + ERA5
# Must be present. Everything else is optional: missing ERA5 columns become NaN features,
# error / is_bust / trigger_reason are recomputed by the ML pipeline anyway.
REQUIRED = KEYS + ["forecast_rain", "observed_rain"]
# Optional ensemble information (e.g. NCMRWF S2S members): spread of forecast_rain across members.
OPTIONAL_FORECAST = ["forecast_rain_spread"]
# Known only after the valid date. Never used as model inputs.
OUTCOME_COLS = ["observed_rain", "error", "is_bust", "trigger_reason"]
TRIGGERS = ("magnitude", "category", "both")
LEAD_DAYS = range(1, 11)

# Column names used by the data/ workstream (DATA_PIPELINE_STATUS.md) and common variants.
ALIASES = {
    "forecast_rain_mm": "forecast_rain", "observed_rain_mm": "observed_rain", "error_mm": "error",
    "forecast_rain_spread_mm": "forecast_rain_spread", "forecast_rain_std_mm": "forecast_rain_spread",
    "era5_msl_pa": "mslp", "era5_msl": "mslp", "era5_mslp": "mslp", "msl": "mslp",
    "era5_sp_pa": "surface_pressure", "era5_sp": "surface_pressure", "sp": "surface_pressure",
    "era5_d2m_k": "dewpoint_2m", "era5_d2m": "dewpoint_2m", "d2m": "dewpoint_2m",
    "era5_t2m_k": "temp_2m", "era5_t2m": "temp_2m", "t2m": "temp_2m",
    "era5_u10_ms": "wind_u10", "era5_u10": "wind_u10", "u10": "wind_u10",
    "era5_v10_ms": "wind_v10", "era5_v10": "wind_v10", "v10": "wind_v10",
    "era5_tp_m": "total_precipitation", "era5_tp_mm": "total_precipitation", "era5_tp": "total_precipitation",
    "tp": "total_precipitation",
}


class SchemaError(ValueError):
    pass


def normalize_columns(df):
    """Rename known aliases to the canonical schema and add optional columns as NaN.
    Returns (df, list of optional columns that were absent)."""
    df = df.rename(columns={c: ALIASES[c.lower()] for c in df.columns
                            if c.lower() in ALIASES and ALIASES[c.lower()] not in df.columns})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing} (have {list(df.columns)})")
    absent = [c for c in COLUMNS + OPTIONAL_FORECAST if c not in df.columns]
    df = df.copy()
    if "error" not in df.columns:
        df["error"] = df["forecast_rain"] - df["observed_rain"]
    for c in absent:
        if c not in df.columns:
            df[c] = None if c in ("is_bust", "trigger_reason") else np.nan
    return df, [c for c in absent if c != "error"]


def load_pairs(path):
    """Read the pairs table (.parquet or .csv), map column names, coerce types."""
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return coerce(df)


def coerce(df):
    df, absent = normalize_columns(df)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    df["subdivision_code"] = df["subdivision_code"].astype(str)
    df["lead_day"] = df["lead_day"].astype(int)
    for c in ["forecast_rain", "observed_rain", "error", "forecast_rain_spread"] + ERA5:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    def _b(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return pd.NA
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return str(v).strip().lower() in ("true", "1", "1.0")
    df["is_bust"] = pd.array([_b(v) for v in df["is_bust"]], dtype="boolean")
    tr = df["trigger_reason"].astype(object)
    df["trigger_reason"] = tr.where(tr.isin(TRIGGERS), None)
    df = df.sort_values(KEYS, kind="stable").reset_index(drop=True)
    df.attrs["absent_columns"] = absent
    return df


def detect_era5_timing(df):
    """'issue' if ERA5 values are constant across lead days for each (date, subdivision),
    'valid' if they change with lead day (i.e. they are the valid-date state)."""
    cols = [c for c in ERA5 if c in df.columns and df[c].notna().any()]
    if not cols:
        return "issue"
    g = df.groupby(["date", "subdivision_code"])[cols].nunique(dropna=True)
    return "valid" if (g > 1).any(axis=None) else "issue"


def validate(df, codes, strict=True):
    """Check the table against the schema. Raises SchemaError on problems that break the
    pipeline; returns a report dict and emits warnings for softer issues."""
    problems, report = [], {}
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    unknown = sorted(set(df["subdivision_code"]) - set(codes))
    if unknown:
        problems.append(f"unknown subdivision codes (not in frontend list): {unknown}")
    absent = sorted(set(codes) - set(df["subdivision_code"]))
    if absent:
        warnings.warn(f"subdivisions with no rows: {absent}")
    bad_lead = sorted(set(df["lead_day"]) - set(LEAD_DAYS))
    if bad_lead:
        problems.append(f"lead_day outside 1..10: {bad_lead}")
    dups = int(df.duplicated(KEYS).sum())
    if dups:
        problems.append(f"{dups} duplicate (date, subdivision_code, lead_day) rows")
    if df["forecast_rain"].isna().any():
        problems.append(f"{int(df['forecast_rain'].isna().sum())} rows with missing forecast_rain")
    if strict and problems:
        raise SchemaError("; ".join(problems))

    lab = df["observed_rain"].notna()
    err_gap = (df.loc[lab, "forecast_rain"] - df.loc[lab, "observed_rain"] - df.loc[lab, "error"]).abs()
    report["error_mismatch_rows"] = int((err_gap > 0.05).sum())
    if report["error_mismatch_rows"]:
        warnings.warn(f"{report['error_mismatch_rows']} rows where error != forecast_rain - observed_rain")

    # The same valid date must have the same observation whatever the issue date.
    vd = df.loc[lab, ["subdivision_code"]].assign(valid=df.loc[lab, "date"] + pd.to_timedelta(df.loc[lab, "lead_day"], "D"),
                                                  obs=df.loc[lab, "observed_rain"])
    spread = vd.groupby(["subdivision_code", "valid"])["obs"].agg(lambda s: s.max() - s.min())
    report["obs_inconsistent_valid_dates"] = int((spread > 0.05).sum())
    if report["obs_inconsistent_valid_dates"]:
        warnings.warn(f"{report['obs_inconsistent_valid_dates']} (subdivision, valid date) pairs have different "
                      "observed_rain across issue dates; check that observed_rain is the valid-date value")

    known = df["is_bust"].notna().to_numpy() if "is_bust" in df else np.zeros(len(df), bool)
    ib = df["is_bust"].fillna(False).to_numpy(bool) if known.any() else np.zeros(len(df), bool)
    has_reason = df["trigger_reason"].notna().to_numpy()
    report["trigger_reason_inconsistent"] = int((known & (ib != has_reason)).sum())
    report["absent_optional_columns"] = list(df.attrs.get("absent_columns", []))
    if report["absent_optional_columns"]:
        warnings.warn(f"optional columns absent (features will be NaN / labels recomputed): "
                      f"{report['absent_optional_columns']}")
    report["era5_timing_detected"] = detect_era5_timing(df)
    report["rows"] = int(len(df))
    report["dates"] = int(df["date"].nunique())
    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    report["provided_bust_rate"] = float(ib[known].mean()) if known.any() else None
    report["era5_missing_frac"] = {c: round(float(df[c].isna().mean()), 4) if c in df else 1.0 for c in ERA5}
    return report
