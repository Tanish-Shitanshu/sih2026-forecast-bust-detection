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
# Known only after the valid date. Never used as model inputs.
OUTCOME_COLS = ["observed_rain", "error", "is_bust", "trigger_reason"]
TRIGGERS = ("magnitude", "category", "both")
LEAD_DAYS = range(1, 11)


class SchemaError(ValueError):
    pass


def load_pairs(path):
    """Read the pairs table (.parquet or .csv) and coerce column types."""
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{path} is missing columns: {missing}")
    return coerce(df)


def coerce(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    df["subdivision_code"] = df["subdivision_code"].astype(str)
    df["lead_day"] = df["lead_day"].astype(int)
    for c in ["forecast_rain", "observed_rain", "error"] + ERA5:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    if "is_bust" in df.columns:
        df["is_bust"] = df["is_bust"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")
                                          if not isinstance(v, (bool, np.bool_)) else bool(v)).astype(bool)
    if "trigger_reason" in df.columns:
        tr = df["trigger_reason"].astype(object)
        df["trigger_reason"] = tr.where(tr.isin(TRIGGERS), None)
    return df.sort_values(KEYS, kind="stable").reset_index(drop=True)


def detect_era5_timing(df):
    """'issue' if ERA5 values are constant across lead days for each (date, subdivision),
    'valid' if they change with lead day (i.e. they are the valid-date state)."""
    g = df.groupby(["date", "subdivision_code"])[ERA5].nunique(dropna=True)
    return "valid" if (g > 1).any(axis=None) else "issue"


def validate(df, codes, strict=True):
    """Check the table against the schema. Raises SchemaError on problems that break the
    pipeline; returns a report dict and emits warnings for softer issues."""
    problems, report = [], {}
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
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

    bust_no_reason = int((df["is_bust"] & df["trigger_reason"].isna()).sum())
    reason_no_bust = int((~df["is_bust"] & df["trigger_reason"].notna()).sum())
    report["trigger_reason_inconsistent"] = bust_no_reason + reason_no_bust
    report["era5_timing_detected"] = detect_era5_timing(df)
    report["rows"] = int(len(df))
    report["dates"] = int(df["date"].nunique())
    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    report["provided_bust_rate"] = float(df.loc[lab, "is_bust"].mean()) if lab.any() else None
    report["era5_missing_frac"] = {c: round(float(df[c].isna().mean()), 4) for c in ERA5}
    return report
