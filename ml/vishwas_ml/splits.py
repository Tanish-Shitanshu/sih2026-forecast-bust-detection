"""Chronological train / calibration / test split by year.

Default ("auto"): last year = test, second-to-last = calibration (early stopping and
isotonic fit), everything earlier = training. Bust thresholds and every fitted feature
table use training years only.
"""
import numpy as np


def year_split(df, split_cfg):
    years = sorted(df["date"].dt.year.unique().tolist())
    test = years[-1:] if split_cfg.get("test_years", "auto") == "auto" else list(split_cfg["test_years"])
    calib = years[-2:-1] if split_cfg.get("calib_years", "auto") == "auto" else list(split_cfg["calib_years"])
    train = [y for y in years if y not in test and y not in calib and y < min(test + calib)]
    if len(years) < 3 or not train:
        raise ValueError(f"need at least 3 years for a train/calibration/test split, got {years}")
    yr = df["date"].dt.year.to_numpy()
    return {
        "train": np.isin(yr, train), "calib": np.isin(yr, calib), "test": np.isin(yr, test),
        "years": {"train": train, "calib": calib, "test": test},
    }
