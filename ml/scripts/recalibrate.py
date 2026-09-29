"""Refit the probability map from approved forecaster outcomes (no retraining).

    python ml/scripts/recalibrate.py --outcomes approved_outcomes.csv
    python ml/scripts/recalibrate.py --outcomes ml/data/feedback_dev.jsonl --reset   # drop it again

The outcomes file needs: predicted_bust_probability, outcome (correct/incorrect/partial),
status (only "approved" rows are used). The backend exports this from its DB.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vishwas_ml.config import load_config  # noqa: E402
from vishwas_ml.feedback import read_outcomes, recalibrate  # noqa: E402
from vishwas_ml.model import BustModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="synthetic | ncmrwf | gefs (default: config.json)")
    ap.add_argument("--outcomes", required=False)
    ap.add_argument("--models", default=None)
    ap.add_argument("--reset", action="store_true", help="remove the recalibration layer")
    a = ap.parse_args()
    cfg = load_config(source=a.source)
    d = Path(a.models or cfg["models_dir"])
    model = BustModel.load(d)
    if a.reset:
        model.recalibrator = None
        model.save(d)
        print("recalibration layer removed")
        return
    model.recalibrator = None  # fit on the base calibrated probability
    rc, summary = recalibrate(model, read_outcomes(a.outcomes), pd.read_parquet(d / "calibration_set.parquet"),
                              cfg["feedback"])
    print(summary)
    if rc is not None:
        model.recalibrator = rc
        model.metadata["recalibration"] = summary
        model.save(d)
        print(f"saved {d / 'recalibrator.json'}")


if __name__ == "__main__":
    main()
