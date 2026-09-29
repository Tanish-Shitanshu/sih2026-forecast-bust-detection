"""Train, calibrate, evaluate and save the model bundle.

    python ml/scripts/train.py                       # data path from ml/config.json
    python ml/scripts/train.py --data path/to/real_pairs.parquet --out ml/models
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vishwas_ml.config import load_config  # noqa: E402
from vishwas_ml.pipeline import train_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="synthetic | ncmrwf | gefs (default: config.json)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None, help="override data_path from the config")
    ap.add_argument("--out", default=None, help="override models_dir from the config")
    a = ap.parse_args()
    res = train_pipeline(load_config(a.config, source=a.source), data_path=a.data, out_dir=a.out)
    m = res["metrics"]["overall"]["model"]
    print("\nTest-year summary (calibrated model)")
    for k in ("n", "base_rate", "pr_auc", "roc_auc", "brier_skill_vs_climatology", "ece",
              "precision_at_alert", "recall_at_alert", "precision_at_watch", "recall_at_watch", "catch_at_10pct"):
        print(f"  {k:28s} {m[k]:.4f}" if isinstance(m[k], float) else f"  {k:28s} {m[k]}")
    print("\nTop factor groups (mean |SHAP|):", ", ".join(list(res["importance"]["groups"])[:6]))


if __name__ == "__main__":
    main()
