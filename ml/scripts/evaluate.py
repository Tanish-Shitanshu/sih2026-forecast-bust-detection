"""Re-evaluate the saved bundle on the test year and print per-lead / per-subdivision tables.

    python ml/scripts/evaluate.py [--data path] [--models ml/models]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vishwas_ml.config import load_config  # noqa: E402
from vishwas_ml.model import BustModel  # noqa: E402
from vishwas_ml.pipeline import evaluate, prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--models", default=None)
    a = ap.parse_args()
    cfg = load_config()
    out = Path(a.models or cfg["models_dir"])
    model = BustModel.load(out)
    df, _, _, split, _, _ = prepare(cfg, a.data)
    metrics, per_sub, _ = evaluate(model, df, split, cfg, out)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    ov = pd.DataFrame(metrics["overall"]).T[["base_rate", "pr_auc", "roc_auc", "brier", "brier_skill_vs_climatology",
                                              "ece", "precision_at_alert", "recall_at_alert", "accuracy_at_0_5"]]
    print(f"Test years {metrics['split_years']['test']}, n = {metrics['n_test']:,}\n")
    print(ov.round(3).to_string(), "\n")
    print("Recall at Orange+ by trigger:", {k: round(v, 3) for k, v in metrics["recall_at_alert_by_trigger"].items()}, "\n")
    pl = pd.DataFrame(metrics["per_lead"])
    print(pl[["lead_day", "base_rate", "pr_auc", "pr_auc_forecast_only", "pr_auc_climatology", "precision_at_alert",
              "recall_at_alert", "ece"]].round(3).to_string(index=False), "\n")
    print(per_sub[["subdivision_code", "n", "base_rate", "pr_auc", "precision_at_alert", "recall_at_alert"]]
          .sort_values("pr_auc").round(3).to_string(index=False))
    print(f"\nwrote {out / 'evaluation'}")


if __name__ == "__main__":
    main()
