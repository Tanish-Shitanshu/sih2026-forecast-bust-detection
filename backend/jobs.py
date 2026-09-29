"""Retraining runs off the request thread: POST /api/v1/retraining starts a
Python thread and returns immediately (202); GET /api/v1/retraining/{job_id}
polls status. A single global flag (not per-source) keeps this simple and
matches the API page's single 409 "already in progress" error -- fine for
a one-process hackathon demo.

A run does two things, in order:
  1. train_pipeline(cfg) with the admin's saved bust definition, written to the
     runtime model folder (service_registry.runtime_dir), never to ml/models/.
  2. recalibration from approved forecaster outcomes (vishwas_ml.feedback):
     applied only when there are enough of them (config feedback.min_outcomes);
     the job result says whether it was applied or skipped and why.
"""
import json
import os
import shutil
import sys
import threading
import uuid

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
from vishwas_ml.config import load_config  # noqa: E402
from vishwas_ml.feedback import recalibrate  # noqa: E402
from vishwas_ml.model import BustModel  # noqa: E402
from vishwas_ml.pipeline import train_pipeline  # noqa: E402

import db
import service_registry

_state_lock = threading.Lock()
_running = False


def is_running() -> bool:
    with _state_lock:
        return _running


def start_job(source: str, approved_outcomes_used: int, bust_overrides: dict | None = None) -> str:
    global _running
    with _state_lock:
        if _running:
            raise RuntimeError("A retraining run is already in progress")
        _running = True

    job_id = f"rt-{uuid.uuid4().hex[:8]}"
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO retraining_jobs (job_id, status, source, approved_outcomes_used, started_at) "
        "VALUES (?, 'queued', ?, ?, ?)",
        (job_id, source, approved_outcomes_used, db.now_iso()),
    )
    conn.commit()
    conn.close()

    threading.Thread(target=_run, args=(job_id, source, bust_overrides), daemon=True).start()
    return job_id


def _approved_outcomes() -> pd.DataFrame:
    conn = db.get_conn()
    rows = conn.execute("SELECT predicted_bust_probability, outcome, status FROM outcomes "
                        "WHERE status = 'approved'").fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows],
                        columns=["predicted_bust_probability", "outcome", "status"])


def _run(job_id: str, source: str, bust_overrides: dict | None):
    global _running
    conn = db.get_conn()
    conn.execute("UPDATE retraining_jobs SET status = 'running' WHERE job_id = ?", (job_id,))
    conn.commit()
    staging = None
    try:
        cfg = load_config(source=source)
        if bust_overrides:
            cfg["bust"] = {**cfg.get("bust", {}), **bust_overrides}
        target = service_registry.runtime_dir(source)
        staging = target.with_name(target.name + f".{job_id}.tmp")
        if staging.exists():
            shutil.rmtree(staging)
        result = train_pipeline(cfg, out_dir=staging, log=lambda *a: None)

        model = BustModel.load(staging)
        rc, recal = recalibrate(model, _approved_outcomes(),
                                pd.read_parquet(staging / "calibration_set.parquet"), cfg["feedback"])
        if rc is not None:
            model.recalibrator = rc
            model.metadata["recalibration"] = recal
            model.save(staging)

        # swap the finished bundle in only once it is complete
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        staging = None

        overall = result["metrics"]["overall"]["model"]
        summary = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in overall.items()}
        summary["recalibration"] = recal
        conn.execute(
            "UPDATE retraining_jobs SET status = 'done', finished_at = ?, metrics_json = ? WHERE job_id = ?",
            (db.now_iso(), json.dumps(summary), job_id),
        )
        conn.commit()
        service_registry.reload_service(source=source)
    except Exception as e:
        conn.execute(
            "UPDATE retraining_jobs SET status = 'failed', finished_at = ?, error = ? WHERE job_id = ?",
            (db.now_iso(), str(e), job_id),
        )
        conn.commit()
    finally:
        conn.close()
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        with _state_lock:
            _running = False


def get_job(job_id: str):
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM retraining_jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    return row
