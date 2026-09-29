"""Retraining runs off the request thread: POST /api/v1/retraining starts a
Python thread and returns immediately (202); GET /api/v1/retraining/{job_id}
polls status. A single global flag (not per-source) keeps this simple and
matches the API page's single 409 "already in progress" error -- fine for
a one-process hackathon demo.
"""
import json
import sys
import os
import threading
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
from vishwas_ml.pipeline import train_pipeline  # noqa: E402
from vishwas_ml.config import load_config  # noqa: E402

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


def _run(job_id: str, source: str, bust_overrides: dict | None):
    global _running
    conn = db.get_conn()
    conn.execute("UPDATE retraining_jobs SET status = 'running' WHERE job_id = ?", (job_id,))
    conn.commit()
    try:
        cfg = load_config(source=source)
        if bust_overrides:
            cfg["bust"] = {**cfg.get("bust", {}), **bust_overrides}
        result = train_pipeline(cfg)
        overall = result["metrics"]["overall"]["model"]
        summary = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in overall.items()}
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
        with _state_lock:
            _running = False


def get_job(job_id: str):
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM retraining_jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    return row
