"""One VishwasService per process, shared by the ML routes and reloaded
after a retraining job finishes. Kept in its own module (not inside app.py)
so jobs.py can reload it without importing the FastAPI app / causing a
circular import.

Model bundles:
  ml/models/<source>/            committed baseline, never written at runtime
  <runtime_dir>/<source>/        written by retraining jobs; loaded in preference
                                 to the baseline when it exists
runtime_dir = $VISHWAS_RUNTIME_MODELS or backend/runtime_models (gitignored).
Retraining used to overwrite ml/models/ in place, so a single retrain (or a
test run) silently changed committed files.
"""
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
from vishwas_ml.config import load_config  # noqa: E402
from vishwas_ml.service import VishwasService  # noqa: E402

_lock = threading.Lock()
_svc = None
_source = os.environ.get("VISHWAS_SOURCE", "ncmrwf")


def runtime_dir(source: str | None = None) -> Path:
    base = Path(os.environ.get("VISHWAS_RUNTIME_MODELS",
                               os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_models")))
    return base / (source or _source)


def active_models_dir(source: str | None = None) -> Path:
    """The retrained bundle if one exists, otherwise the committed baseline."""
    rt = runtime_dir(source)
    if (rt / "booster.txt").exists() and (rt / "metadata.json").exists():
        return rt
    return load_config(source=source or _source)["models_dir"]


def _build(source: str) -> VishwasService:
    return VishwasService(models_dir=active_models_dir(source), cfg=load_config(source=source))


def get_service() -> VishwasService:
    global _svc
    with _lock:
        if _svc is None:
            _svc = _build(_source)
        return _svc


def reload_service(source: str | None = None) -> VishwasService:
    global _svc, _source
    if source:
        _source = source
    with _lock:
        _svc = _build(_source)
        return _svc


def current_source() -> str:
    return _source
