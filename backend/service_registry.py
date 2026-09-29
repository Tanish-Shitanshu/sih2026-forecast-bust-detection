"""One VishwasService per process, shared by the ML routes and reloaded
after a retraining job finishes. Kept in its own module (not inside app.py)
so jobs.py can reload it without importing the FastAPI app / causing a
circular import.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
from vishwas_ml.config import load_config  # noqa: E402
from vishwas_ml.service import VishwasService  # noqa: E402

_lock = threading.Lock()
_svc = None
_source = os.environ.get("VISHWAS_SOURCE", "ncmrwf")


def get_service() -> VishwasService:
    global _svc
    with _lock:
        if _svc is None:
            _svc = VishwasService(cfg=load_config(source=_source))
        return _svc


def reload_service(source: str | None = None) -> VishwasService:
    global _svc, _source
    if source:
        _source = source
    with _lock:
        _svc = VishwasService(cfg=load_config(source=_source))
        return _svc


def current_source() -> str:
    return _source
