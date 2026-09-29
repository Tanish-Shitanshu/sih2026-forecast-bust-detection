"""Config and subdivision metadata. Paths in config.json are relative to ml/."""
import json
import os
from functools import lru_cache
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ML_DIR / p


def load_config(path=None, source=None):
    """Load config.json and select a forecast-source profile.

    Each source (synthetic, ncmrwf, gefs) has its own data file and its own model bundle
    folder, so its bust thresholds and model never mix with another source's.
    Order: `source` argument, then the VISHWAS_SOURCE environment variable, then config "source"."""
    cfg = json.loads(resolve(path or "config.json").read_text(encoding="utf-8"))
    src = source or os.environ.get("VISHWAS_SOURCE") or cfg.get("source")
    if "sources" in cfg:
        if src not in cfg["sources"]:
            raise ValueError(f"unknown source {src!r}; configured: {sorted(cfg['sources'])}")
        cfg.update(cfg["sources"][src])
    cfg["source"] = src
    for k in ("data_path", "subdivisions_path", "models_dir"):
        cfg[k] = resolve(cfg[k])
    return cfg


@lru_cache(maxsize=4)
def _meta(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_meta(cfg=None):
    """Subdivisions, regions, tiers, trust levels and event sets (synced from the frontend)."""
    cfg = cfg or load_config()
    return _meta(str(cfg["subdivisions_path"]))


def subdivision_codes(meta=None):
    return [s["code"] for s in (meta or load_meta())["subdivisions"]]
