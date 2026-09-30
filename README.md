# Vishwas: Forecast Trust Console

SIH 2026 PS 26079: AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts (NCMRWF / Ministry of Earth Sciences). Team PowerPuff Squad.

Vishwas gives, for each of India's 33 meteorological subdivisions and each lead day from Day 1 to Day 10, a calibrated probability that an NCMRWF rainfall forecast will bust, a forecast confidence map, IMD-coded warning levels, plain-language reasons, the closest historical case and a model self-confidence value, in a secure role-based web console.

**Start here: [docs/VISHWAS_HANDBOOK.md](docs/VISHWAS_HANDBOOK.md)**. It covers architecture, data, ML, website, API, setup, runbooks and the presentation material.

## Run it (Windows; use `.venv/bin/` on Linux/macOS)

```bash
cd ml && python -m venv .venv && .venv/Scripts/pip install -r requirements.txt -r ../backend/requirements.txt
cd ../backend && ../ml/.venv/Scripts/python -m uvicorn app:app --port 8000      # API  → http://localhost:8000/docs
cd ../frontend && ../ml/.venv/Scripts/python -m http.server 5500                # site → http://localhost:5500
```

Sign in as `forecaster`, `senior`, `admin` or `observer` with password `vishwas123`.

| Folder | Contents |
|---|---|
| `frontend/` | The web console (single HTML file + India map) |
| `backend/` | FastAPI + SQLite: auth, roles, outcomes, admin, retraining |
| `ml/` | ML package, trained models, data, scripts, tests ([ml/README.md](ml/README.md)) |
| `data/` | GEFS + ERA5 comparison-track data pipeline |
| `docs/` | Handbook and project history |
