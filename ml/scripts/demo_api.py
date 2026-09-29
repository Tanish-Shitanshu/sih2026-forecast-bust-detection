"""Print every ML-backed endpoint's response for one cycle (a contract check for backend).

    python ml/scripts/demo_api.py [--cycle 2023-08-15] [--subdivision KL] [--lead 3]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vishwas_ml.service import VishwasService, dumps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", default=None, help="issue date; default = latest in the data")
    ap.add_argument("--subdivision", default=None, help="default = highest bust risk on this lead day")
    ap.add_argument("--lead", type=int, default=3)
    a = ap.parse_args()
    svc = VishwasService()
    cm = svc.confidence_map(a.lead, a.cycle)
    code = a.subdivision or max(cm["items"], key=lambda x: x["bust_probability"])["code"]
    shown = dict(cm, items=cm["items"][:3], truncated=f"3 of {cm['count']} items shown")
    calls = [
        ("GET /api/v1/confidence-map", shown),
        ("GET /api/v1/bust-probability", svc.bust_probability(code, a.lead, a.cycle)),
        (f"GET /api/v1/subdivisions/{code}/explain", svc.explain(code, a.lead, a.cycle)),
        ("GET /api/v1/model-trust", dict(t := svc.model_trust(lead_day=a.lead, cycle=a.cycle), items=t["items"][:3])),
        ("GET /analogs (prompt extra)", svc.analogs(code, a.lead, a.cycle)),
        ("GET /action (prompt extra)", svc.action(code, a.lead, a.cycle)),
        ("GET /model-trust?region= (prompt extra)",
         dict(t2 := svc.model_trust(subdivision=code, cycle=a.cycle), items=t2["items"][:3])),
    ]
    for name, res in calls:
        print(f"\n=== {name}\n{dumps(res)}")


if __name__ == "__main__":
    main()
