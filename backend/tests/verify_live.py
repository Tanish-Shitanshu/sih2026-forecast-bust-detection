"""Not a pytest suite -- a scripted walk through the zero-loophole checklist
against the REAL, already-running server (real VishwasService, real SQLite
DB), printing PASS/FAIL for each assertion. Run once the server is up:

    ../ml/.venv/bin/python tests/verify_live.py
"""
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
FAILS = []


def check(label, cond, extra=""):
    ok = bool(cond)
    print(("PASS" if ok else "FAIL") + f" - {label}" + (f"  ({extra})" if extra and not ok else ""))
    if not ok:
        FAILS.append(label)
    return ok


def login(user_id, password="vishwas123"):
    r = httpx.post(f"{BASE}/api/v1/auth/login", json={"id": user_id, "password": password})
    check(f"login {user_id}", r.status_code == 200, r.text)
    return r.json()["token"] if r.status_code == 200 else None


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


print("=== logging in as all 4 demo roles ===")
tok_duty = login("forecaster")
tok_senior = login("senior")
tok_admin = login("admin")
tok_observer = login("observer")

print("\n=== wrong password / unknown user ===")
r = httpx.post(f"{BASE}/api/v1/auth/login", json={"id": "forecaster", "password": "wrong"})
check("wrong password -> 401", r.status_code == 401)
r = httpx.post(f"{BASE}/api/v1/auth/login", json={"id": "nobody", "password": "x"})
check("unknown user -> 401", r.status_code == 401)

print("\n=== no token at all ===")
r = httpx.get(f"{BASE}/api/v1/confidence-map", params={"lead_day": 3})
check("no token -> 401", r.status_code == 401)

print("\n=== every ML route, real VishwasService, cycle=2015-12-01 ===")
CYCLE = "2015-12-01"
h = auth_headers(tok_duty)

r = httpx.get(f"{BASE}/api/v1/cycles", headers=h)
check("GET /cycles -> 200", r.status_code == 200)
check("cycles includes 2015-12-01T00Z", "2015-12-01T00Z" in r.json()["items"])

r = httpx.get(f"{BASE}/api/v1/confidence-map", params={"lead_day": 4, "cycle": CYCLE}, headers=h)
check("GET /confidence-map -> 200", r.status_code == 200)
cm = r.json()
check("confidence-map has 33 items", cm.get("count") == 33 and len(cm.get("items", [])) == 33, str(cm.get("count")))

r = httpx.get(f"{BASE}/api/v1/bust-probability", params={"subdivision": "TN/PY", "lead_day": 4, "cycle": CYCLE}, headers=h)
check("GET /bust-probability -> 200", r.status_code == 200)
bp = r.json()
check("Day-4 TN/PY is Orange ~40% (the documented Chennai floods story)",
      bp.get("level") == "orange" and abs(bp.get("bust_probability", 0) - 0.40) < 0.02, str(bp))

r = httpx.get(f"{BASE}/api/v1/subdivisions/TN%2FPY/explain", params={"lead_day": 4, "cycle": CYCLE}, headers=h)
check("GET /subdivisions/TN%2FPY/explain -> 200 (URL-encoded / edge case)", r.status_code == 200)
check("explain has factors + closest_analog", "factors" in r.json() and "closest_analog" in r.json())

r = httpx.get(f"{BASE}/api/v1/subdivisions/J%26K/explain", params={"lead_day": 3, "cycle": CYCLE}, headers=h)
check("GET /subdivisions/J%26K/explain -> 200 (URL-encoded & edge case)", r.status_code == 200,
      f"status={r.status_code} body={r.text[:200]}")

r = httpx.get(f"{BASE}/api/v1/model-trust", params={"lead_day": 4, "cycle": CYCLE}, headers=h)
check("GET /model-trust?lead_day -> 200", r.status_code == 200)

r = httpx.get(f"{BASE}/api/v1/model-trust", params={"subdivision": "TN/PY", "cycle": CYCLE}, headers=h)
check("GET /model-trust?subdivision -> 200 (extra)", r.status_code == 200)

r = httpx.get(f"{BASE}/api/v1/model-trust", headers=h)
check("GET /model-trust with neither param -> 422", r.status_code == 422)

r = httpx.get(f"{BASE}/api/v1/analogs", params={"subdivision": "TN/PY", "lead_day": 4, "cycle": CYCLE}, headers=h)
check("GET /analogs -> 200 (extra)", r.status_code == 200)

r = httpx.get(f"{BASE}/api/v1/action", params={"subdivision": "TN/PY", "lead_day": 4, "cycle": CYCLE}, headers=h)
check("GET /action -> 200 (extra)", r.status_code == 200)

print("\n=== invalid lead day -> 422, unknown cycle -> 404 (by actually calling, not reading code) ===")
r = httpx.get(f"{BASE}/api/v1/bust-probability", params={"subdivision": "TN/PY", "lead_day": 11, "cycle": CYCLE}, headers=h)
check("lead_day=11 -> 422", r.status_code == 422, f"got {r.status_code}: {r.text}")
r = httpx.get(f"{BASE}/api/v1/bust-probability", params={"subdivision": "TN/PY", "lead_day": 0, "cycle": CYCLE}, headers=h)
check("lead_day=0 -> 422", r.status_code == 422, f"got {r.status_code}: {r.text}")
r = httpx.get(f"{BASE}/api/v1/bust-probability", params={"subdivision": "TN/PY", "lead_day": 4, "cycle": "1901-01-01"}, headers=h)
check("unknown cycle 1901-01-01 -> 404", r.status_code == 404, f"got {r.status_code}: {r.text}")
r = httpx.get(f"{BASE}/api/v1/bust-probability", params={"subdivision": "NOT-A-CODE", "lead_day": 4, "cycle": CYCLE}, headers=h)
check("unknown subdivision -> 422", r.status_code == 422, f"got {r.status_code}: {r.text}")

print("\n=== role boundaries: POSITIVE cases ===")
r = httpx.post(f"{BASE}/api/v1/outcomes",
                json={"subdivision": "TN/PY", "lead_day": 4, "outcome": "incorrect", "note": "Day 4 test", "cycle": CYCLE},
                headers=auth_headers(tok_duty))
check("duty CAN POST /outcomes -> 201", r.status_code == 201, f"got {r.status_code}: {r.text}")
duty_outcome_id = r.json().get("id")

r = httpx.post(f"{BASE}/api/v1/outcomes",
                json={"subdivision": "TN/PY", "lead_day": 5, "outcome": "correct", "cycle": CYCLE},
                headers=auth_headers(tok_senior))
check("senior CAN POST /outcomes -> 201", r.status_code == 201, f"got {r.status_code}: {r.text}")
senior_outcome_id = r.json().get("id")

print("\n=== role boundaries: NEGATIVE cases, explicit ===")
r = httpx.post(f"{BASE}/api/v1/outcomes",
                json={"subdivision": "TN/PY", "lead_day": 6, "outcome": "correct", "cycle": CYCLE},
                headers=auth_headers(tok_admin))
check("admin CANNOT POST /outcomes -> 403", r.status_code == 403, f"got {r.status_code}: {r.text}")

r = httpx.post(f"{BASE}/api/v1/outcomes",
                json={"subdivision": "TN/PY", "lead_day": 6, "outcome": "correct", "cycle": CYCLE},
                headers=auth_headers(tok_observer))
check("observer CANNOT POST /outcomes -> 403", r.status_code == 403, f"got {r.status_code}: {r.text}")

r = httpx.post(f"{BASE}/api/v1/outcomes/{duty_outcome_id}/review", json={"decision": "approve"}, headers=auth_headers(tok_duty))
check("duty CANNOT review (not senior) -> 403", r.status_code == 403, f"got {r.status_code}: {r.text}")

r = httpx.post(f"{BASE}/api/v1/outcomes/{senior_outcome_id}/review", json={"decision": "approve"}, headers=auth_headers(tok_senior))
check("senior CANNOT review THEIR OWN entry -> 403", r.status_code == 403, f"got {r.status_code}: {r.text}")

r = httpx.post(f"{BASE}/api/v1/outcomes/{duty_outcome_id}/review", json={"decision": "approve"}, headers=auth_headers(tok_senior))
check("senior CAN review someone else's entry -> 200", r.status_code == 200, f"got {r.status_code}: {r.text}")

r = httpx.post(f"{BASE}/api/v1/outcomes/{duty_outcome_id}/review", json={"decision": "approve"}, headers=auth_headers(tok_senior))
check("reviewing an already-reviewed entry -> 409", r.status_code == 409, f"got {r.status_code}: {r.text}")

r = httpx.get(f"{BASE}/api/v1/outcomes", headers=h)
check("GET /outcomes (any role) -> 200", r.status_code == 200)
check("GET /outcomes includes the 16 seeded + new entries", r.json()["count"] >= 16 + 2, str(r.json()["count"]))

print("\n=== admin: users, self-deactivation blocked ===")
r = httpx.get(f"{BASE}/api/v1/users", headers=auth_headers(tok_admin))
check("admin CAN list users -> 200", r.status_code == 200)
check("8 seeded users present", r.json()["count"] == 8, str(r.json()["count"]))

r = httpx.get(f"{BASE}/api/v1/users", headers=auth_headers(tok_duty))
check("duty CANNOT list users -> 403", r.status_code == 403)

r = httpx.post(f"{BASE}/api/v1/users", json={"id": "mkulkarni", "name": "M. Kulkarni", "role": "senior"}, headers=auth_headers(tok_admin))
check("admin CAN create a user -> 201", r.status_code == 201, f"got {r.status_code}: {r.text}")

r = httpx.post(f"{BASE}/api/v1/users", json={"id": "mkulkarni", "name": "M. Kulkarni", "role": "senior"}, headers=auth_headers(tok_admin))
check("duplicate user id -> 409", r.status_code == 409)

r = httpx.post(f"{BASE}/api/v1/users", json={"id": "BAD ID!", "name": "x", "role": "duty"}, headers=auth_headers(tok_admin))
check("invalid user id format -> 422", r.status_code == 422)

r = httpx.patch(f"{BASE}/api/v1/users/admin", json={"active": False}, headers=auth_headers(tok_admin))
check("admin CANNOT deactivate THEMSELVES -> 403", r.status_code == 403, f"got {r.status_code}: {r.text}")

r = httpx.patch(f"{BASE}/api/v1/users/mkulkarni", json={"role": "observer"}, headers=auth_headers(tok_admin))
check("admin CAN change someone else's role -> 200", r.status_code == 200, f"got {r.status_code}: {r.text}")

print("\n=== settings ===")
r = httpx.put(f"{BASE}/api/v1/settings/bust-definition", json={"error_threshold_mm": 30, "regional_percentile": 96},
              headers=auth_headers(tok_admin))
check("admin CAN set bust-definition -> 200", r.status_code == 200, f"got {r.status_code}: {r.text}")
r = httpx.put(f"{BASE}/api/v1/settings/bust-definition", json={"error_threshold_mm": 30, "regional_percentile": 96},
              headers=auth_headers(tok_duty))
check("duty CANNOT set bust-definition -> 403", r.status_code == 403)
r = httpx.put(f"{BASE}/api/v1/settings/bust-definition", json={"error_threshold_mm": 9999, "regional_percentile": 96},
              headers=auth_headers(tok_admin))
check("out-of-range error_threshold_mm -> 422", r.status_code == 422)

print("\n=== audit ===")
r = httpx.get(f"{BASE}/api/v1/audit", headers=auth_headers(tok_admin))
check("admin CAN read audit -> 200", r.status_code == 200)
check("audit log recorded the user-add and settings actions", len(r.json()["items"]) >= 2, str(r.json()))
r = httpx.get(f"{BASE}/api/v1/audit", headers=auth_headers(tok_observer))
check("observer CANNOT read audit -> 403", r.status_code == 403)

print("\n=== CORS headers present (actual response, not assumed) ===")
r = httpx.options(f"{BASE}/api/v1/cycles", headers={"Origin": "http://localhost:5500",
                   "Access-Control-Request-Method": "GET"})
check("OPTIONS preflight -> allow-origin header present", "access-control-allow-origin" in {k.lower() for k in r.headers.keys()},
      dict(r.headers))

print("\n=== retraining: off the request thread, then poll status ===")
t0 = time.time()
r = httpx.post(f"{BASE}/api/v1/retraining", headers=auth_headers(tok_admin))
elapsed = time.time() - t0
check("POST /retraining -> 202", r.status_code == 202, f"got {r.status_code}: {r.text}")
check(f"POST /retraining returned FAST ({elapsed:.2f}s, not blocking ~30s training)", elapsed < 5)
job = r.json()
job_id = job.get("job_id")
check("retraining response has job_id + status=queued", bool(job_id) and job.get("status") == "queued", str(job))

r2 = httpx.post(f"{BASE}/api/v1/retraining", headers=auth_headers(tok_admin))
check("second concurrent retraining -> 409", r2.status_code == 409, f"got {r2.status_code}: {r2.text}")

print(f"polling GET /retraining/{job_id} until done (up to 120s)...")
status = None
for _ in range(60):
    r = httpx.get(f"{BASE}/api/v1/retraining/{job_id}", headers=auth_headers(tok_admin))
    status = r.json()
    print(f"  status: {status.get('status')}")
    if status.get("status") in ("done", "failed"):
        break
    time.sleep(2)
check("retraining job reached a terminal state", status and status.get("status") in ("done", "failed"), str(status))
check("retraining job finished successfully", status and status.get("status") == "done", str(status))
if status and status.get("status") == "done":
    check("retraining response includes metrics", "metrics" in status, str(status))

print(f"\n{'='*50}\n{len(FAILS)} FAILURE(S)" if FAILS else f"\n{'='*50}\nALL CHECKS PASSED")
if FAILS:
    for f in FAILS:
        print("  -", f)
sys.exit(1 if FAILS else 0)
