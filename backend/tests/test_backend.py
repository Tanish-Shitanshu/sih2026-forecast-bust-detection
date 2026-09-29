"""Backend-only tests: auth, roles, persistence, and the two API-page
additions (login, retraining status). The ML routes are hit through the
REAL VishwasService (TestClient boots the real app, real model, real
service_registry) -- nothing here is mocked. ml/tests covers the model
itself; this covers what the backend adds around it.
"""
import time

from conftest import CYCLE


# ---------- auth ----------

def test_login_all_four_demo_roles(client):
    for user_id in ("forecaster", "senior", "admin", "observer"):
        r = client.post("/api/v1/auth/login", json={"id": user_id, "password": "vishwas123"})
        assert r.status_code == 200
        assert r.json()["role"] in ("duty", "senior", "admin", "observer")


def test_login_wrong_password(client):
    r = client.post("/api/v1/auth/login", json={"id": "forecaster", "password": "wrong"})
    assert r.status_code == 401


def test_login_unknown_user(client):
    r = client.post("/api/v1/auth/login", json={"id": "nobody", "password": "x"})
    assert r.status_code == 401


def test_no_token_rejected(client):
    r = client.get("/api/v1/confidence-map", params={"lead_day": 3})
    assert r.status_code == 401


# ---------- ML routes, real VishwasService ----------

def test_confidence_map_33_subdivisions(client, duty_h):
    r = client.get("/api/v1/confidence-map", params={"lead_day": 4, "cycle": CYCLE}, headers=duty_h)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 33
    assert len(body["items"]) == 33


def test_chennai_floods_day4_orange(client, duty_h):
    """The documented demo story: TN/PY Day 4, cycle 2015-12-01, ~40% Orange."""
    r = client.get("/api/v1/bust-probability", params={"subdivision": "TN/PY", "lead_day": 4, "cycle": CYCLE}, headers=duty_h)
    assert r.status_code == 200
    body = r.json()
    assert body["level"] == "orange"
    assert abs(body["bust_probability"] - 0.40) < 0.02


def test_subdivision_code_with_slash_url_encoded(client, duty_h):
    r = client.get("/api/v1/subdivisions/TN%2FPY/explain", params={"lead_day": 4, "cycle": CYCLE}, headers=duty_h)
    assert r.status_code == 200
    assert "factors" in r.json() and "closest_analog" in r.json()


def test_subdivision_code_with_ampersand_url_encoded(client, duty_h):
    r = client.get("/api/v1/subdivisions/J%26K/explain", params={"lead_day": 3, "cycle": CYCLE}, headers=duty_h)
    assert r.status_code == 200


def test_model_trust_requires_a_param(client, duty_h):
    r = client.get("/api/v1/model-trust", headers=duty_h)
    assert r.status_code == 422


def test_model_trust_by_lead_day(client, duty_h):
    r = client.get("/api/v1/model-trust", params={"lead_day": 4, "cycle": CYCLE}, headers=duty_h)
    assert r.status_code == 200


def test_model_trust_by_subdivision(client, duty_h):
    r = client.get("/api/v1/model-trust", params={"subdivision": "TN/PY", "cycle": CYCLE}, headers=duty_h)
    assert r.status_code == 200


def test_analogs_and_action_extras(client, duty_h):
    assert client.get("/api/v1/analogs", params={"subdivision": "TN/PY", "lead_day": 4, "cycle": CYCLE}, headers=duty_h).status_code == 200
    assert client.get("/api/v1/action", params={"subdivision": "TN/PY", "lead_day": 4, "cycle": CYCLE}, headers=duty_h).status_code == 200


def test_invalid_lead_day_422(client, duty_h):
    for bad in (0, 11, -1):
        r = client.get("/api/v1/bust-probability", params={"subdivision": "TN/PY", "lead_day": bad, "cycle": CYCLE}, headers=duty_h)
        assert r.status_code == 422, bad


def test_unknown_cycle_404(client, duty_h):
    r = client.get("/api/v1/bust-probability", params={"subdivision": "TN/PY", "lead_day": 4, "cycle": "1901-01-01"}, headers=duty_h)
    assert r.status_code == 404


def test_unknown_subdivision_422(client, duty_h):
    r = client.get("/api/v1/bust-probability", params={"subdivision": "NOT-A-CODE", "lead_day": 4, "cycle": CYCLE}, headers=duty_h)
    assert r.status_code == 422


# ---------- outcomes: role boundaries ----------

def test_duty_can_post_outcome(client, duty_h):
    r = client.post("/api/v1/outcomes", json={"subdivision": "TN/PY", "lead_day": 4, "outcome": "incorrect", "cycle": CYCLE}, headers=duty_h)
    assert r.status_code == 201
    assert r.json()["status"] == "pending"


def test_admin_cannot_post_outcome(client, admin_h):
    r = client.post("/api/v1/outcomes", json={"subdivision": "TN/PY", "lead_day": 4, "outcome": "correct", "cycle": CYCLE}, headers=admin_h)
    assert r.status_code == 403


def test_observer_cannot_post_outcome(client, observer_h):
    r = client.post("/api/v1/outcomes", json={"subdivision": "TN/PY", "lead_day": 4, "outcome": "correct", "cycle": CYCLE}, headers=observer_h)
    assert r.status_code == 403


def test_duty_cannot_review(client, duty_h):
    r = client.post("/api/v1/outcomes", json={"subdivision": "KL", "lead_day": 2, "outcome": "correct", "cycle": CYCLE}, headers=duty_h)
    outcome_id = r.json()["id"]
    r = client.post(f"/api/v1/outcomes/{outcome_id}/review", json={"decision": "approve"}, headers=duty_h)
    assert r.status_code == 403


def test_senior_cannot_review_own_entry(client, senior_h):
    r = client.post("/api/v1/outcomes", json={"subdivision": "KL", "lead_day": 3, "outcome": "correct", "cycle": CYCLE}, headers=senior_h)
    outcome_id = r.json()["id"]
    r = client.post(f"/api/v1/outcomes/{outcome_id}/review", json={"decision": "approve"}, headers=senior_h)
    assert r.status_code == 403


def test_senior_can_review_others_entry_then_not_again(client, duty_h, senior_h):
    r = client.post("/api/v1/outcomes", json={"subdivision": "KL", "lead_day": 4, "outcome": "correct", "cycle": CYCLE}, headers=duty_h)
    outcome_id = r.json()["id"]
    r = client.post(f"/api/v1/outcomes/{outcome_id}/review", json={"decision": "approve"}, headers=senior_h)
    assert r.status_code == 200
    assert r.json()["status"] == "approved"
    r = client.post(f"/api/v1/outcomes/{outcome_id}/review", json={"decision": "approve"}, headers=senior_h)
    assert r.status_code == 409  # not pending any more


def test_get_outcomes_any_role(client, observer_h):
    r = client.get("/api/v1/outcomes", headers=observer_h)
    assert r.status_code == 200
    assert r.json()["count"] >= 16  # the seeded FB0 entries at minimum


# ---------- users: admin only, self-protection ----------

def test_non_admin_cannot_list_users(client, duty_h):
    assert client.get("/api/v1/users", headers=duty_h).status_code == 403


def test_admin_lists_seeded_users(client, admin_h):
    r = client.get("/api/v1/users", headers=admin_h)
    assert r.status_code == 200
    assert r.json()["count"] == 8


def test_create_user_then_duplicate_id_conflicts(client, admin_h):
    r = client.post("/api/v1/users", json={"id": "testuser1", "name": "Test User", "role": "duty"}, headers=admin_h)
    assert r.status_code == 201
    r = client.post("/api/v1/users", json={"id": "testuser1", "name": "Test User", "role": "duty"}, headers=admin_h)
    assert r.status_code == 409


def test_create_user_bad_id_format(client, admin_h):
    r = client.post("/api/v1/users", json={"id": "BAD ID", "name": "x", "role": "duty"}, headers=admin_h)
    assert r.status_code == 422


def test_admin_cannot_deactivate_self(client, admin_h):
    r = client.patch("/api/v1/users/admin", json={"active": False}, headers=admin_h)
    assert r.status_code == 403


def test_admin_cannot_change_own_role(client, admin_h):
    r = client.patch("/api/v1/users/admin", json={"role": "observer"}, headers=admin_h)
    assert r.status_code == 403


def test_admin_can_change_others(client, admin_h):
    r = client.patch("/api/v1/users/testuser1", json={"role": "observer"}, headers=admin_h)
    assert r.status_code == 200
    assert r.json()["role"] == "observer"


# ---------- settings ----------

def test_non_admin_cannot_set_settings(client, duty_h):
    r = client.put("/api/v1/settings/bust-definition", json={"error_threshold_mm": 30, "regional_percentile": 96}, headers=duty_h)
    assert r.status_code == 403


def test_admin_sets_settings(client, admin_h):
    r = client.put("/api/v1/settings/bust-definition", json={"error_threshold_mm": 30, "regional_percentile": 96}, headers=admin_h)
    assert r.status_code == 200
    assert r.json()["saved"] is True


def test_settings_out_of_range(client, admin_h):
    r = client.put("/api/v1/settings/bust-definition", json={"error_threshold_mm": 9999, "regional_percentile": 96}, headers=admin_h)
    assert r.status_code == 422
    r = client.put("/api/v1/settings/bust-definition", json={"error_threshold_mm": 30, "regional_percentile": 100}, headers=admin_h)
    assert r.status_code == 422


# ---------- audit ----------

def test_non_admin_cannot_read_audit(client, observer_h):
    assert client.get("/api/v1/audit", headers=observer_h).status_code == 403


def test_admin_reads_audit_after_actions(client, admin_h):
    r = client.get("/api/v1/audit", headers=admin_h)
    assert r.status_code == 200
    assert len(r.json()["items"]) >= 1  # earlier tests already performed admin actions


# ---------- CORS ----------

def test_cors_preflight_allows_origin(client):
    r = client.options("/api/v1/cycles", headers={"Origin": "http://localhost:5500", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" in {k.lower() for k in r.headers.keys()}


# ---------- retraining: fast checks (full run-to-completion is in verify_live.py) ----------

def test_retraining_returns_immediately_and_rejects_concurrent(client, admin_h):
    t0 = time.time()
    r = client.post("/api/v1/retraining", headers=admin_h)
    elapsed = time.time() - t0
    assert r.status_code == 202
    assert elapsed < 5, f"retraining blocked the request thread for {elapsed:.1f}s"
    job = r.json()
    assert job["status"] == "queued"
    assert job["job_id"]

    r2 = client.post("/api/v1/retraining", headers=admin_h)
    assert r2.status_code == 409

    # give it a moment, confirm status polling itself works (not waiting for completion here)
    time.sleep(1)
    r3 = client.get(f"/api/v1/retraining/{job['job_id']}", headers=admin_h)
    assert r3.status_code == 200
    assert r3.json()["status"] in ("queued", "running", "done")

    # let the background thread actually finish before the process/test DB is torn down
    for _ in range(90):
        r = client.get(f"/api/v1/retraining/{job['job_id']}", headers=admin_h)
        if r.json()["status"] in ("done", "failed"):
            break
        time.sleep(2)
    assert r.json()["status"] == "done", r.json()


def test_non_admin_cannot_start_retraining(client, duty_h):
    r = client.post("/api/v1/retraining", headers=duty_h)
    assert r.status_code == 403
