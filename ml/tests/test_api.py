"""The dev API serves the frontend paths with the service's shapes and proper HTTP errors."""
import urllib.parse

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(service):
    import serve

    serve.svc.cache_clear()
    serve.svc = lambda: service  # use the small test bundle
    return TestClient(serve.app)


def test_endpoints_return_frontend_shapes(client, service):
    c = f"{service.cycles()[-3]:%Y-%m-%d}"
    r = client.get("/api/v1/confidence-map", params={"lead_day": 3, "cycle": c})
    assert r.status_code == 200 and r.json()["count"] == 33
    assert client.get("/api/v1/bust-probability", params={"subdivision": "KL", "lead_day": 2}).status_code == 200
    code = urllib.parse.quote("TN/PY", safe="")
    r = client.get(f"/api/v1/subdivisions/{code}/explain", params={"lead_day": 4})
    assert r.status_code == 200 and r.json()["subdivision"] == "TN/PY" and len(r.json()["factors"]) == 3
    assert len(client.get("/api/v1/model-trust", params={"lead_day": 1}).json()["items"]) == 33
    assert client.get("/api/v1/analogs", params={"subdivision": "ASM", "lead_day": 3}).json()["count"] == 5
    assert "recommended_action" in client.get("/api/v1/action", params={"subdivision": "OD", "lead_day": 1}).json()
    assert client.get("/health").json()["status"] == "ok"


def test_errors_and_roles(client):
    assert client.get("/api/v1/bust-probability", params={"subdivision": "XX", "lead_day": 2}).status_code == 422
    assert client.get("/api/v1/confidence-map", params={"lead_day": 11}).status_code == 422
    assert client.get("/api/v1/confidence-map", params={"lead_day": 2, "cycle": "1999-01-01"}).status_code == 404
    body = {"subdivision": "W.UP", "lead_day": 3, "outcome": "incorrect", "note": "late spell"}
    assert client.post("/api/v1/outcomes", json=body, headers={"X-Role": "observer"}).status_code == 403
    r = client.post("/api/v1/outcomes", json=body, headers={"X-Role": "duty", "X-User": "R. Nair"})
    assert r.status_code == 201 and r.json()["status"] == "pending" and r.json()["submitted_by"] == "R. Nair"
    assert client.get("/api/v1/outcomes", params={"status": "pending"}).json()["count"] >= 1
