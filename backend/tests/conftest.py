import os
import shutil
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="session")
def client():
    # a throwaway DB file per test session, never the real demo DB
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    runtime = tempfile.mkdtemp(prefix="vishwas_runtime_")
    os.environ["VISHWAS_DB_PATH"] = path
    os.environ["VISHWAS_RUNTIME_MODELS"] = runtime  # retraining must never touch ml/models/
    from fastapi.testclient import TestClient
    import app as app_module
    import jobs
    with TestClient(app_module.app) as c:
        yield c
    for _ in range(600):  # let a background retraining thread finish before deleting its DB
        if not jobs.is_running():
            break
        time.sleep(0.5)
    os.remove(path)
    shutil.rmtree(runtime, ignore_errors=True)


def login(client, user_id, password="vishwas123"):
    r = client.post("/api/v1/auth/login", json={"id": user_id, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture(scope="session")
def duty_h(client):
    return login(client, "forecaster")


@pytest.fixture(scope="session")
def senior_h(client):
    return login(client, "senior")


@pytest.fixture(scope="session")
def admin_h(client):
    return login(client, "admin")


@pytest.fixture(scope="session")
def observer_h(client):
    return login(client, "observer")


CYCLE = "2015-12-01"
