"""SQLite persistence: users, outcomes, audit log, settings, retraining jobs.

Chosen per the ML handoff's own suggestion ("SQLite is plenty for the
hackathon demo") -- four small, stable tables, no concurrent-write load
beyond a single-process demo. A single connection per request, opened with
check_same_thread=False since FastAPI's threadpool (used for retraining)
needs to share it; SQLite serializes writes internally so this is safe at
this scale.
"""
import os
import sqlite3
import json
import datetime

import bcrypt

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("VISHWAS_DB_PATH", os.path.join(HERE, "vishwas.db"))
SCHEMA_PATH = os.path.join(HERE, "schema.sql")

# id, name, role -- from frontend/index.html's mkUsers(). Demo accounts (the
# first 4, matching DEMO[]) share the frontend's demo password; the rest get
# distinct generated passwords printed once at seed time (not meant to be
# guessable, but this is a hackathon demo, not a production credential store).
SEED_USERS = [
    ("forecaster", "R. Nair", "duty"),
    ("senior", "S. Iyer", "senior"),
    ("admin", "V. Rao", "admin"),
    ("observer", "Guest observer", "observer"),
    ("abhosale", "A. Bhosale", "duty"),
    ("pdutta", "P. Dutta", "duty"),
    ("nsharma", "N. Sharma", "duty"),
    ("kmenon", "K. Menon", "duty"),
]
DEMO_PASSWORD = "vishwas123"

# date, subdivision name, lead_day, predicted (0-100), outcome, submitted_by name
# -- frontend/index.html's FB0[], name->code resolved via ml/data/subdivisions.json.
SEED_OUTCOMES = [
    ("25 Sep 2026", "Coastal Andhra Pradesh", 2, 71, "incorrect", "R. Nair"),
    ("25 Sep 2026", "Konkan & Goa", 4, 58, "partial", "S. Iyer"),
    ("24 Sep 2026", "Vidarbha", 6, 22, "correct", "A. Bhosale"),
    ("24 Sep 2026", "Gangetic West Bengal", 3, 66, "incorrect", "P. Dutta"),
    ("23 Sep 2026", "Kerala", 1, 34, "correct", "N. Sharma"),
    ("23 Sep 2026", "Saurashtra & Kutch", 5, 81, "incorrect", "K. Menon"),
    ("22 Sep 2026", "Telangana", 2, 45, "partial", "R. Nair"),
    ("22 Sep 2026", "East Uttar Pradesh", 7, 19, "correct", "S. Iyer"),
    ("21 Sep 2026", "Odisha", 3, 63, "incorrect", "A. Bhosale"),
    ("21 Sep 2026", "Marathwada", 4, 52, "partial", "P. Dutta"),
    ("20 Sep 2026", "Himachal Pradesh", 8, 74, "incorrect", "N. Sharma"),
    ("20 Sep 2026", "Coastal Karnataka", 1, 15, "correct", "K. Menon"),
    ("19 Sep 2026", "Assam & Meghalaya", 6, 39, "correct", "R. Nair"),
    ("19 Sep 2026", "Rayalaseema", 9, 28, "correct", "S. Iyer"),
    ("18 Sep 2026", "Bihar", 2, 60, "partial", "A. Bhosale"),
    ("18 Sep 2026", "Sub-Himalayan West Bengal & Sikkim", 10, 88, "incorrect", "P. Dutta"),
]


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _name_to_code(subdivisions_path):
    with open(subdivisions_path) as fh:
        data = json.load(fh)
    return {s["name"]: s["code"] for s in data["subdivisions"]}


def init_db(subdivisions_path, reset=False):
    """Creates the schema and seeds demo data if the DB is empty (or reset=True)."""
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    fresh = not os.path.exists(DB_PATH)
    conn = get_conn()
    with open(SCHEMA_PATH) as fh:
        conn.executescript(fh.read())
    conn.commit()

    if fresh or conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        name_to_code = _name_to_code(subdivisions_path)
        name_to_id = {}
        for user_id, name, role in SEED_USERS:
            pw = DEMO_PASSWORD if user_id in {"forecaster", "senior", "admin", "observer"} else DEMO_PASSWORD
            conn.execute(
                "INSERT INTO users (id, name, role, password_hash, active) VALUES (?, ?, ?, ?, 1)",
                (user_id, name, role, hash_password(pw)),
            )
            name_to_id[name] = user_id
        for date, region_name, lead_day, predicted, outcome, by_name in SEED_OUTCOMES:
            code = name_to_code.get(region_name, region_name)
            conn.execute(
                "INSERT INTO outcomes (date, subdivision, subdivision_name, lead_day, "
                "predicted_bust_probability, outcome, submitted_by, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'approved')",
                (date, code, region_name, lead_day, predicted / 100.0, outcome, name_to_id.get(by_name, by_name)),
            )
        conn.execute("INSERT OR IGNORE INTO settings (id, error_threshold_mm, regional_percentile) VALUES (1, 25, 95)")
        conn.commit()
    conn.close()


def now_iso():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def audit_log(actor: str, action: str):
    conn = get_conn()
    conn.execute("INSERT INTO audit (time, actor, action) VALUES (?, ?, ?)", (now_iso(), actor, action))
    conn.commit()
    conn.close()
