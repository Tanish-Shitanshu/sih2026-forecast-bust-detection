CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('duty','senior','admin','observer')),
  password_hash TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS outcomes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT NOT NULL,
  subdivision TEXT NOT NULL,
  subdivision_name TEXT,
  lead_day INTEGER NOT NULL,
  predicted_bust_probability REAL,
  outcome TEXT NOT NULL CHECK(outcome IN ('correct','incorrect','partial')),
  note TEXT,
  submitted_by TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected')),
  reviewed_by TEXT,
  cycle TEXT
);

CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  time TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  error_threshold_mm INTEGER NOT NULL DEFAULT 25,
  regional_percentile INTEGER NOT NULL DEFAULT 95
);

CREATE TABLE IF NOT EXISTS retraining_jobs (
  job_id TEXT PRIMARY KEY,
  status TEXT NOT NULL CHECK(status IN ('queued','running','done','failed')),
  source TEXT NOT NULL,
  approved_outcomes_used INTEGER,
  started_at TEXT,
  finished_at TEXT,
  metrics_json TEXT,
  error TEXT
);
