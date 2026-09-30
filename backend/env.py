"""Loads backend/.env into os.environ (KEY=VALUE per line, # comments, blank lines
skipped). No third-party dependency -- one file, one line right now (SARVAM_API_KEY),
doesn't justify adding python-dotenv to requirements.txt. Real environment variables
already set take precedence and are never overwritten.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(HERE, ".env")


def load_env():
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
