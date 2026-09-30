"""Standalone check: does the backend actually load SARVAM_API_KEY from backend/.env?
Run manually (not part of the pytest suite -- this checks local machine state, not code):

    cd backend && ../ml/.venv/bin/python tests/check_sarvam_key.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env  # noqa: E402

env.load_env()
key = os.environ.get("SARVAM_API_KEY")

if key is None:
    print("SARVAM_API_KEY: NOT SET -- backend/.env missing or has no SARVAM_API_KEY line")
    sys.exit(1)
elif key == "your_key_here":
    print("SARVAM_API_KEY: present in backend/.env but still the placeholder value "
          "('your_key_here') -- replace it with a real key")
    sys.exit(1)
else:
    print(f"SARVAM_API_KEY: loaded, {len(key)} characters, starts with {key[:4]!r} "
          "-- looks like a real key was set")
    sys.exit(0)
