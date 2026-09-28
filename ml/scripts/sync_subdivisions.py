"""Copy the 33 subdivisions and display constants out of frontend/index.html.

The frontend is the source of truth for subdivision codes, region grouping,
warning tiers (thresholds + action text), model-trust levels and weather-event
tags. This script parses them from the JS source into ml/data/subdivisions.json
so the ML side never keeps its own list.

    python ml/scripts/sync_subdivisions.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
FRONTEND = ML.parent / "frontend" / "index.html"
OUT = ML / "data" / "subdivisions.json"


def _block(src, name):
    """Return the text of `var NAME=[ ... ];`."""
    m = re.search(r"var " + name + r"=\[(.*?)\n?\];", src, re.S)
    if not m:
        raise SystemExit(f"could not find var {name} in {FRONTEND}")
    return m.group(1)


def _objects(block):
    return re.findall(r"\{([^{}]*)\}", block)


def _field(obj, key):
    m = re.search(key + r':("(?:[^"\\]|\\.)*"|[-\d.]+)', obj)
    if not m:
        return None
    v = m.group(1)
    return json.loads(v) if v.startswith('"') else float(v) if "." in v else int(v)


def _strings(text):
    return [json.loads(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', text)]


def _set(src, name):
    m = re.search(r"var " + name + r"=setOf\(\[(.*?)\]\)", src)
    return _strings(m.group(1)) if m else []


def parse(src):
    subs = [{"id": _field(o, "id"), "code": _field(o, "code"), "name": _field(o, "name"),
             "map_col": _field(o, "col"), "map_row": _field(o, "row")} for o in _objects(_block(src, "SUBS"))]
    regions = []
    for m in re.finditer(r'\{k:"(\w+)",name:"([^"]+)",codes:\[(.*?)\]\}', _block(src, "RG")):
        regions.append({"key": m.group(1), "name": m.group(2), "codes": _strings(m.group(3))})
    region_of = {c: r for r in regions for c in r["codes"]}
    for s in subs:
        r = region_of[s["code"]]
        s["region_key"], s["region_name"] = r["key"], r["name"]
    tiers = [{"key": _field(o, "k"), "label": _field(o, "lab"), "short": _field(o, "short"),
              "max": _field(o, "max"), "band": _field(o, "band"), "confidence_band": _field(o, "band2"),
              "action": _field(o, "act")} for o in _objects(_block(src, "TIERS"))]
    trust = [{"max": _field(o, "max"), "label": _field(o, "lab")} for o in _objects(_block(src, "VIO"))]
    events = _strings(re.search(r"var EVENTS=\[(.*?)\];", src).group(1))
    return {
        "subdivisions": subs,
        "regions": regions,
        "tiers": tiers,
        "trust_levels": trust,
        "events": events,
        "event_sets": {k: _set(src, "EV_" + k) for k in ("COAST", "NW", "HEAT", "HEAVY", "NODEP")},
    }


def main():
    src = FRONTEND.read_text(encoding="utf-8")
    out = parse(src)
    n = len(out["subdivisions"])
    if n != 33 or len({s["code"] for s in out["subdivisions"]}) != 33:
        sys.exit(f"expected 33 unique subdivisions in the frontend, found {n}")
    try:
        rev = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(FRONTEND)], cwd=ML.parent,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = ""
    out = {"source": "frontend/index.html", "source_commit": rev, **out}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ML.parent)}: {n} subdivisions, {len(out['regions'])} regions, "
          f"{len(out['tiers'])} tiers (frontend commit {rev or 'unknown'})")


if __name__ == "__main__":
    main()
