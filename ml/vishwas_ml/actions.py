"""Rules engine: probability band -> warning tier -> recommended action (no ML).

Tier thresholds and action text come from the frontend's TIERS table so the map colour
and the instruction always agree with the UI.
"""


def tier_for(p, tiers):
    for t in tiers:
        if p < t["max"]:
            return t
    return tiers[-1]


def trust_level(u, levels):
    for lv in levels:
        if u < lv["max"]:
            return lv["label"]
    return levels[-1]["label"]


def as_pct(p):
    """Same rounding as the frontend: whole percent, returned as a fraction."""
    return round(float(p) * 100) / 100


def trust_note(confidence_label):
    if confidence_label in ("Reduced", "Substantially reduced"):
        return ("Model self-confidence is " + confidence_label.lower()
                + " here. Weigh the forecaster's judgement above this probability.")
    return None
