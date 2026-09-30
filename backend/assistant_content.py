"""Grounding content for the assistant.

SITE_FACTS is copied (condensed, not paraphrased into new claims) from the real
Methodology, Help/glossary and role-permission text already live in
frontend/index.html (#v-methodology, #v-help). It must be kept in sync with that
copy by hand -- it is not generated from the HTML at runtime, so a change to one
should prompt a check of the other.

live_data_block() formats real, per-request VishwasService output (the same
bust_probability()/explain() calls the dashboard drawer itself uses) into the
"why is this like this" grounding block. Nothing here is invented.
"""

SITE_FACTS = """
Vishwas is a Forecast Trust Console built for NCMRWF (National Centre for Medium Range
Weather Forecasting), Ministry of Earth Sciences, Government of India -- a prototype for
Smart India Hackathon 2026, Problem Statement 26079.

WHAT IT DOES: Takes an NCMRWF rainfall forecast that has already been issued, for Day 1 to
Day 10, and estimates, for each of 33 meteorological subdivisions and each lead day, the
probability that it will bust -- before the outcome is known. It explains the estimate,
shows the closest past case, and says how much precedent the model has.

WHAT IT DOES NOT DO: It does not forecast the weather itself, does not replace IMD or
NCMRWF guidance, and does not issue public warnings. It is a second opinion on how much to
trust a forecast that has already been issued, not a replacement for the duty forecaster's
judgement.

WHAT COUNTS AS A BUST: A forecast busts when EITHER condition holds (checked separately,
the trigger reason is recorded):
- Magnitude: the error is at least 25 mm, or at least the subdivision's 95th-percentile
  error for that lead day, whichever is larger. The percentile is fit on training years only.
- Category: the rain/no-rain call was wrong, using IMD's rainy-day line of 2.5 mm.

DATA: NCMRWF's S2S reforecast archive (about 60 km resolution), 1993 to 2015, one run per
month, Days 1 to 10 (91,080 forecast rows). IMD gridded daily rainfall (0.25 degree),
averaged per subdivision. Weather state at issue time: the forecast's own sea-level
pressure, surface pressure, 10 m winds, plus IMD rain observed the day before issue.

WARNING LEVELS (IMD colour code): Green = no warning. Yellow = watch. Orange = alert.
Red = warning. These map directly to bust-probability bands shown on the dashboard.

TWO KINDS OF CONFIDENCE (different things, never conflate them):
- Forecast confidence = 100% minus the bust probability. How far the issued forecast can
  be trusted for this subdivision and lead day. Uses the warning colours.
- Model self-confidence = how much historical precedent the model has for the current
  pattern: enough past busts for this subdivision and lead day, a familiar pattern, inputs
  inside the training range, agreement with similar past cases. Uses a separate violet
  scale, never the warning colours. Low self-confidence means "weigh the forecaster's
  judgement more," not "the forecast is bad."

BIG-MISS WATCH: Most busts are rain/no-rain flips. A separate calibrated model estimates
the risk that the forecast misses by a LARGE amount (a magnitude bust) and flags it when
that risk crosses a threshold set on held-out data. Shown on the dashboard next to the
bust probability; it never changes the main bust probability, it's an extra signal.

HOW WELL IT WORKS: Evaluated on 2013-2015 (years never seen in training, 11,880
forecasts). PR-AUC (bust-detection score) 0.541 for Vishwas vs 0.452 using the forecast
amount alone vs 0.272 using past bust rates alone. Calibration error 0.02 (a stated 40%
chance is close to being right 40% of the time). The big-miss watch raises the share of
large misses caught at Alert-or-above from 41% to 75%, for 2.5% more flagged forecasts.

KNOWN LIMITATIONS (state these plainly if asked, never smooth them over):
- Days 1 to 3: little gain over the raw forecast amount alone, because the archive has one
  run a month so the signal from how a forecast changes between runs is missing.
- Large misses: the bust probability is driven mostly by rain/no-rain flips; the big-miss
  watch helps but extreme events can still be missed (example: Tamil Nadu, 2 December 2015).
- Coastal and hilly subdivisions (Coastal Karnataka, Kerala, Konkan & Goa) are the weakest
  -- the ~60 km model resolves their terrain-driven rain poorly.
- The reforecast archive ends in 2015, so this console replays past forecast cycles rather
  than showing today's forecast.
- Weather-event tags (e.g. "cyclone", "heat wave") are rule-based labels from the
  forecast's pressure, winds and season, not a trained model.

ROLES AND PERMISSIONS (four roles, one workspace each):
- Duty forecaster (Shift desk): can view everything; can record an outcome after a day
  verifies. Cannot approve outcomes, cannot manage users or settings.
- Senior forecaster (Review desk): everything a duty forecaster can do, plus can approve
  or reject outcomes recorded by others -- never their own entries.
- Administrator (Control room): can add users, change roles, set the bust definition and
  start retraining. Cannot record or approve outcomes.
- Observer (Briefing): view-only across every page. Cannot record, approve, or manage
  anything.

GLOSSARY:
- Forecast bust: a large deviation between a forecast and what actually happened.
- Lead day: how many days ahead the forecast is for, Day 1 to Day 10.
- Subdivision: one of the 33 meteorological subdivisions Vishwas monitors.
- Weather event: the type of event most likely to cause a bust in a flagged subdivision
  (e.g. cyclone, heat wave) -- a rule-based tag, not a trained model.
- Historical analog: a past event with a similar forecast pattern, shown with how its
  forecast verified.
- Outcome: what a forecaster recorded after the fact -- correct, incorrect, or partial.
- Approved outcome: an outcome a senior forecaster has checked. Only approved outcomes are
  used to improve the model.

USING THE SITE: Sign in with a user ID; the role decides which workspace opens. Pick the
forecast cycle in the blue bar -- every page then shows that cycle (the archive's cycles
run 1993-2015; the console replays a selected historical cycle, not live weather). On the
dashboard, choose a lead day and select a subdivision on the map or from the list below it.
The card on the right explains the estimate: the main factors, the closest past case, and
the recommended action. Model Trust shows where the model itself is less sure. Text size
controls (A-/A/A+) and Dark mode are in the top bar; every subdivision on the map has a
text label with its value and warning level, so colour is never the only signal.
""".strip()


SYSTEM_PROMPT = """You are the Vishwas assistant, embedded in the Vishwas Forecast Trust \
Console (NCMRWF, Ministry of Earth Sciences, Government of India). You help forecasters, \
administrators and observers understand the product and its output.

{live_data_status}

Ground rules, never break these:
1. If LIVE DATA IS ATTACHED above, and the question is about what's on screen right now \
(e.g. "why is this like this", "what changed", "why this level") -- answer using ONLY the \
numbers and factors in that LIVE DATA block. It is real, current, and already given to you;\
 do not say you lack the selection when this block is present.
2. If NO LIVE DATA IS ATTACHED and the question needs current numbers, say plainly that you \
don't have the current selection and ask the user to pick a subdivision, lead day and cycle \
on the dashboard -- never guess or invent a number.
3. For general "what is this / how does X work" questions, answer using the SITE FACTS \
below. Do not invent metrics, thresholds, or behaviour not stated there.
4. Never state a bust probability, warning level, confidence value, or factor that is not \
explicitly present in the LIVE DATA block or the SITE FACTS block.
5. Keep answers short and spoken-friendly (2-4 sentences) -- this answer may be read aloud.
6. Reply in {language}.

SITE FACTS (background reference):
{site_facts}
"""


def live_data_block(bp: dict, ex: dict, subdivision_name: str) -> str:
    """bp: VishwasService.bust_probability() output. ex: VishwasService.explain() output.
    Both are the exact real values the dashboard drawer renders for this selection."""
    factors = ex.get("factors") or []
    factors_txt = "; ".join(
        f"{f['feature']} ({f['direction']} it by {f['contribution_percent']}%)" for f in factors
    ) or "none available"
    analog = ex.get("closest_analog")
    analog_txt = (f"{analog['date']}, {analog['description']}. Outcome: {analog['outcome']}."
                  if analog else "none recorded")
    return (
        f"LIVE DATA for {subdivision_name} ({bp['subdivision']}), lead day {bp['lead_day']}:\n"
        f"- Bust probability: {round(bp['bust_probability'] * 100)}%\n"
        f"- Warning level: {bp['level']} ({bp['band']})\n"
        f"- Forecast confidence: {round(bp['forecast_confidence'] * 100)}%\n"
        f"- Model self-confidence: {round(bp['model_self_confidence'] * 100)}%\n"
        f"- Weather event tag: {bp.get('weather_event') or 'none'}\n"
        f"- Recommended action: {bp['recommended_action']}\n"
        f"- Top contributing factors: {factors_txt}\n"
        f"- Closest historical analog: {analog_txt}"
    )
