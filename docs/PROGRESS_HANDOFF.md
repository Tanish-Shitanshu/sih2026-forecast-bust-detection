# Vishwas: progress handoff (verification pass + model improvement), 2026-09-30

**Repo:** https://github.com/Tanish-Shitanshu/sih2026-forecast-bust-detection, branch **`main`** (all work below is merged and pushed: commit `c62fd95`).
**Context:** this continues Tanish's brief ("Part A: verify PR #3 backend/frontend and fix; Part B: push on the model's weaknesses, validation-only selection, no test-year tuning"). Read `BACKEND_HANDOFF.md`, `ml/README.md`, `ml/BACKEND_INTEGRATION.md` and `backend/README.md` for the full system context.

---

## Part A: DONE (verified, fixed, merged)

Ran all tests, drove the app live (backend :8000 + frontend :5500, genuinely cross-origin) as all 4 roles.

**Test status on `main`:** `ml/` 44/44 pass, `backend/` **39/39** pass. `ml/models/` is byte-identical after the full backend suite, including a real retraining run.

### Bugs found and fixed
| # | Bug | Fix |
|---|---|---|
| 1 | **Retraining overwrote the committed `ml/models/<source>/` bundle** (one test run rewrote 11 files; the old README said "git checkout afterwards") | Retraining trains into a staging folder and swaps into `backend/runtime_models/<source>/` (gitignored, override `VISHWAS_RUNTIME_MODELS`). `service_registry` loads the runtime bundle if present, else the committed baseline. A test asserts `git status ml/models` is empty. |
| 2 | **"Retrain on approved outcomes" never used the outcomes** (it only reported a count) | The job now calls `vishwas_ml.feedback.recalibrate` with the DB's approved outcomes. The job metrics say `applied` or `skipped: not enough approved outcomes` (min 50; `partial` excluded). |
| 3 | **Roles were read from the JWT**, so a demoted user kept old powers for 12 h | The role is read from the DB on every request (+ test). Verified live: the same token went 201 → 403 after demotion, 401 after deactivation. |
| 4 | **Frontend silently showed fake `calc()` sample numbers as model output** when a cycle failed to load | A cycle loads into a fresh cache and switches only when complete; on failure it stays on the previous cycle. `calcReal()` never falls back to the sample (it throws instead). Tested live by injecting a bogus cycle. |
| 5 | Own-entry checks compared display names | They compare user ids (4 places). |
| 6 | The dashboard and footer labelled real model output "Illustrative data" | Wording corrected. |
| 7 | Windows test teardown deleted the DB while the retraining thread held it | Teardown waits for the job. |
| 8 | README claimed "no skill over climatology at Days 1–3" | Corrected: no gain over *forecast-only*; it beats climatology at every lead. |

### Verified live (all correct)
- **Negative cases:** 401 without a token. Observer/admin recording → 403. Duty approving → 403. Senior self-review → 403. Senior reviewing another's entry → 200, then 409 on a repeat. Admin self-deactivation → 403.
- **Other roles:** duty listing users → 403; observer changing settings → 403; senior retraining → 403.
- **Encoded codes and errors:** `TN%2FPY` / `J%26K` → 200. Bad subdivision, lead day, user id or setting → 422; unknown cycle → 404.
- **Retraining:** `POST` in 19 ms; the map still served in 98 ms during training; a concurrent start → 409; the job completed and the service reloaded.
- **UI per role:** the observer sees no record form or review column; a senior's own entries have Approve disabled; the admin's own Deactivate is disabled.

---

## Part B: IN PROGRESS (validation done, final test runs NOT yet done)

**Harness:** `ml/experiments/ncmrwf_improve.py`
- **Selection:** train 1993–2008, calibrate 2009, **validate 2010–2012**. Test years 2013–2015 are **dropped from the data before selection**.
- **Results:** `ml/experiments/results/select.csv` (includes paired block-bootstrap CIs vs base).
- **Run:** `ml/.venv/Scripts/python ml/experiments/ncmrwf_improve.py select [--variants a,b,...]`
- **Final test:** `... final --variants base,<chosen>` retrains on 1993–2011 / calib 2012 and evaluates **once** on 2013–2015 with CIs.
- It needs local files outside the repo: `SIH/s2s_raw/ncmrwf_fc_1993_2015.parquet` (per-lead S2S fields) and `SIH/s2s_raw/imd_subdivision_daily_1992_2016.parquet` (built on first run from `SIH/geo_data/imd`). These are on Mohit's machine.

### Validation results (base PR-AUC 0.521; "real" = 95% CI excludes 0)
| Item | Tried | Result on validation |
|---|---|---|
| **Days 1–3** | Diagnosis first (SHAP by lead) | The model is *not* blind at Days 1–3 (PR-AUC 0.537 vs 0.545 at Days 4–10). The forecast amount already carries the signal: 90% of Day 1–3 busts are rain/no-rain flips; bust rate is 34% when the forecast is within 2 mm of 2.5 mm, vs 8.5% otherwise. The run-to-run feature is dead (monthly inits). |
| | NCMRWF per-lead weather, IMD antecedent rain (1/3/7 days), interactions, separate models per lead band, static terrain/coast | **Real gains at Days 1–3:** `feat_all_band` +0.021 [+0.007, +0.036]; `static_cluster` +0.019 [+0.006, +0.033]; `all_static_cluster_gefs` **+0.028 [+0.012, +0.045]**. |
| **Magnitude busts (21% caught)** | Class weights ×3/×6/×12, positive weight ×2, combined | Recall rises (up to 41%) but overall PR-AUC and coastal **drop significantly**. Rejected. |
| | **Separate calibrated "big-miss" head** (main model untouched; threshold picked on the calibration year for a precision target) | **Works:** at precision target 0.15, +6.0% extra flags → big-miss recall **21% → 65%**; at 0.25, +3.3% flags → 55%. Main-model metrics unchanged. |
| **Coastal (CST.KA, KL, KNK/GA)** | Elevation proxy (from NCMRWF pressure) + coastal flag; coastal-vs-inland calibration; both; coastal weight ×2 | **All null** (CIs span 0). |
| | GEFS pooled as auxiliary training rows (weight 0.3) | **The only real coastal gain:** +0.016 [+0.002, +0.029]. Combining it with the other features loses it again. |
| **Data scarcity / GEFS as auxiliary** | Pool at weights 0.1/0.3/1.0; pretrain on GEFS then fine-tune | Pool 0.3 is best (+0.007 overall, +0.017 Days 1–3, +0.016 coastal). Weight 1.0 and pretraining are worse. **Caveat to state:** only GEFS 2016–2019 was used (no shared observation days with the NCMRWF test years), but that is later in time than the test years. GEFS keeps its own thresholds (never merged). |
| **Best overall** | `all_static_cluster_gefs` (antecedent + interactions + static + coastal/inland calibration + GEFS pool 0.3) | **+0.012 [+0.003, +0.022]** overall, +0.028 at Days 1–3, but coastal −0.011 (n.s.) |

### 2015 archive cutoff: rechecked, NEW LEAD FOUND
- The RDS portal is unchanged: 9 datasets, S2S `year: [1993..2015]`, init days 09/17/25 `"disabled": true` in the portal's own config.
- **NCMRWF's operational medium-range ensemble (NEPS) is archived in TIGGE, from about Aug 2017 to the present**: daily runs, a 10–15 day horizon, precipitation included, GRIB2, **ECMWF Data Store** (https://ecds.ecmwf.int/datasets/tigge-forecasts). NCMRWF is confirmed as one of TIGGE's 13 centres, and a peer-reviewed paper downloaded NCMRWF precip from TIGGE.
- **This would remove the biggest limitations:** it's the operational system rather than S2S, covers recent years, runs daily (so run-to-run change works again), and has ensemble spread (the pipeline already supports `forecast_rain_spread`).
- It needs a free ECMWF account and acceptance of the TIGGE licence. A **human** must create it; Claude may not create accounts.

---

## What to do next (in order)

1. **Final test runs for Part B** (one run, reported as is, no further tuning after seeing test):
   ```bash
   ml/.venv/Scripts/python ml/experiments/ncmrwf_improve.py final --variants base,static_cluster,gefs_pool_w03,all_static_cluster_gefs,mag_head_p15
   ```
   Report the before/after numbers on 2013–2015 with the CIs from `ml/experiments/results/final_bootstrap.json`.
2. **Promote the winner into the product only if it holds on test:**
   - Add the chosen feature groups and calibration to `vishwas_ml/features.py` / `pipeline.py` (they currently live only in the experiment file).
   - Retrain `models/ncmrwf`, keep all 44 + 39 tests green, and update `ml/README.md` metrics.
   - Recommendation: ship the **big-miss head** as an extra field or flag (e.g. `big_miss_risk`). That needs a small, additive API and frontend change; agree it with the frontend first.
3. **TIGGE NCMRWF NEPS:** a human registers at ECMWF, then build a TIGGE reader like `vishwas_ml/ncmrwf_s2s.py` (GRIB2 via cfgrib), with 00Z daily inits and 24 h accumulations from the 6-hourly steps. Reuse `subdivision_weights.build_weights` and `build_s2s_pairs.build_pairs`. It becomes a new source profile (`ncmrwf_neps`) with its own thresholds.
4. Update `ml/README.md` "Known limitations" with the Part B findings above (coastal: tried 4 things, only GEFS pooling helped; magnitude: weighting rejected, separate head works).

## Rules that still apply
- No leakage: every feature must be known at 00Z on the issue date.
- No tuning on the test years (selection only on validation).
- NCMRWF and GEFS thresholds are never merged.
- `ml/models/` is never written at runtime.
- Report real numbers only.
