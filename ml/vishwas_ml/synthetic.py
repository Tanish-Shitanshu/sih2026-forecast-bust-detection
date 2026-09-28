"""Synthetic pairs table generator (see scripts/make_synthetic.py for the description)."""
import numpy as np
import pandas as pd

from .config import load_config, load_meta
from .labels import fit_thresholds, label_frame
from .schema import COLUMNS, coerce
from .splits import year_split

# Synthetic climate assumptions per subdivision (used only to generate fake data).
SW_AMP = {"KNK/GA": 26, "CST.KA": 26, "KL": 20, "ASM": 16, "NE.HL": 15, "AR": 17, "SIK/NWB": 18, "MDH.MH": 7,
          "W.RJ": 3.5, "SAU/KCH": 4, "J&K": 2.5, "PB": 5, "HR/DL": 5, "TN/PY": 2.5, "RYL": 3.5, "CST.AP": 5,
          "SI.KA": 5, "GJ": 9, "OD": 11, "GWB": 11, "CG": 11, "E.MP": 11, "JH": 10}
NE_AMP = {"TN/PY": 10, "CST.AP": 5, "RYL": 4, "KL": 6, "SI.KA": 4, "CST.KA": 2, "TG": 1.5, "NI.KA": 1.5}
WD_AMP = {"J&K": 4.5, "HP": 4, "PB": 1.5, "HR/DL": 1.0, "W.UP": 0.8, "E.UP": 0.5, "W.RJ": 0.4, "E.RJ": 0.4}
PRE_AMP = {"ASM": 7, "NE.HL": 6, "AR": 7, "SIK/NWB": 6, "GWB": 2.5, "OD": 1.5, "KL": 3, "SI.KA": 1.5, "CST.KA": 1}
ELEV = {"J&K": 2800, "HP": 1800, "AR": 1200, "SIK/NWB": 700, "NE.HL": 600, "ASM": 150, "W.MP": 450, "E.MP": 450,
        "CG": 350, "JH": 400, "SI.KA": 800, "NI.KA": 550, "TG": 450, "RYL": 350, "MWD": 450, "MDH.MH": 550,
        "KNK/GA": 60, "CST.KA": 60, "KL": 80, "TN/PY": 120, "CST.AP": 60, "GWB": 20, "OD": 80, "SAU/KCH": 80}
SOUTH = {"TN/PY", "CST.AP", "RYL", "KL", "SI.KA", "CST.KA", "NI.KA", "TG"}


def bump(doy, center, width):
    d = (doy - center + 182.5) % 365 - 182.5
    return np.exp(-((d / width) ** 2))


def ar1(rng, shape, phi):
    eps = rng.normal(size=shape)
    x = np.empty(shape)
    x[0] = eps[0]
    k = np.sqrt(1 - phi**2)
    for t in range(1, shape[0]):
        x[t] = phi * x[t - 1] + k * eps[t]
    return x


def shocks(rng, T, members, prob, dur, mag, share=0.7):
    """Multi-day disturbance events: prob[t] = daily chance of one starting in this group."""
    out = np.zeros((T, len(members)))
    for t in np.where(rng.random(T) < prob)[0]:
        hit = rng.random(len(members)) < share
        m = rng.uniform(*mag)
        d = rng.integers(dur[0], dur[1] + 1)
        ramp = np.sin(np.linspace(0.3, np.pi - 0.3, d))
        seg = slice(t, min(T, t + d))
        out[seg, :] += np.outer(ramp[: seg.stop - t], hit * m)
    return out


def generate(seed=0, start="2020-01-01", end="2023-12-31", every=3):
    rng = np.random.default_rng(seed)
    meta = load_meta()
    subs = meta["subdivisions"]
    codes = [s["code"] for s in subs]
    S = len(codes)
    rkeys = [r["key"] for r in meta["regions"]]
    reg = np.array([rkeys.index(s["region_key"]) for s in subs])
    row = np.array([s["map_row"] for s in subs], float)
    ev = meta["event_sets"]
    coast = np.array([c in ev["COAST"] for c in codes])
    nw = np.array([c in ev["NW"] for c in codes])
    south = np.array([c in SOUTH for c in codes])

    spin = 30
    days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=spin), pd.Timestamp(end) + pd.Timedelta(days=12))
    T = len(days)
    doy = days.dayofyear.to_numpy()[:, None].astype(float)

    # ---- seasonal climatology (mm/day)
    sw_c = np.where(nw, 210, np.where(np.isin(codes, ["KL", "KNK/GA", "CST.KA"]), 195, 203))
    sw_w = np.where(nw, 36, 46)
    m_sw = bump(doy, sw_c, sw_w)
    m_ne = bump(doy, 315, 28)
    m_wd = bump(doy, 30, 45)
    m_pre = bump(doy, 120, 30)
    amp = lambda d, default: np.array([d.get(c, default) for c in codes], float)  # noqa: E731
    sw_a, ne_a, wd_a, pre_a = amp(SW_AMP, 9), amp(NE_AMP, 0), amp(WD_AMP, 0), amp(PRE_AMP, 0.3)
    mu = sw_a * m_sw + ne_a * m_ne + wd_a * m_wd + pre_a * m_pre + 0.3

    # ---- disturbance index: regional + local persistence, plus depressions and cyclones
    z = (0.75 * ar1(rng, (T, 4), 0.9)[:, reg] + 0.55 * ar1(rng, (T, S), 0.7)) / np.hypot(0.75, 0.55)
    sh = np.zeros((T, S))
    season_act = (m_sw.mean(1) + 0.6 * m_ne[:, 0] + 0.4 * m_wd[:, 0])
    for r in range(4):
        idx = np.where(reg == r)[0]
        sh[:, idx] += shocks(rng, T, idx, 0.012 * season_act + 0.002, (3, 5), (1.8, 3.0))
        cidx = idx[coast[idx]]
        if len(cidx):
            cyc_season = bump(days.dayofyear.to_numpy(), 140, 20) + bump(days.dayofyear.to_numpy(), 310, 25)
            sh[:, cidx] += shocks(rng, T, cidx, 0.006 * cyc_season, (2, 4), (3.0, 4.2), share=0.8)
    zt = z + sh
    heat = (0.7 * ar1(rng, (T, 4), 0.93)[:, reg] + 0.7 * ar1(rng, (T, S), 0.8)) / np.hypot(0.7, 0.7)
    # Convective instability: moist boundary layer that NWP handles badly (shows up in dewpoint).
    conv = ar1(rng, (T, S), 0.8)

    # ---- observed rain (IMD-like)
    p_wet = 1 / (1 + np.exp(-(-2.2 + 1.5 * np.log1p(mu) + 1.1 * zt)))
    wet = rng.random((T, S)) < p_wet
    amt = mu * np.exp(0.5 * zt) / np.maximum(p_wet, 0.25) * rng.gamma(0.8, 1 / 0.8, (T, S))
    trace = rng.uniform(0, 2.4, (T, S)) * (rng.random((T, S)) < 0.15)
    obs = np.round(np.minimum(np.where(wet, amt, trace), 450), 1)

    # ---- ERA5-like state
    elev = amp(ELEV, 300)
    moist_season = np.minimum(1, (sw_a * m_sw + ne_a * m_ne + pre_a * m_pre) / 8)
    mslp = (101300 - 500 * bump(doy, 190, 60) * (1 + 0.6 * nw) + 250 * bump(doy, 15, 50)
            - 320 * zt + rng.normal(0, 110, (T, S)))
    sp = mslp * np.exp(-elev / 8400) + rng.normal(0, 40, (T, S))
    t_mean = 27.5 - 0.6 * (9 - row)
    t_amp = 3 + 1.1 * (9 - row)
    t2m = (t_mean + t_amp * np.sin(2 * np.pi * (doy - 49) / 365) - 3.5 * m_sw * (sw_a > 4) - 0.9 * zt
           + 2.2 * heat - 0.0065 * elev + rng.normal(0, 0.7, (T, S)))
    dpd = np.clip(14 - 10 * moist_season - 2.5 * zt - 1.6 * conv + 1.8 * np.maximum(heat, 0) - 3 * coast
                  + rng.normal(0, 1.4, (T, S)), 0.3, 32)
    u10 = (4.5 * m_sw - 3.0 * m_ne * south + 1.5 * m_wd * nw + 1.3 * zt * np.where(coast, 1, 0.5)
           + rng.normal(0, 1.3, (T, S)))
    v10 = 3.0 * m_sw - 2.5 * m_ne * south - 1.0 * m_wd + 0.9 * zt + rng.normal(0, 1.3, (T, S))
    tp = np.where(obs > 0, obs * rng.lognormal(0, 0.4, (T, S)), rng.uniform(0, 0.5, (T, S)) * (rng.random((T, S)) < 0.2))
    era5 = {"mslp": mslp, "surface_pressure": sp, "dewpoint_2m": t2m - dpd + 273.15, "temp_2m": t2m + 273.15,
            "wind_u10": u10, "wind_v10": v10, "total_precipitation": tp / 1000.0}

    # ---- forecasts: issue every `every` days, leads 1..10
    issue_t = np.arange(spin, spin + (pd.Timestamp(end) - pd.Timestamp(start)).days + 1, every)
    leads = np.arange(1, 11)
    shared = rng.normal(size=(T, S))  # systematic error tied to the valid day (runs agree on it)
    I, L, Sx = np.meshgrid(issue_t, leads, np.arange(S), indexing="ij")
    I, L, Sx = I.ravel(), L.ravel(), Sx.ravel()
    V = I + L
    zi = zt[I, Sx]
    disturbed = np.maximum(zi, 0)
    sigma = 0.30 + 0.06 * L + 0.22 * disturbed + 0.35 * (sh[I, Sx] > 0.5)
    eps = sigma * (0.6 * shared[V, Sx] + 0.8 * rng.normal(size=V.size))
    q_disp = np.clip(0.02 + 0.018 * L + 0.05 * disturbed, 0, 0.45)
    shift = np.where(rng.random(V.size) < q_disp, rng.choice([-2, -1, 1, 2], V.size), 0)
    src = obs[np.clip(V + shift, 0, T - 1), Sx]
    w = np.minimum(0.08 * L, 0.7)
    base = (1 - w) * src + w * mu[V, Sx]
    fc = base * np.exp(np.clip(eps, -3, 2.2))
    fc = np.where(fc > 60, 60 + (fc - 60) * 0.6, fc)
    fc = fc + rng.uniform(0, 1.8, V.size) * (rng.random(V.size) < 0.2) * (mu[V, Sx] > 0.5)
    # Convective misses and false alarms, more likely when the issue-date air is unstable.
    ci = np.maximum(conv[I, Sx], 0) * moist_season[V, Sx]
    miss = rng.random(V.size) < np.clip(0.02 + 0.22 * ci + 0.06 * disturbed, 0, 0.6)
    fc = np.where(miss, fc * rng.uniform(0, 0.25, V.size), fc)
    spur = rng.random(V.size) < np.clip(0.12 * ci, 0, 0.4)
    fc = np.where(spur & (fc < 2.5), fc + rng.uniform(2, 10, V.size), fc)
    fc = np.round(np.maximum(fc, 0), 1)

    df = pd.DataFrame({
        "date": days[I], "subdivision_code": np.asarray(codes)[Sx], "lead_day": L,
        "forecast_rain": fc, "observed_rain": obs[V, Sx],
    })
    df["error"] = np.round(df["forecast_rain"] - df["observed_rain"], 1)
    for k, a in era5.items():
        df[k] = a[I, Sx]
    df = df.round({"mslp": 1, "surface_pressure": 1, "dewpoint_2m": 2, "temp_2m": 2, "wind_u10": 2,
                   "wind_v10": 2, "total_precipitation": 6})
    df["is_bust"], df["trigger_reason"] = False, None
    df = coerce(df)

    # Labels exactly as the data team will produce them: thresholds on training years only.
    cfg = load_config()
    split = year_split(df, cfg["split"])
    thr = fit_thresholds(df[split["train"]], cfg["bust"]["percentile"], cfg["bust"]["min_error_mm"])
    lab = label_frame(df, thr, cfg["bust"])
    df["is_bust"], df["trigger_reason"] = lab["is_bust"], lab["trigger_reason"]
    return df[COLUMNS]
