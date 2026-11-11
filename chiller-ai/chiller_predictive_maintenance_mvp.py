"""
UAE Centrifugal Chiller Predictive Maintenance & FDD  -  app.py
=================================================================
Single-file Streamlit application.

Install:   pip install streamlit pandas numpy plotly scikit-learn openpyxl xlrd
Run:       streamlit run app.py

(`xlrd` is required only to read legacy .xls files such as the ASHRAE RP-1043
 workbooks; `openpyxl` reads .xlsx.)

Pipeline
--------
raw BMS / RP-1043 table
  -> normalize_bms_columns()   header aliasing, degF->degC, psig->kPa(abs), voltage detect, time axis
  -> physics_engine()          R-134a saturation, enthalpy-based flow-meter-free tonnage, kW/TR, G14 baseline
  -> classify_faults()         load-aware healthy reference + dual-loop thermodynamic classifier
  -> financials()              kWh / AED waste with DEWA tariff

Engineering notes (read before trusting numbers)
------------------------------------------------
* Approach temperatures depend strongly on load and entering-water temperature.
  A single constant offset taken from a "first N %" window would raise false alarms
  whenever load changes, so the healthy reference is a small regression
  (approach / kW-per-TR as a function of leaving-chilled-water temp, entering
  condenser-water temp and tonnage) fitted on the healthy window (or on a user
  supplied healthy log).  Rows outside the training domain are NOT classified.
* Waste energy is integrated over time:  kWh = sum(max(dKW,0) * dt_hours).
  (Summing kW x tariff without dt would give AED/hour, not AED.)
* The enthalpy-based tonnage needs an effective compressor/motor efficiency
  `eta` (default 0.69, calibrated on the RP-1043 lab chiller).  Calibrate it on
  your own plant from a commissioning point before relying on absolute kW/TR.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SL0 = "SL0_Healthy"
CF = "CF_Fouling"
UC = "UC_Leak"
MIXED = "Mixed_Fault"
STANDBY = "Standby"
TRANSIENT = "Transient"
OOD = "Out_of_Domain"
TRANSITIONAL = "Transitional"  # ground-truth only (simulator ramp region)

CLASS_LABELS = [SL0, CF, UC]
DISPLAY_NAME = {
    SL0: "Normal / Healthy Baseline (SL0)",
    CF: "Condenser Tube Fouling (Type A)",
    UC: "Refrigerant Leak / Undercharge (Type B)",
    MIXED: "Mixed Fault (Type A + B signatures)",
    "INSUFFICIENT": "Insufficient steady-state data",
    "INDETERMINATE": "Indeterminate (too little data inside the healthy-reference envelope)",
}

PSI_TO_KPA = 6.89476
ATM_PSI = 14.696
KW_PER_TR = 3.517

# R-134a vapour pressure, Antoine form  log10(P[kPa]) = A - B/(T[degC] + C)
# Least-squares fit to standard saturation tables (-30..70 degC), max error 0.23 % in P.
ANT_A, ANT_B, ANT_C = 6.34273926, 953.72750615, 246.0

# Cubic fits of R-134a saturated vapour / liquid enthalpy, kJ/kg (IIR reference), valid -10..60 degC.
HG_POLY = np.array([-2.04545455e-05, -7.09956710e-04, 5.81937229e-01, 3.98580519e02])
HF_POLY = np.array([1.36363636e-05, 1.06060606e-03, 1.34474026e00, 2.00036797e02])
CP_VAP_SUCTION = 0.85  # kJ/kg.K  superheated vapour at evaporator pressure
CP_VAP_DISCHARGE = 1.05  # kJ/kg.K  superheated vapour at condenser pressure

REQUIRED_MIN = ["t_cond_out", "t_evap_out", "p_cond", "p_evap"]

ALIASES: Dict[str, List[str]] = {
    "time": ["time", "time (minutes)", "time_min", "timestamp", "datetime", "date_time", "date", "ts", "elapsed", "time_s"],
    "t_cond_in": ["tci", "t_cond_in", "cond_in", "cond_ewt", "cwst", "cws", "cwt_in", "cond_enter", "condenser_entering_temp", "cond_water_in"],
    "t_cond_out": ["tco", "t_cond_out", "cond_out", "cond_lwt", "cwrt", "cwr", "cwt_out", "cond_leave", "condenser_leaving_temp", "cond_water_out"],
    "t_evap_out": ["teo", "t_evap_out", "evap_out", "chws", "chwst", "chw_supply", "evap_lwt", "chilled_water_supply", "chwst_c"],
    "t_evap_in": ["tei", "t_evap_in", "evap_in", "chwr", "chwrt", "chw_return", "evap_ewt", "chilled_water_return"],
    "p_cond": ["prc", "p_cond", "cond_pressure", "cond_press", "pcond", "discharge_pressure", "head_pressure"],
    "p_evap": ["pre", "p_evap", "evap_pressure", "evap_press", "pevap", "suction_pressure"],
    "t_dis": ["tr_dis", "t_dis", "discharge_temp", "disch_temp", "tdis", "compressor_discharge_temp"],
    "t_suc": ["t_suc", "suction_temp", "tsuc", "compressor_suction_temp"],
    "amps": ["amps", "current", "motor_current", "amp", "compressor_amps", "rla_amps"],
    "kw": ["kw", "power", "kw_comp", "compressor_kw", "input_kw", "power_kw", "chiller_kw"],
    "tons_meas": ["evap tons", "tons", "tr", "capacity_tons", "cooling_load_tr"],
    "t_amb": ["tamb", "t_amb", "oat", "outdoor_air_temp", "ambient", "ambient_temp", "dry_bulb", "tdb", "oa_temp"],
    "subcool_meas": ["trc_sub", "subcooling"],
    "superheat_meas": ["tsh_suc", "superheat"],
}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")



@dataclass
class Config:
    tariff: float = 0.44  # AED / kWh (DEWA default)
    eta: float = 0.69  # effective compressor+motor efficiency for enthalpy tonnage
    median_win: int = 5  # rolling-median samples (transient noise filter)
    thr_mode: str = "auto"  # "auto" (k-sigma of healthy noise) | "spec" | "custom"
    thr_cond: float = 1.5  # degC  condenser approach residual trigger (spec/custom modes)
    thr_evap: float = 1.5  # degC  evaporator approach residual trigger (spec/custom modes)
    thr_kwtr_pct: float = 10.0  # %     specific-power residual trigger (spec/custom modes)
    auto_k: float = 3.0  # sigma multiple used in auto mode
    auto_floor: float = 0.25  # K, minimum trigger in auto mode
    persist_s: float = 1800.0  # s, persistence window (median of residuals) before classification
    ref_mode: str = "auto"  # "auto" | "builtin" | "window"
    ref_pct: float = 12.0  # %     healthy reference window (first N % of records)
    capacity_source: str = "enthalpy"  # "enthalpy" | "measured"
    steady_gate: str = "auto"  # "auto" | "on" | "off"
    min_fault_share: float = 25.0  # % of classified rows needed to declare a fault dataset-wide
    assumed_dt_s: float = 60.0  # s, used only if the log has no usable time column
    temp_unit: str = "auto"  # "auto" | "degF" | "degC"
    pressure_unit: str = "auto"  # "auto" | "psig" | "kPa_abs" | "kPa_g" | "bar_abs"


# ----------------------------------------------------------------------------
# 1. Ingestion & unit normalisation
# ----------------------------------------------------------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_bms_columns(raw: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, dict]:
    """Map vendor headers to canonical names and convert to SI (degC, kPa abs).

    Returns (normalised_df, meta).  Canonical output columns (when present):
    t_cond_in, t_cond_out, t_evap_in, t_evap_out, t_dis, t_suc, t_amb [degC],
    p_cond, p_evap [kPa abs], kw, amps, tons_meas, subcool_k, superheat_k,
    t_s (elapsed seconds), x (plot axis).
    """
    meta: dict = {"warnings": [], "notes": [], "mapping": {}}
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    # Alias priority (not file column order) decides which header wins.
    col_by_key: Dict[str, str] = {}
    for col in raw.columns:
        col_by_key.setdefault(_key(col), col)
    mapping: Dict[str, str] = {}
    for canon, names in ALIASES.items():
        for nm in names:
            if _key(nm) in col_by_key:
                mapping[canon] = col_by_key[_key(nm)]
                break
    meta["mapping"] = dict(mapping)

    missing = [c for c in REQUIRED_MIN if c not in mapping]
    if missing:
        raise ValueError(
            "Missing required signals: "
            + ", ".join(missing)
            + ". Needed at minimum: condenser leaving water (TCO), evaporator leaving water (TEO), "
            "condenser pressure (PRC), evaporator pressure (PRE), plus kW (or Amps). "
            f"Detected headers: {list(raw.columns)[:40]}"
        )
    if "kw" not in mapping and "amps" not in mapping:
        raise ValueError("Neither a kW column nor an Amps column was found; electrical power is required.")

    out = pd.DataFrame(index=raw.index)
    for canon, col in mapping.items():
        if canon == "time":
            continue
        out[canon] = _to_num(raw[col])

    # ---- temperature unit -------------------------------------------------
    unit = cfg.temp_unit
    if unit == "auto":
        if "t_evap_out" in out and out["t_evap_out"].notna().any():
            unit = "degF" if out["t_evap_out"].median() > 25 else "degC"
        elif "t_cond_in" in out and out["t_cond_in"].notna().any():
            unit = "degF" if out["t_cond_in"].median() > 45 else "degC"
        else:
            unit = "degC"
    meta["temp_unit"] = unit
    temp_cols = [c for c in ["t_cond_in", "t_cond_out", "t_evap_in", "t_evap_out", "t_dis", "t_suc", "t_amb"] if c in out]
    if unit == "degF":
        for c in temp_cols:
            out[c] = (out[c] - 32.0) * 5.0 / 9.0
    # Differences (sub-cooling / superheat) convert with the 5/9 scale only.
    if "subcool_meas" in out:
        out["subcool_k"] = out["subcool_meas"] * (5.0 / 9.0 if unit == "degF" else 1.0)
    if "superheat_meas" in out:
        out["superheat_k"] = out["superheat_meas"] * (5.0 / 9.0 if unit == "degF" else 1.0)
    out = out.drop(columns=[c for c in ["subcool_meas", "superheat_meas"] if c in out])

    # ---- pressure unit ----------------------------------------------------
    punit = cfg.pressure_unit
    if punit == "auto":
        med = out["p_cond"].median()
        punit = "kPa_abs" if med > 400 else "psig"
        if punit == "kPa_abs":
            meta["warnings"].append("Pressure looked like kPa; assumed ABSOLUTE. Override in the sidebar if it is gauge.")
    meta["pressure_unit"] = punit
    for c in ["p_cond", "p_evap"]:
        if punit == "psig":
            out[c] = (out[c] + ATM_PSI) * PSI_TO_KPA  # P_kPa_abs = (P_psig + 14.696) * 6.89476
        elif punit == "kPa_g":
            out[c] = out[c] + 101.325
        elif punit == "bar_abs":
            out[c] = out[c] * 100.0

    # ---- electrical: voltage auto-detect (spec rule) -----------------------
    amps_max = float(out["amps"].max()) if "amps" in out and out["amps"].notna().any() else np.nan
    if np.isnan(amps_max):
        voltage = 460.0
    else:
        voltage = 460.0 if amps_max < 150.0 else 3300.0
    meta["voltage"] = voltage
    meta["amps_max"] = amps_max
    if "kw" in out and "amps" in out:
        ok = out["kw"] > 0.2 * out["kw"].max()
        if ok.any():
            v_impl = float((out.loc[ok, "kw"] * 1000.0 / (np.sqrt(3) * out.loc[ok, "amps"].replace(0, np.nan) * 0.88)).median())
            meta["voltage_implied"] = v_impl
            if np.isfinite(v_impl) and abs(np.log(v_impl / voltage)) > 0.5:
                meta.setdefault("notes", []).append(
                    f"Amps rule selected {voltage:.0f} V but kW/Amps imply ~{v_impl:.0f} V "
                    "(voltage is only used when kW is missing)."
                )
    if "kw" not in out:
        out["kw"] = np.sqrt(3) * voltage * out["amps"] * 0.88 / 1000.0
        meta["warnings"].append("kW column missing: power estimated from Amps x voltage x PF 0.88 (approximate).")
        meta["kw_from_amps"] = True

    # ---- time axis ----------------------------------------------------------
    t_s, x, tinfo = _build_time_axis(raw, mapping.get("time"), len(raw), cfg)
    out["t_s"] = t_s
    out["x"] = x
    meta["time_info"] = tinfo

    out = out.assign(_row=np.arange(len(out)))
    if not out["t_s"].is_monotonic_increasing:
        out = out.sort_values("t_s", kind="stable")
    out = out.reset_index(drop=True)
    return out, meta


def _build_time_axis(raw: pd.DataFrame, tcol: Optional[str], n: int, cfg: Config):
    info = {"source": "row_count", "assumed": True}
    if tcol is not None:
        s = raw[tcol]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s, errors="coerce")
            t = (dt - dt.min()).dt.total_seconds()
            if dt.notna().mean() > 0.9 and t.max() > 0:
                return t.fillna(0).values.astype(float), dt.values, {"source": f"datetime '{tcol}'", "assumed": False}
        num = _to_num(s) if not pd.api.types.is_datetime64_any_dtype(s) else pd.Series(np.nan, index=s.index)
        if num.notna().mean() > 0.9:
            scale = 60.0 if "minute" in str(tcol).lower() else 1.0
            t = num.astype(float) * scale
            t = t - t.min()
            if t.max() > 0:
                info = {"source": f"numeric '{tcol}' ({'minutes' if scale == 60 else 'elapsed seconds'})", "assumed": False}
                return t.values, (t / 3600.0).values, info
        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().mean() > 0.9:
            t = (dt - dt.min()).dt.total_seconds()
            if t.max() > 0:
                info = {"source": f"datetime '{tcol}'", "assumed": False}
                return t.values.astype(float), dt.values, info
    t = np.arange(n, dtype=float) * cfg.assumed_dt_s
    info = {"source": f"row count x {cfg.assumed_dt_s:.0f}s (no usable time column)", "assumed": True}
    return t, t / 3600.0, info


# ----------------------------------------------------------------------------
# 2. First-principles physics engine
# ----------------------------------------------------------------------------
def psat_r134a_kpa(t_c):
    t = np.asarray(t_c, dtype=float)
    return 10.0 ** (ANT_A - ANT_B / (t + ANT_C))


def tsat_r134a_c(p_kpa_abs):
    p = np.clip(np.asarray(p_kpa_abs, dtype=float), 20.0, 4000.0)
    return ANT_B / (ANT_A - np.log10(p)) - ANT_C


def hg_r134a(t_c):
    return np.polyval(HG_POLY, np.clip(np.asarray(t_c, dtype=float), -15.0, 70.0))


def hf_r134a(t_c):
    return np.polyval(HF_POLY, np.clip(np.asarray(t_c, dtype=float), -15.0, 70.0))


def _rolling_median(s: pd.Series, win: int) -> pd.Series:
    win = max(int(win), 1)
    return s.rolling(win, center=True, min_periods=1).median()


def _median_dt_s(t_s: np.ndarray) -> float:
    d = np.diff(np.asarray(t_s, dtype=float))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 60.0


def physics_engine(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add saturation temperatures, approaches, enthalpy-based mass flow / tonnage and kW/TR."""
    d = df.copy()
    d["tsat_cond"] = tsat_r134a_c(d["p_cond"])
    d["tsat_evap"] = tsat_r134a_c(d["p_evap"])
    d["cond_approach"] = d["tsat_cond"] - d["t_cond_out"]
    d["evap_approach"] = d["t_evap_out"] - d["tsat_evap"]

    # suction state
    if "t_suc" in d and d["t_suc"].notna().any():
        t_suc = d["t_suc"].fillna(d["tsat_evap"] + 5.0)
    elif "superheat_k" in d:
        t_suc = d["tsat_evap"] + d["superheat_k"].fillna(5.0)
    else:
        t_suc = d["tsat_evap"] + 5.0
    t_suc = np.maximum(t_suc, d["tsat_evap"])
    if ("t_suc" in d and d["t_suc"].notna().any()):
        d["superheat_k"] = t_suc - d["tsat_evap"]
    elif "superheat_k" not in d:
        d["superheat_k"] = np.nan
    if "subcool_k" not in d:
        d["subcool_k"] = np.nan
    h_suc = hg_r134a(d["tsat_evap"]) + CP_VAP_SUCTION * (t_suc - d["tsat_evap"])

    # liquid state leaving condenser
    sub = d["subcool_k"].fillna(3.0) if "subcool_k" in d else pd.Series(3.0, index=d.index)
    h_liq = hf_r134a(d["tsat_cond"] - sub)

    # actual enthalpy rise across compressor
    if "t_dis" in d and d["t_dis"].notna().any():
        t_dis = d["t_dis"].fillna(d["tsat_cond"] + 8.0)
        h_dis = hg_r134a(d["tsat_cond"]) + CP_VAP_DISCHARGE * np.maximum(t_dis - d["tsat_cond"], 0.0)
        d["dh_source"] = "measured discharge temperature"
    else:
        # isentropic fallback: dh = cp*T_suc*((Pc/Pe)^((k-1)/k)-1) / eta_is
        k = 1.12
        pr = np.maximum(d["p_cond"] / d["p_evap"], 1.05)
        dh_s = CP_VAP_SUCTION * (t_suc + 273.15) * (pr ** ((k - 1.0) / k) - 1.0)
        h_dis = h_suc + dh_s / 0.75
        d["dh_source"] = "isentropic fallback (eta_is 0.75)"
    d["h_suc"] = h_suc
    d["h_dis"] = h_dis
    d["h_liq"] = h_liq
    d["dh_actual"] = (h_dis - h_suc).clip(lower=5.0)

    d["m_ref"] = cfg.eta * d["kw"] / d["dh_actual"]  # kg/s
    d["tr_est"] = d["m_ref"] * (h_suc - h_liq) / KW_PER_TR
    d["tr_est"] = d["tr_est"].where(d["tr_est"] > 0)

    if "tons_meas" in d and d["tons_meas"].notna().any():
        d["tr_meas"] = d["tons_meas"]
    if cfg.capacity_source == "measured" and "tr_meas" in d:
        d["tr"] = d["tr_meas"].where(d["tr_meas"] > 0)
    else:
        d["tr"] = d["tr_est"]

    d["kw_per_tr"] = d["kw"] / d["tr"].where(d["tr"] > 1.0)

    # ASHRAE Guideline 14 style weather-normalised condenser approach baseline
    if "t_amb" in d and d["t_amb"].notna().any():
        t_ref = d["t_amb"]
    elif "t_cond_in" in d:
        t_ref = d["t_cond_in"]
    else:
        t_ref = pd.Series(25.0, index=d.index)
    d["cond_approach_g14"] = 2.0 + 0.03 * (t_ref - 25.0)

    # condenser-side entering temperature used as regression feature
    if "t_cond_in" in d and d["t_cond_in"].notna().any():
        d["t_cw"] = d["t_cond_in"]
    else:
        d["t_cw"] = d["t_cond_out"]

    # running mask
    kw_p95 = np.nanpercentile(d["kw"], 95) if d["kw"].notna().any() else 0.0
    tr_p95 = np.nanpercentile(d["tr"].dropna(), 95) if d["tr"].notna().any() else 0.0
    d["running"] = (d["kw"] >= 0.15 * kw_p95) & (d["tr"] >= 0.10 * tr_p95) & d["tr"].notna()
    return d


# ----------------------------------------------------------------------------
# 3. Reference model & fault classifier
# ----------------------------------------------------------------------------
FEATS = ["t_evap_out", "t_cw", "tr"]
CORE_TARGETS = ["cond_approach", "evap_approach", "kw_per_tr"]
EXTRA_TARGETS = ["tsat_cond", "subcool_k", "superheat_k"]
SMOOTH_COLS = ["cond_approach", "evap_approach", "kw_per_tr", "t_evap_out", "t_cw", "tr", "kw", "tsat_cond", "subcool_k", "superheat_k"]
SIG_FEATS = ["dC", "dE", "kwtr_resid_pct", "dTc", "dSub", "dSh"]
RESID_SPEC = {  # residual column -> (target, relative?)
    "dC": ("cond_approach", False), "dE": ("evap_approach", False), "kwtr_resid_pct": ("kw_per_tr", True),
    "dTc": ("tsat_cond", False), "dSub": ("subcool_k", False), "dSh": ("superheat_k", False),
}


def _mad_sd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _persist(series: pd.Series, eligible: pd.Series, w: int) -> pd.Series:
    """Rolling median over eligible rows only (others are NaN and ignored)."""
    x = series.where(eligible)
    if w <= 1:
        return x
    return x.rolling(w, center=True, min_periods=max(1, w // 4)).median().where(eligible)


class ReferenceModel:
    """Healthy-state regression: approaches, kW/TR (and saturation temp, sub-cooling, superheat when
    available) as f(T_evap_out, T_cond_in, tonnage).  Stores healthy-noise levels for sigma thresholds."""

    TOL = {"t_evap_out": 1.5, "t_cw": 2.5}

    def __init__(self):
        self.kind = "const"
        self.mu = None
        self.sd = None
        self.coef: Dict[str, np.ndarray] = {}
        self.lo = None
        self.hi = None
        self.n_fit = 0
        self.source = ""
        self.noise: Dict[str, float] = {}
        self.kwtr_mean = 1.0

    @staticmethod
    def _basis(Z: np.ndarray, kind: str) -> np.ndarray:
        n = len(Z)
        te, tc, ld = Z[:, 0], Z[:, 1], Z[:, 2]
        if kind == "quad":
            return np.column_stack([np.ones(n), te, tc, ld, ld**2, te * ld, tc * ld])
        if kind == "lin":
            return np.column_stack([np.ones(n), te, tc, ld])
        return np.ones((n, 1))

    def fit(self, d: pd.DataFrame, source: str = "", persist_rows: int = 1, noise_inflate: float = 1.25) -> "ReferenceModel":
        X_all = d[FEATS].to_numpy(float)
        ok = np.isfinite(X_all).all(axis=1) & d[CORE_TARGETS].notna().all(axis=1).to_numpy()
        d = d.loc[ok]
        X = X_all[ok]
        n = len(X)
        if n < 8:
            raise ValueError(f"Only {n} usable healthy-reference rows; widen the reference window or provide a healthy log.")
        self.n_fit, self.source = n, source
        self.mu = X.mean(axis=0)
        floors = np.array([1.0, 1.0, max(0.02 * abs(self.mu[2]), 1.0)])
        self.sd = np.maximum(X.std(axis=0), floors)
        span_load = np.ptp(X[:, 2])
        self.kind = "quad" if (n >= 150 and span_load >= 0.2 * max(self.mu[2], 1.0)) else ("lin" if n >= 30 else "const")
        Z = (X - self.mu) / self.sd
        B = self._basis(Z, self.kind)
        reg = np.sqrt(1e-4 * n) * np.eye(B.shape[1])
        reg[0, 0] = 0.0
        targets = CORE_TARGETS + [t for t in EXTRA_TARGETS if t in d and d[t].notna().mean() > 0.9]
        self.coef, self.noise = {}, {}
        w = max(1, int(persist_rows))
        for t in targets:
            m = d[t].notna().to_numpy()
            Bm = B[m]
            A = np.vstack([Bm, reg])
            yy = np.concatenate([d[t].to_numpy(float)[m], np.zeros(B.shape[1])])
            self.coef[t] = np.linalg.lstsq(A, yy, rcond=None)[0]
            res = pd.Series(d[t].to_numpy(float)[m] - Bm @ self.coef[t])
            res = res.rolling(w, center=True, min_periods=max(1, w // 4)).median() if w > 1 else res
            self.noise[t] = _mad_sd(res.to_numpy()) * noise_inflate
        self.kwtr_mean = float(np.nanmedian(d["kw_per_tr"]))
        self.noise["kwtr_pct"] = 100.0 * self.noise["kw_per_tr"] / max(self.kwtr_mean, 1e-6)
        span = np.ptp(X, axis=0)
        tol = np.array([self.TOL["t_evap_out"], self.TOL["t_cw"], 0.15 * max(abs(self.mu[2]), 1.0)])
        margin = np.maximum(0.10 * span, tol)
        self.lo, self.hi = X.min(axis=0) - margin, X.max(axis=0) + margin
        return self

    def predict(self, d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = d[FEATS].to_numpy(float)
        in_dom = pd.Series(np.isfinite(X).all(axis=1) & (X >= self.lo).all(axis=1) & (X <= self.hi).all(axis=1), index=d.index)
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        B = self._basis(np.nan_to_num(Z), self.kind)
        return pd.DataFrame({t: B @ c for t, c in self.coef.items()}, index=d.index), in_dom

    def describe(self) -> str:
        return f"{self.kind} regression on {self.n_fit} healthy rows ({self.source})"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "mu": list(map(float, self.mu)), "sd": list(map(float, self.sd)),
                "lo": list(map(float, self.lo)), "hi": list(map(float, self.hi)), "n_fit": self.n_fit,
                "coef": {k: list(map(float, v)) for k, v in self.coef.items()},
                "noise": {k: float(v) for k, v in self.noise.items()}, "kwtr_mean": self.kwtr_mean, "source": self.source}

    @classmethod
    def from_dict(cls, dct: dict) -> "ReferenceModel":
        m = cls()
        m.kind, m.n_fit, m.source = dct["kind"], dct["n_fit"], dct["source"]
        m.mu, m.sd, m.lo, m.hi = (np.array(dct[k], dtype=float) for k in ["mu", "sd", "lo", "hi"])
        m.coef = {k: np.array(v, dtype=float) for k, v in dct["coef"].items()}
        m.noise, m.kwtr_mean = dict(dct["noise"]), dct["kwtr_mean"]
        return m


# Built-in healthy (SL0) reference fitted on the ASHRAE RP-1043 lab chiller `normal.xls`
# (90-ton centrifugal, R-134a).  Valid for that machine's operating envelope; site chillers fall
# back to the first-N % window automatically.  Noise levels come from held-out 30-min blocks.
BUILTIN_SL0_JSON = '{"kind":"quad","mu":[7.226429652042372,23.579793242562364,58.205261882393785],"sd":[2.2642370675345527,4.308464300341488,23.224827875533],"lo":[2.416666666666665,14.3,17.389961218665853],"hi":[11.944444444444443,32.23888888888889,106.22744596818578],"n_fit":3305,"coef":{"cond_approach":[1.8812692530384394,0.09738477806668877,-0.23327919566926475,0.7363775591887352,0.025606658333492908,0.09418171493955334,-0.15257902944602278],"evap_approach":[3.0996367440686727,1.05832429062093,-0.28962421110033176,0.5087768375124787,-0.06598959664657739,0.06483927686336692,-0.09795336041051018],"kw_per_tr":[0.9030958564447835,-0.016570488968940277,0.10022941227039131,-0.1770174213878012,0.08339993828276421,-0.01263065303479395,-0.0006579182205153771],"tsat_cond":[28.903922441562504,0.08804278387740683,4.059317007594411,2.1262048215391887,0.03309674594275703,0.09200080979614529,-0.16984429874362858],"subcool_k":[3.511617482009146,0.04892434967805234,-0.02787619427690428,1.106035349460721,-0.004010070720693403,0.07123481247697715,-0.15695493311155745],"superheat_k":[2.1062013487899027,0.8368170086467374,-0.22874475810086325,-0.10475814406014122,-0.025159380491089324,-0.007263598597237766,-0.06782565255433322]},"noise":{"cond_approach":0.10493136286176583,"evap_approach":0.1738384382647301,"kw_per_tr":0.01968562654925404,"tsat_cond":0.10228845111662563,"subcool_k":0.23985068749613175,"superheat_k":0.0872921367081879,"kwtr_pct":2.1089034602152923},"kwtr_mean":0.9334531864841449,"source":"ASHRAE RP-1043 normal.xls (SL0), 90-ton lab chiller"}'


def builtin_reference() -> Optional[ReferenceModel]:
    import json
    if BUILTIN_SL0_JSON.startswith("__BUILTIN"):
        return None
    return ReferenceModel.from_dict(json.loads(BUILTIN_SL0_JSON))


def _steady_mask(d: pd.DataFrame, cfg: Config) -> pd.Series:
    dt = _median_dt_s(d["t_s"].values)
    mode = cfg.steady_gate
    if mode == "auto":
        mode = "on" if dt <= 120.0 else "off"
    if mode == "off":
        return d["running"].copy()
    w = max(5, int(round(300.0 / dt)))
    tr = d["tr"]
    cv_tr = tr.rolling(w, center=True, min_periods=w).std() / tr.rolling(w, center=True, min_periods=w).mean()
    sd_te = d["t_evap_out"].rolling(w, center=True, min_periods=w).std()
    sd_tc = d["t_cw"].rolling(w, center=True, min_periods=w).std()
    steady = (cv_tr < 0.03) & (sd_te < 0.15) & (sd_tc < 0.30)
    return (steady.fillna(False)) & d["running"]


def prepare_features(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply the rolling-median noise filter and steady-state gate."""
    d = d.copy()
    for c in SMOOTH_COLS:
        if c in d:
            d[c + "_raw"] = d[c]
            d[c] = _rolling_median(d[c], cfg.median_win)
    d["steady"] = _steady_mask(d, cfg)
    return d


def _persist_rows(d: pd.DataFrame, cfg: Config) -> int:
    return max(1, int(round(cfg.persist_s / _median_dt_s(d["t_s"].values))))


def fit_reference_from_window(d: pd.DataFrame, cfg: Config) -> ReferenceModel:
    n = len(d)
    k = max(int(np.ceil(n * cfg.ref_pct / 100.0)), 10)
    win = d.iloc[:k]
    cand = win[win["steady"]]
    src = f"first {cfg.ref_pct:.0f}% of records, steady rows"
    if len(cand) < 20:
        cand = win[win["running"]]
        src = f"first {cfg.ref_pct:.0f}% of records, running rows (too few steady rows)"
    if len(cand) < 8:
        raise ValueError("The healthy reference window contains almost no running data. Increase the window % or supply a healthy log.")
    return ReferenceModel().fit(cand, src, _persist_rows(d, cfg), noise_inflate=2.0)


def fit_reference_from_log(d: pd.DataFrame, mask: Optional[pd.Series] = None, source: str = "healthy reference log",
                           persist_rows: int = 1) -> ReferenceModel:
    cand = d[d["steady"]] if mask is None else d[d["steady"] & mask]
    if len(cand) < 20:
        cand = d[d["running"]] if mask is None else d[d["running"] & mask]
    return ReferenceModel().fit(cand, source, persist_rows)


def domain_coverage(d: pd.DataFrame, ref: ReferenceModel) -> float:
    base = d["steady"] if d["steady"].sum() >= 20 else d["running"]
    if base.sum() == 0:
        return 0.0
    _, in_dom = ref.predict(d)
    return float((in_dom & base).sum() / base.sum())


def select_reference(feat: pd.DataFrame, cfg: Config) -> ReferenceModel:
    if cfg.ref_mode in ("auto", "builtin"):
        b = builtin_reference()
        if b is not None:
            cov = domain_coverage(feat, b)
            if cfg.ref_mode == "builtin" or cov >= 0.6:
                b.source = f"{b.source}; covers {100 * cov:.0f}% of this file's steady rows"
                return b
    return fit_reference_from_window(feat, cfg)


def resolve_thresholds(cfg: Config, ref: ReferenceModel) -> dict:
    if cfg.thr_mode == "auto":
        k, fl = cfg.auto_k, cfg.auto_floor
        nz = ref.noise
        g = lambda key, default: max(fl, k * (nz.get(key) if np.isfinite(nz.get(key, np.nan)) else default))
        return {"C": g("cond_approach", 0.1), "E": g("evap_approach", 0.1),
                "P": max(1.5, k * nz.get("kwtr_pct", 1.0)), "Sub": g("subcool_k", 0.2), "Sh": g("superheat_k", 0.2),
                "secondary": True}
    return {"C": cfg.thr_cond, "E": cfg.thr_evap, "P": cfg.thr_kwtr_pct, "Sub": np.inf, "Sh": np.inf, "secondary": False}


def classify_faults(d: pd.DataFrame, cfg: Config, ref: ReferenceModel, eval_mask: Optional[pd.Series] = None,
                    signature=None) -> pd.DataFrame:
    """Dual-loop, reference-calibrated thermodynamic classifier.

    Residuals vs the healthy reference at the SAME operating point, median-filtered over a persistence window:
      dC  condenser-approach residual [K]      dE  evaporator-approach residual [K]
      kwtr_resid_pct  specific-power residual [%]   dSub / dSh  sub-cooling / suction-superheat residual [K]
    Type A (fouling):  dC >= thr_C AND dE <  thr_E
    Type B (leak):     dE >= thr_E AND dC < thr_C AND  [kW/TR resid > thr_P  (spec mode)
                       or any of: kW/TR resid > thr_P, sub-cooling drop, superheat rise (auto mode)]
    Both loops high -> Mixed.  A learned signature model (optional) replaces these rules.
    """
    d = d.copy()
    pred, in_dom = ref.predict(d)
    d["in_domain"] = in_dom
    for t in pred.columns:
        d["ref_" + t] = pred[t]
    d["ref_kw_per_tr"] = d["ref_kw_per_tr"].clip(lower=0.1)
    d["ref_cond_approach"] = d["ref_cond_approach"]
    d["ref_evap_approach"] = d["ref_evap_approach"]

    eligible = d["steady"] & d["in_domain"]
    if eval_mask is not None:
        eligible &= eval_mask
    w = _persist_rows(d, cfg)
    raw_res = {
        "dC": d["cond_approach"] - d["ref_cond_approach"],
        "dE": d["evap_approach"] - d["ref_evap_approach"],
        "kwtr_resid_pct": 100.0 * (d["kw_per_tr"] / d["ref_kw_per_tr"] - 1.0),
        "dTc": d["tsat_cond"] - d["ref_tsat_cond"] if "ref_tsat_cond" in d else pd.Series(np.nan, index=d.index),
        "dSub": d["subcool_k"] - d["ref_subcool_k"] if "ref_subcool_k" in d else pd.Series(np.nan, index=d.index),
        "dSh": d["superheat_k"] - d["ref_superheat_k"] if "ref_superheat_k" in d else pd.Series(np.nan, index=d.index),
    }
    for k, v in raw_res.items():
        d[k + "_row"] = v
        d[k] = _persist(v, eligible, w)

    thr = resolve_thresholds(cfg, ref)
    d["thr_cond"], d["thr_evap"], d["thr_kwtr"] = thr["C"], thr["E"], thr["P"]
    hi_c = d["dC"] >= thr["C"]
    hi_e = d["dE"] >= thr["E"]
    hi_p = d["kwtr_resid_pct"] > thr["P"]
    if thr["secondary"]:
        hi_p = hi_p | (d["dSub"] <= -thr["Sub"]).fillna(False) | (d["dSh"] >= thr["Sh"]).fillna(False)

    label = pd.Series(SL0, index=d.index, dtype=object)
    label[hi_c & ~hi_e] = CF
    label[hi_e & hi_p & ~hi_c] = UC
    label[hi_c & hi_e] = MIXED

    classified = eligible & d["dC"].notna()
    if signature is not None:
        X = d.loc[classified, SIG_FEATS].fillna(0.0)
        if len(X):
            label.loc[classified] = signature.predict(X)
    label[~classified] = OOD
    label[~d["steady"]] = TRANSIENT
    label[~d["running"]] = STANDBY
    if eval_mask is not None:
        label[~eval_mask] = "Excluded"
    d["fault_label"] = label
    d["classified"] = label.isin([SL0, CF, UC, MIXED])

    d["kw_expected"] = d["ref_kw_per_tr"] * d["tr"]
    d["delta_kw"] = np.where(d["classified"], d["kw"] - d["kw_expected"], 0.0)
    return d


def primary_diagnosis(d: pd.DataFrame, cfg: Config) -> Tuple[str, dict]:
    c = d[d["classified"]]
    n = len(c)
    if n < 10:
        return "INSUFFICIENT", {"n": n}
    n_cf = int(((c["fault_label"] == CF) | (c["fault_label"] == MIXED)).sum())
    n_uc = int(((c["fault_label"] == UC) | (c["fault_label"] == MIXED)).sum())
    s_cf, s_uc = 100.0 * n_cf / n, 100.0 * n_uc / n
    shares = {"n": n, "share_cf": s_cf, "share_uc": s_uc, "share_sl0": 100.0 * (c["fault_label"] == SL0).mean()}
    m = cfg.min_fault_share
    if s_cf >= m and s_uc >= m:
        return (CF if s_cf >= s_uc * 1.5 else UC if s_uc >= s_cf * 1.5 else MIXED), shares
    if s_cf >= m:
        return CF, shares
    if s_uc >= m:
        return UC, shares
    return SL0, shares


# ----------------------------------------------------------------------------
# 4. Time span & financials
# ----------------------------------------------------------------------------
def _dt_hours(t_s: np.ndarray) -> np.ndarray:
    t = np.asarray(t_s, dtype=float)
    if len(t) < 2:
        return np.array([1.0 / 60.0] * len(t))
    med = _median_dt_s(t)
    dt = np.diff(t, append=t[-1] + med)
    dt = np.clip(dt, 0.0, 5.0 * med)  # a data gap never counts as running time
    return dt / 3600.0


def financials(d: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, dict]:
    d = d.copy()
    t = d["t_s"].to_numpy(float)
    total_hours = (np.nanmax(t) - np.nanmin(t)) / 3600.0
    if not np.isfinite(total_hours) or total_hours <= 0:
        total_hours = len(d) * cfg.assumed_dt_s / 3600.0
    total_days = max(total_hours / 24.0, 0.01)
    d["dt_h"] = _dt_hours(t)
    d["waste_kwh"] = np.maximum(d["delta_kw"], 0.0) * d["dt_h"]
    d["waste_aed"] = d["waste_kwh"] * cfg.tariff
    total = float(d["waste_aed"].sum())
    fin = {
        "total_hours": total_hours,
        "total_days": total_days,
        "total_waste_aed": total,
        "daily_waste_aed": total / total_days,
        "total_waste_kwh": float(d["waste_kwh"].sum()),
    }
    return d, fin


def run_pipeline(raw: pd.DataFrame, cfg: Config, ref: Optional[ReferenceModel] = None, eval_mask_fn=None, signature=None):
    """normalise -> physics -> features -> (select/fit reference) -> classify -> financials."""
    norm, meta = normalize_bms_columns(raw, cfg)
    ph = physics_engine(norm, cfg)
    feat = prepare_features(ph, cfg)
    if ref is None:
        ref = select_reference(feat, cfg)
    em = eval_mask_fn(feat) if eval_mask_fn else None
    scored = classify_faults(feat, cfg, ref, em, signature)
    scored, fin = financials(scored, cfg)
    dx, shares = primary_diagnosis(scored, cfg)
    cov = 100.0 * scored["classified"].sum() / max(int(scored["running"].sum()), 1)
    if cov < 30.0 and dx == SL0:
        dx = "INDETERMINATE"
    c = scored[scored["classified"]]
    summary = {
        **fin,
        "diagnosis": dx,
        "shares": shares,
        "coverage_pct": cov,
        "kwtr_actual": float(c["kw_per_tr"].median()) if len(c) else np.nan,
        "kwtr_ideal": float(c["ref_kw_per_tr"].median()) if len(c) else np.nan,
        "med_dC": float(c["dC"].median()) if len(c) else np.nan,
        "med_dE": float(c["dE"].median()) if len(c) else np.nan,
        "ref": ref.describe(),
        "thr": resolve_thresholds(cfg, ref),
    }
    if "tr_meas" in scored and "tr_est" in scored:
        ok = scored["running"] & scored["tr_meas"].gt(5)
        if ok.any():
            summary["tr_ratio"] = float((scored.loc[ok, "tr_est"] / scored.loc[ok, "tr_meas"]).median())
    return scored, meta, summary


# ----------------------------------------------------------------------------
# 5. Live UAE simulator (emits raw RP-1043-style columns, degF / psig)
# ----------------------------------------------------------------------------
WEATHER = {
    "Summer (Jul-Aug)": dict(db=(40.0, 6.0), wb=(28.0, 1.5)),
    "Shoulder (Apr / Oct)": dict(db=(33.0, 6.0), wb=(22.0, 2.0)),
    "Winter (Jan)": dict(db=(23.0, 5.0), wb=(16.0, 2.0)),
    "Heat-wave (extreme)": dict(db=(46.0, 5.0), wb=(31.0, 1.5)),
}
SCENARIOS = ["Normal", "Tube Fouling", "Leak", "Extreme Weather"]


def simulate_uae(weather: str, scenario: str, days: int = 7, step_min: int = 15, design_tr: float = 500.0,
                 eta: float = 0.69, seed: int = 7, fault_start_frac: float = 0.30,
                 fouling_k: float = 4.0, leak_evap_k: float = 3.5, leak_kwtr_pct: float = 12.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / step_min)
    t_h = np.arange(n) * step_min / 60.0
    hod = t_h % 24.0
    wx = WEATHER["Heat-wave (extreme)"] if scenario == "Extreme Weather" else WEATHER[weather]
    diurnal = np.sin(2 * np.pi * (hod - 9.0) / 24.0)
    day_walk = np.repeat(rng.normal(0, 0.8, days + 1), int(24 * 60 / step_min))[:n]
    tdb = wx["db"][0] + wx["db"][1] * diurnal + day_walk + rng.normal(0, 0.3, n)
    twb = wx["wb"][0] + wx["wb"][1] * diurnal + 0.5 * day_walk + rng.normal(0, 0.2, n)

    lf = np.clip(0.30 + 0.45 * 0.5 * (1 + diurnal) + 0.10 * (tdb - 30.0) / 10.0 + rng.normal(0, 0.03, n), 0.25, 1.0)
    tr_true = design_tr * lf

    tci = twb + 3.5 + rng.normal(0, 0.12, n)
    teo = 6.7 + rng.normal(0, 0.10, n)
    tei = teo + 5.5 * lf

    ramp = np.clip((t_h - fault_start_frac * t_h[-1]) / ((1 - fault_start_frac) * t_h[-1]), 0, 1)
    fouling = ramp * fouling_k if scenario == "Tube Fouling" else np.zeros(n)
    leak_e = ramp * leak_evap_k if scenario == "Leak" else np.zeros(n)
    leak_p = ramp * leak_kwtr_pct / 100.0 if scenario == "Leak" else np.zeros(n)

    cond_h = 2.0 + 0.03 * (tdb - 25.0) + 1.4 * (lf - 0.7)
    evap_h = 1.2 + 1.3 * lf
    # iterate once: kW/TR depends on lift, lift depends on condenser leaving temp, which depends on kW/TR
    kwtr = np.full(n, 0.65)
    for _ in range(3):
        tco = tci + 6.0 * lf * (1 + kwtr / KW_PER_TR) / 1.19
        tsc = tco + cond_h + fouling
        tse = teo - evap_h - leak_e
        te_k, tc_k = tse + 273.15, tsc + 273.15
        cop = 0.60 * te_k / np.maximum(tc_k - te_k, 5.0)
        kwtr = KW_PER_TR / cop * (1 + 0.15 * (1 - lf) ** 2) * (1 + leak_p)
    kw = kwtr * tr_true

    # state points consistent with the eta used in physics_engine()
    sub = 3.0
    t_suc = tse + 4.0 + rng.normal(0, 0.2, n)
    h_suc = hg_r134a(tse) + CP_VAP_SUCTION * (t_suc - tse)
    h_liq = hf_r134a(tsc - sub)
    dh = eta * (h_suc - h_liq) * kwtr / KW_PER_TR
    h_dis = h_suc + dh
    t_dis = tsc + np.maximum(h_dis - hg_r134a(tsc), 0.5) / CP_VAP_DISCHARGE

    # sensor noise
    p_c = psat_r134a_kpa(tsc) * (1 + rng.normal(0, 0.003, n))
    p_e = psat_r134a_kpa(tse) * (1 + rng.normal(0, 0.003, n))
    kw_m = kw * (1 + rng.normal(0, 0.005, n))
    tco_m = tco + rng.normal(0, 0.10, n)
    teo_m = teo + rng.normal(0, 0.05, n)
    v_gen = 460.0 if design_tr <= 150 else 3300.0
    amps = kw_m * 1000.0 / (np.sqrt(3) * v_gen * 0.88)

    f = lambda c: c * 9.0 / 5.0 + 32.0
    kpa_to_psig = lambda p: p / PSI_TO_KPA - ATM_PSI
    truth = np.where(ramp >= 0.40, {"Tube Fouling": CF, "Leak": UC}.get(scenario, SL0), np.where(ramp <= 0.15, SL0, TRANSITIONAL))
    if scenario in ("Normal", "Extreme Weather"):
        truth = np.full(n, SL0, dtype=object)

    ts = pd.Timestamp("2026-07-01 00:00:00") + pd.to_timedelta(t_h, unit="h")
    return pd.DataFrame({
        "Timestamp": ts,
        "TCI": f(tci), "TCO": f(tco_m), "TEI": f(tei), "TEO": f(teo_m),
        "PRC": kpa_to_psig(p_c), "PRE": kpa_to_psig(p_e),
        "TR_dis": f(t_dis), "T_suc": f(t_suc),
        "Amps": amps, "kW": kw_m, "Tamb": f(tdb), "truth": truth,
    })


# ----------------------------------------------------------------------------
# 6. File reading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_sheets(name: str, data: bytes) -> List[str]:
    if name.lower().endswith(".csv"):
        return []
    return _excel_file(name, data).sheet_names


def _excel_file(name: str, data: bytes) -> pd.ExcelFile:
    engine = "xlrd" if name.lower().endswith(".xls") else "openpyxl"
    try:
        return pd.ExcelFile(io.BytesIO(data), engine=engine)
    except ImportError as e:
        raise ImportError(f"Reading {name} needs the '{engine}' package: pip install {engine}") from e


def _pick_sheet(names: List[str]) -> str:
    for n in names:
        if "complete" in n.lower():
            return n
    return names[0]


@st.cache_data(show_spinner=False)
def read_table(name: str, data: bytes, sheet: Optional[str] = None) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    xl = _excel_file(name, data)
    sh = sheet if sheet in xl.sheet_names else _pick_sheet(xl.sheet_names)
    return xl.parse(sh)


# ----------------------------------------------------------------------------
# 7. Benchmark helpers
# ----------------------------------------------------------------------------
def infer_label(filename: str) -> str:
    f = filename.lower()
    if any(k in f for k in ["normal", "sl0", "healthy", "baseline"]):
        return SL0
    if f.startswith("cf") or "foul" in f:
        return CF
    if f.startswith(("rl", "uc")) or "leak" in f or "undercharge" in f:
        return UC
    return SL0


def benchmark_uploaded(files: Dict[str, Tuple[pd.DataFrame, str]], cfg: Config, ref_choice: str = "builtin"):
    """files: {name: (raw_df, truth_label)}.

    ref_choice "builtin": healthy reference shipped with the app (fitted on RP-1043 normal.xls).
    ref_choice "sl0_file": fit on the even 30-min blocks of the uploaded SL0 file; that file is scored on its odd blocks only.
    Returns (reference, per_file_df, y_true, y_pred, frames) where frames = [(name, scored_df, truth)].
    """
    prepared = {}
    for name, (raw, truth) in files.items():
        norm, _ = normalize_bms_columns(raw, cfg)
        prepared[name] = (prepare_features(physics_engine(norm, cfg), cfg), truth)
    base_name = None
    if ref_choice == "builtin":
        ref = builtin_reference()
        if ref is None:
            raise ValueError("Built-in reference is not available in this build.")
    else:
        sl0_names = [n for n, (_, t) in prepared.items() if t == SL0]
        if not sl0_names:
            raise ValueError("Upload at least one file labelled SL0_Healthy, or choose the built-in reference.")
        base_name = sl0_names[0]
        base, _ = prepared[base_name]
        even = ((base["t_s"] // 1800).astype(int) % 2 == 0)
        ref = fit_reference_from_log(base, even, f"{base_name}: even 30-min blocks", _persist_rows(base, cfg))
    rows, ytrue, ypred, frames = [], [], [], []
    for name, (feat, truth) in prepared.items():
        em = ((feat["t_s"] // 1800).astype(int) % 2 == 1) if name == base_name else None
        sc = classify_faults(feat, cfg, ref, em)
        frames.append((name, sc, truth))
        c = sc[sc["classified"]]
        ytrue += [truth] * len(c)
        ypred += list(c["fault_label"])
        counts = c["fault_label"].value_counts()
        rows.append({
            "file": name, "truth": truth, "rows scored": len(c),
            "med dCond [K]": round(c["dC"].median(), 2) if len(c) else np.nan,
            "med dEvap [K]": round(c["dE"].median(), 2) if len(c) else np.nan,
            "med kW/TR [%]": round(c["kwtr_resid_pct"].median(), 1) if len(c) else np.nan,
            "med dSubcool [K]": round(c["dSub"].median(), 2) if len(c) else np.nan,
            "med dSuperheat [K]": round(c["dSh"].median(), 2) if len(c) else np.nan,
            **{f"% {k}": round(100 * counts.get(k, 0) / max(len(c), 1), 1) for k in [SL0, CF, UC, MIXED]},
        })
    return ref, pd.DataFrame(rows), ytrue, ypred, frames


def train_signature(frames: List[Tuple[str, pd.DataFrame, str]], step: int = 10):
    """Learn a fault signature (standardised logistic regression on the residual vector SIG_FEATS)
    from labelled, already-scored files.  Returns (model, report_df, confusion_df, validation_note)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xs, ys, gs, bl = [], [], [], []
    for name, sc, truth in frames:
        c = sc[sc["classified"]].iloc[::step]
        if len(c) == 0:
            continue
        Xs.append(c[SIG_FEATS].fillna(0.0))
        ys += [truth] * len(c)
        gs += [name] * len(c)
        bl += list(((c["t_s"] // 1800).astype(int) % 2).values)
    if not Xs:
        raise ValueError("No classified rows available for training.")
    X = pd.concat(Xs, ignore_index=True)
    y, g, bl = np.array(ys), np.array(gs), np.array(bl)
    if len(set(y)) < 2:
        raise ValueError("Need at least two different ground-truth classes to train.")
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
    files_per_class = {k: len(set(g[y == k])) for k in sorted(set(y))}
    yp = np.empty(len(y), dtype=object)
    if all(v >= 2 for v in files_per_class.values()) and len(files_per_class) >= 2:
        for grp in sorted(set(g)):
            te = g == grp
            if len(set(y[~te])) < 2:
                yp[te] = y[~te][0]
                continue
            yp[te] = mk().fit(X[~te], y[~te]).predict(X[te])
        note = f"Leave-one-FILE-out validation across {len(set(g))} files (files per class: {files_per_class}). This is the honest estimate."
        mask = np.ones(len(y), dtype=bool)
    else:
        tr, te = bl == 0, bl == 1
        yp[:] = ""
        yp[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        mask = te
        note = (f"Only {files_per_class} file(s) per class, so validation used held-out 30-min time blocks of the SAME files. "
                "This is OPTIMISTIC (the model can learn file identity). Load >= 2 files per class for leave-one-file-out validation.")
    labs = [l for l in CLASS_LABELS if l in set(y)]
    rep = pd.DataFrame(classification_report(y[mask], yp[mask], labels=labs, zero_division=0, output_dict=True)).T.round(3)
    cm = confusion_matrix(y[mask], yp[mask], labels=labs)
    cm_df = pd.DataFrame(cm, index=[f"true {l}" for l in labs], columns=[f"pred {l}" for l in labs])
    model = mk().fit(X, y)
    return model, rep, cm_df, note


def benchmark_synthetic(cfg: Config, weather: str, runs_per_class: int = 3):
    ytrue, ypred, rows = [], [], []
    for scen in SCENARIOS:
        for k in range(runs_per_class):
            raw = simulate_uae(weather, scen, days=7, step_min=15, eta=cfg.eta, seed=100 + 17 * k + len(scen))
            scored, _, _ = run_pipeline(raw, replace(cfg, steady_gate="off"))
            m = scored["classified"].to_numpy()
            truth = raw["truth"].to_numpy()
            # align by original row order
            tr = pd.Series(truth, index=np.arange(len(raw)))
            t_aligned = tr.reindex(scored["_row"].to_numpy()).to_numpy()
            sel = m & (t_aligned != TRANSITIONAL)
            ytrue += list(t_aligned[sel])
            ypred += list(scored.loc[sel, "fault_label"])
            rows.append({"scenario": scen, "seed": k, "rows scored": int(sel.sum()),
                         "accuracy %": round(100 * float((t_aligned[sel] == scored.loc[sel, "fault_label"].to_numpy()).mean()), 1) if sel.any() else np.nan})
    return pd.DataFrame(rows), ytrue, ypred


def report_frames(ytrue: List[str], ypred: List[str]):
    rep = classification_report(ytrue, ypred, labels=CLASS_LABELS, zero_division=0, output_dict=True)
    rep_df = pd.DataFrame(rep).T.round(3)
    pred_labels = CLASS_LABELS + ([MIXED] if MIXED in set(ypred) else [])
    cm = confusion_matrix(ytrue, ypred, labels=pred_labels)
    cm_df = pd.DataFrame(cm[: len(CLASS_LABELS)], index=[f"true {l}" for l in CLASS_LABELS], columns=[f"pred {l}" for l in pred_labels])
    return rep_df, cm_df


# ----------------------------------------------------------------------------
# 8. Plotting
# ----------------------------------------------------------------------------
def _decimate(d: pd.DataFrame, max_pts: int = 4000) -> pd.DataFrame:
    if len(d) <= max_pts:
        return d
    return d.iloc[:: int(np.ceil(len(d) / max_pts))]


def fig_approach(d: pd.DataFrame, cfg: Config, title: str) -> go.Figure:
    p = _decimate(d)
    ok = p["classified"]
    x = p["x"]
    fig = go.Figure()
    cond = p["cond_approach"].where(ok)
    evap = p["evap_approach"].where(ok)
    fig.add_trace(go.Scatter(x=x, y=cond, name="Condenser approach (actual)", line=dict(color="#d62728", width=1.6), connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=p["ref_cond_approach"].where(ok), name="Condenser approach (healthy ref)", line=dict(color="#d62728", dash="dash", width=1.2), connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=(p["ref_cond_approach"] + p["thr_cond"]).where(ok), name="Type A trigger", line=dict(color="#d62728", dash="dot", width=0.8), opacity=0.6, connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=evap, name="Evaporator approach (actual)", line=dict(color="#1f77b4", width=1.6), connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=p["ref_evap_approach"].where(ok), name="Evaporator approach (healthy ref)", line=dict(color="#1f77b4", dash="dash", width=1.2), connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=(p["ref_evap_approach"] + p["thr_evap"]).where(ok), name="Type B trigger", line=dict(color="#1f77b4", dash="dot", width=0.8), opacity=0.6, connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=p["cond_approach_g14"].where(ok), name="ASHRAE G14 baseline (2.0+0.03(T-25))", line=dict(color="#7f7f7f", dash="longdash", width=1.0), connectgaps=False))
    fig.update_layout(title=title, yaxis_title="Approach temperature [K]", xaxis_title="Elapsed hours" if not np.issubdtype(np.asarray(d["x"]).dtype, np.datetime64) else "Time",
                      height=430, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25), template="plotly_white")
    return fig


def fig_waste(d: pd.DataFrame, fin: dict) -> go.Figure:
    t = d["t_s"].to_numpy(float)
    if fin["total_days"] >= 2.0:
        bucket, unit, label = 86400.0, "AED/day", "Day"
    else:
        bucket, unit, label = 3600.0, "AED/hour", "Hour"
    g = d.assign(_b=(t - t.min()) // bucket).groupby("_b")["waste_aed"].sum().reset_index()
    g["_b"] = g["_b"] + 1
    fig = go.Figure(go.Scatter(x=g["_b"], y=g["waste_aed"], mode="lines", fill="tozeroy", line=dict(color="#ff7f0e"), name="Energy waste"))
    fig.update_layout(title=f"Financial waste ({unit})", xaxis_title=f"{label} index", yaxis_title=unit, height=320,
                      margin=dict(l=10, r=10, t=50, b=10), template="plotly_white")
    return fig


def fig_confusion(cm_df: pd.DataFrame) -> go.Figure:
    z = cm_df.values
    fig = go.Figure(go.Heatmap(z=z, x=list(cm_df.columns), y=list(cm_df.index), colorscale="Blues", text=z, texttemplate="%{text}", showscale=False))
    fig.update_layout(title="Confusion matrix (row-level, steady & in-domain rows)", height=380, margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(autorange="reversed"))
    return fig


# ----------------------------------------------------------------------------
# 9. Streamlit UI
# ----------------------------------------------------------------------------
def _stretch(fn, *args, **kwargs):
    """Call a Streamlit widget with full-width across old/new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def show_fig(fig):
    _stretch(st.plotly_chart, fig)


def show_df(df, **kw):
    _stretch(st.dataframe, df, **kw)


def render_dashboard(scored: pd.DataFrame, meta: dict, summary: dict, cfg: Config, title: str, tab3: bool = False):
    for w in meta.get("warnings", []):
        st.warning(w)
    if summary["coverage_pct"] < 50.0 and summary["shares"].get("n", 0) >= 10:
        st.warning(
            f"Only {summary['coverage_pct']:.0f}% of running rows lie inside the healthy-reference operating envelope "
            "(or pass the steady-state gate). The diagnosis and waste figures cover that subset only; "
            "supply a healthy reference log that spans the load / temperature range for full coverage."
        )
    dx = summary["diagnosis"]
    c1, c2, c3, c4 = st.columns(4)
    if tab3:
        c1.metric("Total Period Waste", f"{summary['total_waste_aed']:,.0f} AED")
        c2.metric("Average Daily Waste Rate", f"{summary['daily_waste_aed']:,.0f} AED/day")
        c3.metric("Primary Dataset Diagnosis", DISPLAY_NAME.get(dx, dx))
        c4.metric("Span", f"{summary['total_hours']:.1f} h / {summary['total_days']:.2f} d")
    else:
        c1.metric("Primary Diagnosis", DISPLAY_NAME.get(dx, dx))
        c2.metric("Daily Waste Rate", f"{summary['daily_waste_aed']:,.0f} AED/day")
        kr, ki = summary["kwtr_actual"], summary["kwtr_ideal"]
        c3.metric("kW/TR actual vs ideal", f"{kr:.3f} / {ki:.3f}" if np.isfinite(kr) else "n/a",
                  delta=f"{100 * (kr / ki - 1):+.1f}%" if np.isfinite(kr) and ki else None, delta_color="inverse")
        c4.metric("Total Period Waste", f"{summary['total_waste_aed']:,.0f} AED")
    sh = summary["shares"]
    if "share_cf" in sh:
        st.caption(
            f"Scored rows: {sh['n']:,} ({summary['coverage_pct']:.0f}% of running rows) | healthy {sh['share_sl0']:.1f}% | "
            f"condenser-fouling signature {sh['share_cf']:.1f}% | leak signature {sh['share_uc']:.1f}% | "
            f"reference: {summary['ref']} | time axis: {meta['time_info']['source']} | voltage {meta['voltage']:.0f} V | "
            f"{meta['temp_unit']} / {meta['pressure_unit']} detected | enthalpy dh from {scored['dh_source'].iloc[0]}"
        )
    else:
        st.info("Too few steady, in-domain rows to diagnose. Widen the reference window, relax steady-state gating, or supply a healthy reference log.")
    th = summary.get("thr")
    if th:
        st.caption(f"Triggers used: condenser approach +{th['C']:.2f} K, evaporator approach +{th['E']:.2f} K, kW/TR +{th['P']:.1f}%"
                   + (f", sub-cooling -{th['Sub']:.2f} K, superheat +{th['Sh']:.2f} K" if th.get("secondary") else "")
                   + f" | persistence window {cfg.persist_s:.0f} s")
    if summary["diagnosis"] == "INDETERMINATE":
        st.warning("Indeterminate: too few rows fall inside the healthy-reference operating envelope to call this chiller healthy. "
                   "Upload a healthy reference log from the same plant (Tab 3) or widen the reference window.")
    for nt in meta.get("notes", []):
        st.caption(nt)
    if "tr_ratio" in summary:
        st.caption(f"Enthalpy tonnage / measured tonnage (median): {summary['tr_ratio']:.3f}. If far from 1.0, adjust compressor efficiency eta in the sidebar.")
    show_fig(fig_approach(scored, cfg, title))
    show_fig(fig_waste(scored, summary))


def main():
    st.set_page_config(page_title="UAE Chiller FDD", page_icon="❄️", layout="wide")
    st.title("❄️ UAE Centrifugal Chiller Predictive Maintenance & FDD")
    st.caption("First-principles R-134a enthalpy physics + reference-calibrated thermodynamic fault classifier "
               "(optionally with a signature learned from your labelled files).")
    ss = st.session_state

    sb = st.sidebar
    sb.header("Live simulator")
    weather = sb.selectbox("Weather profile", list(WEATHER.keys()), index=0)
    scenario = sb.selectbox("Simulation scenario", SCENARIOS, index=1)
    design_tr = sb.number_input("Simulated plant design capacity (TR)", 100, 3000, 500, step=50)
    sim_seed = sb.number_input("Random seed", 0, 9999, 7)

    sb.header("Classifier & economics")
    tariff = sb.number_input("DEWA tariff (AED/kWh)", 0.05, 2.0, 0.44, step=0.01, format="%.2f")
    eta = sb.slider("Compressor/motor effective efficiency (eta)", 0.50, 1.00, 0.69, 0.01,
                    help="Scales kW to refrigerant-side power in the enthalpy tonnage. 0.69 was calibrated on the RP-1043 lab chiller; calibrate on your plant.")
    profile = sb.selectbox("Detection sensitivity", ["Auto: 3-sigma of healthy noise (recommended)", "Spec: 1.5 K / 1.5 K / +10 %", "Custom"], index=0)
    thr_mode, thr_c, thr_e, thr_p = "auto", 1.5, 1.5, 10.0
    if profile.startswith("Spec"):
        thr_mode = "spec"
    elif profile == "Custom":
        thr_mode = "custom"
        thr_c = sb.number_input("Condenser approach trigger (K)", 0.1, 10.0, 1.5, 0.05)
        thr_e = sb.number_input("Evaporator approach trigger (K)", 0.1, 10.0, 1.5, 0.05)
        thr_p = sb.number_input("kW/TR residual trigger (%)", -5.0, 50.0, 10.0, 0.5)
    ref_choice = sb.selectbox("Healthy reference", ["Auto (built-in RP-1043 SL0 if the data fits, else first-N % window)",
                                                    "Built-in RP-1043 SL0 reference", "First-N % window of the file"], index=0)
    ref_mode = "auto" if ref_choice.startswith("Auto") else "builtin" if ref_choice.startswith("Built-in") else "window"
    med_win = sb.slider("Rolling-median samples (noise filter)", 1, 61, 5, 2)
    persist_s = sb.slider("Persistence window (s)", 0, 7200, 1800, 300, help="Residuals are median-filtered over this window before classification.")
    steady = sb.selectbox("Steady-state gating", ["auto", "on", "off"], index=0,
                          help="auto = on when sample interval <= 2 min (lab data), off for 15-min BMS trends.")
    min_share = sb.slider("Min. share of rows to declare a dataset-level fault (%)", 5, 90, 25, 5)
    cap_src = sb.selectbox("Capacity source", ["Enthalpy estimate (flow-meter-free)", "Measured tons column (if present)"], index=0)
    signature = None
    if "signature" in ss:
        if sb.checkbox("Use learned signature model (trained in Tab 2)", value=True):
            signature = ss["signature"]
    with sb.expander("Unit overrides / time fallback"):
        tu = st.selectbox("Temperature unit", ["auto", "degF", "degC"])
        pu = st.selectbox("Pressure unit", ["auto", "psig", "kPa_abs", "kPa_g", "bar_abs"])
        adt = st.number_input("Assumed sample interval (s) if no time column", 1.0, 3600.0, 60.0)

    cfg = Config(tariff=tariff, eta=eta, median_win=med_win, thr_mode=thr_mode, thr_cond=thr_c, thr_evap=thr_e, thr_kwtr_pct=thr_p,
                 persist_s=float(persist_s), ref_mode=ref_mode, min_fault_share=float(min_share),
                 capacity_source="measured" if cap_src.startswith("Measured") else "enthalpy",
                 steady_gate=steady, temp_unit=tu, pressure_unit=pu, assumed_dt_s=adt)

    tab1, tab2, tab3 = st.tabs(["🇦🇪 Live UAE Simulator", "📐 RP-1043 Benchmark Validation", "🧾 Zero-Risk BMS Log File Audit"])

    # ------------------------------------------------------------------ Tab 1
    with tab1:
        st.subheader(f"{scenario} | {weather if scenario != 'Extreme Weather' else 'Heat-wave (extreme)'}")
        ref_pct1 = st.slider("Healthy reference window (first % of records)", 5, 30, 12, key="ref1")
        c1cfg = replace(cfg, ref_pct=float(ref_pct1), steady_gate="off", ref_mode="window", persist_s=min(cfg.persist_s, 3600.0))
        try:
            raw = simulate_uae(weather, scenario, design_tr=float(design_tr), eta=eta, seed=int(sim_seed))
            scored, meta, summ = run_pipeline(raw, c1cfg, signature=None)
            render_dashboard(scored, meta, summ, c1cfg, "Condenser & evaporator approach: actual vs weather-normalised healthy reference")
            st.caption("Simulator note: the fault ramps in after ~30 % of the run so the healthy window is clean. "
                       "'Extreme Weather' is a healthy chiller in a heat-wave, i.e. a false-alarm test of the weather normalisation. "
                       "The simulator always uses the first-N % window reference and the rule-based classifier.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Simulation failed: {e}")

    # ------------------------------------------------------------------ Tab 2
    with tab2:
        st.subheader("Classifier validation against ground-truth labels")
        mode = st.radio("Benchmark source", ["Upload ASHRAE RP-1043 files", "Synthetic ground-truth generator"], horizontal=True)
        if mode.startswith("Upload"):
            st.write("Upload any number of RP-1043 workbooks (`normal`, `cf*`, `rl*`...). Labels are inferred from file names; correct them if needed. "
                     "Using the built-in reference, each file is judged against the healthy SL0 model; the SL0 file you upload is then just another test case.")
            bm_ref = st.radio("Healthy reference for the benchmark", ["Built-in SL0 reference", "Fit on the uploaded SL0 file (held-out blocks)"], horizontal=True)
            ups = st.file_uploader("RP-1043 workbooks", type=["xls", "xlsx", "csv"], accept_multiple_files=True, key="bm_up")
            if ups:
                labels = {}
                for u in ups:
                    labels[u.name] = st.selectbox(f"Ground truth for {u.name}", CLASS_LABELS, index=CLASS_LABELS.index(infer_label(u.name)), key=f"lab_{u.name}")
                if st.button("Run benchmark", type="primary"):
                    try:
                        with st.spinner("Reading workbooks and classifying..."):
                            files = {u.name: (read_table(u.name, u.getvalue()), labels[u.name]) for u in ups}
                            ref, per_file, yt, yp, frames = benchmark_uploaded(files, cfg, "builtin" if bm_ref.startswith("Built") else "sl0_file")
                        ss["bm"] = {"ref": ref.describe(), "per_file": per_file, "yt": yt, "yp": yp, "frames": frames}
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Benchmark failed: {e}")
            bm = ss.get("bm")
            if bm:
                st.caption(f"Reference: {bm['ref']}")
                rep_df, cm_df = report_frames(bm["yt"], bm["yp"])
                st.markdown("**Per-file residual summary** (what the classifier actually sees; send this table back if you want thresholds tuned)")
                show_df(bm["per_file"])
                st.download_button("⬇️ Download per-file summary CSV", bm["per_file"].to_csv(index=False).encode("utf-8"),
                                   file_name="rp1043_per_file_summary.csv", mime="text/csv")
                st.markdown("**sklearn classification report (rule-based classifier)**")
                show_df(rep_df)
                show_fig(fig_confusion(cm_df))
                st.markdown("---")
                st.markdown("### Learn the fault signature from these labelled files")
                st.write("Fits a standardised logistic regression on the residual vector (condenser approach, evaporator approach, kW/TR, "
                         "saturation temp, sub-cooling, superheat). With >= 2 files per class it is validated leave-one-file-out. "
                         "Once trained, tick the sidebar box to use it in Tab 3.")
                if st.button("Train signature model"):
                    try:
                        model, trep, tcm, note = train_signature(bm["frames"])
                        ss["signature"] = model
                        ss["sig_report"] = (trep, tcm, note)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Training failed: {e}")
                if "sig_report" in ss:
                    trep, tcm, note = ss["sig_report"]
                    st.info(note)
                    show_df(trep)
                    show_fig(fig_confusion(tcm))
        else:
            runs = st.slider("Runs per scenario", 1, 6, 3)
            if st.button("Generate & evaluate", type="primary"):
                with st.spinner("Simulating and classifying..."):
                    per_run, yt, yp = benchmark_synthetic(replace(cfg, ref_pct=12.0, ref_mode="window"), weather, runs)
                rep_df, cm_df = report_frames(yt, yp)
                show_df(per_run)
                show_df(rep_df)
                show_fig(fig_confusion(cm_df))
                st.caption("Synthetic truth marks a row faulty only once the injected ramp is >= 40 % complete (rows between 15 % and 40 % are excluded as transitional). "
                           "Synthetic results verify the logic, not field accuracy.")

    # ------------------------------------------------------------------ Tab 3
    with tab3:
        st.subheader("Read-only audit of a BMS / lab log")
        up = st.file_uploader("BMS log or RP-1043 workbook", type=["csv", "xlsx", "xls"], key="audit_up")
        ref_log = st.file_uploader("Optional: known-healthy reference log from the same plant (recommended if the audited file may already be faulty at row 0)",
                                   type=["csv", "xlsx", "xls"], key="ref_up")
        ref_pct = st.slider("Healthy reference window (first % of records) - used only when no built-in/healthy-log reference applies", 5, 30, 12, key="ref3")
        if up is not None:
            try:
                data = up.getvalue()
                sheets = list_sheets(up.name, data)
                sheet = st.selectbox("Sheet", sheets, index=sheets.index(_pick_sheet(sheets))) if len(sheets) > 1 else None
                raw3 = read_table(up.name, data, sheet)
                st.caption(f"Loaded {len(raw3):,} rows x {raw3.shape[1]} columns")
                c3cfg = replace(cfg, ref_pct=float(ref_pct))
                ref_model = None
                if ref_log is not None:
                    rnorm, _ = normalize_bms_columns(read_table(ref_log.name, ref_log.getvalue()), c3cfg)
                    rfeat = prepare_features(physics_engine(rnorm, c3cfg), c3cfg)
                    ref_model = fit_reference_from_log(rfeat, None, f"healthy log '{ref_log.name}'", _persist_rows(rfeat, c3cfg))
                scored3, meta3, summ3 = run_pipeline(raw3, c3cfg, ref=ref_model, signature=signature)
                if ref_model is None and summ3["ref"].find("first") >= 0:
                    st.info("Reference = first-N % window of this file. A fault already present at row 0 will be absorbed into the baseline; upload a healthy log above to avoid that.")
                if signature is not None:
                    st.info("Fault labels come from the learned signature model.")
                render_dashboard(scored3, meta3, summ3, c3cfg, "Approach temperatures vs healthy reference", tab3=True)
                with st.expander("Detected column mapping"):
                    show_df(pd.DataFrame({"canonical": list(meta3["mapping"].keys()), "source column": list(meta3["mapping"].values())}))
                cols = ["_row", "x", "t_s", "running", "steady", "in_domain", "fault_label", "tr", "kw", "kw_per_tr", "ref_kw_per_tr",
                        "cond_approach", "ref_cond_approach", "dC", "evap_approach", "ref_evap_approach", "dE", "kwtr_resid_pct",
                        "dSub", "dSh", "delta_kw", "waste_kwh", "waste_aed"]
                out = scored3[[c for c in cols if c in scored3]].rename(columns={"_row": "source_row"})
                st.download_button("⬇️ Download scored audit CSV", out.to_csv(index=False).encode("utf-8"),
                                   file_name="chiller_audit_scored.csv", mime="text/csv")
            except Exception as e:  # noqa: BLE001
                st.error(f"Audit failed: {e}")
        else:
            st.info("Upload a log to begin. Nothing is written back to your BMS; the audit is read-only.")


if __name__ == "__main__":
    main()
