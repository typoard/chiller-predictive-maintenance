"""
Centrifugal Chiller Predictive Maintenance Platform  -  UAE edition
===================================================================
Run:   streamlit run app.py
Deps:  pip install streamlit pandas numpy scikit-learn plotly openpyxl xlrd
       (openpyxl reads .xlsx, xlrd reads legacy .xls such as ASHRAE RP-1043 cf12.xls)

TABS
  1. Live UAE Operational Simulator  (Normal / Tube Fouling / Leak / Weather Spike)
  2. ASHRAE RP-1043 Benchmark Validation (classification report + confusion matrix)
  3. BMS File Audit Ingestion (CSV/Excel -> waste in AED, primary diagnosis, plot, download)

PHYSICS (no water flow meter anywhere)
  W_elec  = measured kW, or sqrt(3)*V*I*PF from compressor current
  dh      = h_discharge - h_suction_vapour               (measured discharge temperature)
  m_ref   = W_elec / dh ;  TR_est = m_ref*(h_suction_vap - h_liquid)/3.517 ;  kW/TR = W_elec/TR_est
  T_approach(cond) = T_sat,cond(P_cond) - T_cond,out
  T_approach(evap) = T_evap,out - T_sat,evap(P_evap)
  T_baseline(t)    = 2.0 + 0.03*(T_amb - 25)    (ASHRAE Guideline 14 style weather normalisation;
                                                 water-cooled lab files without T_amb use T_cond_in)

FAULT SIGNATURES (file-specific baselines calibrated on the first N % of operating records)
  Type A  Condenser fouling : cond-approach residual >= +1.0 C while under load (Compressor_A > 5 A)
  Type B  Refrigerant leak  : evap-approach rise >= +1.5 C over its OWN baseline AND cond-approach
          residual <= -0.5 C AND kW/TR residual > +10 %        (B overrides A)
"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
KW_PER_TR = 3.517
SAT_A, SAT_B = 15.428, 2662.6          # R-134a vapour-pressure fit: ln(P_kPa) = A - B/T_K
CP_VAP_ISENT, K_ISENT = 0.95, 1.09     # suction-vapour cp [kJ/kg.K] and isentropic exponent
CP_SUPERHEAT = 1.15                    # kJ/kg.K, superheated vapour near discharge
ETA_IS_NOM = 0.80                      # healthy isentropic efficiency (baseline twin)
EVAP_APPROACH_NOM = 1.5                # K, default healthy T_evap_out - T_sat_evap (before calibration)
POWER_FACTOR = 0.90
SIM_SUPPLY_V, SIM_RATED_TR = 3300.0, 2500      # simulated MV district-cooling machine

HOURS, TRAIN_H, FAULT_H = 21 * 24, 14 * 24, 5 * 24
WARN_MARGIN_C, HARD_TRIP_C = 1.5, 5.5
PERSIST_WINDOW, PERSIST_HITS = 5, 3
DEWA_TARIFF, SEED = 0.44, 42
# Bump whenever the pipeline changes: it is part of every cache key, because st.cache_data does NOT
# notice edits to the functions a cached function calls (stale results survive hot-reloads/redeploys).
APP_VERSION = "2026.11-v7"
SMOOTH_N = 5                           # rolling-median window for raw sensor glitch removal
MIN_LOAD_AMPS = 5.0                    # "operating under load" threshold
STEADY_WINDOW_MIN = 5.0                # steady-state detector window [min] (fast-sampled logs only)
STEADY_REL_STD = 0.04                  # max relative std of heat rejection inside that window

# Fault-signature thresholds (all relative to file-specific calibrated baselines)
A_COND_RESID_MIN = 1.0     # C above weather-adjusted baseline
B_EVAP_RESID_MIN = 1.5     # C above the evaporator approach's own reference baseline
B_COND_RESID_MAX = -0.5    # C (condenser approach drops below baseline)
B_KW_MIN_PCT = 10.0        # % kW/TR residual

CLASSES = ["SL0_Healthy", "CF_Fouling", "UC_Leak"]
LABEL_OF = {"": "SL0_Healthy", "A": "CF_Fouling", "B": "UC_Leak"}
DIAG_NAME = {"A": "Condenser Tube Fouling (Type A)", "B": "Refrigerant Leak / Undercharge (Type B)"}
SCENARIOS = ["Normal Operation", "Gradual Tube Fouling",
             "Sudden Refrigerant Leak/Pressure Loss", "Extreme Weather Spike"]


# --------------------------------------------------------------------------- #
# R-134a property helpers
# --------------------------------------------------------------------------- #
def p_sat(t_c):
    """Saturation pressure [kPa abs] from saturation temperature [C]."""
    return np.exp(SAT_A - SAT_B / (np.asarray(t_c, dtype=float) + 273.15))


def t_sat(p_kpa):
    """Saturation temperature [C] from pressure [kPa abs]."""
    return SAT_B / (SAT_A - np.log(np.asarray(p_kpa, dtype=float))) - 273.15


def h_liq(t_c):
    """Saturated-liquid enthalpy [kJ/kg] (IIR reference)."""
    return 200.0 + 1.34 * t_c + 0.00175 * t_c ** 2


def h_vap(t_c):
    """Saturated-vapour enthalpy [kJ/kg]."""
    return 398.6 + 0.6 * t_c - 0.002 * t_c ** 2


def dh_isentropic(tsc, tse):
    """Isentropic enthalpy rise [kJ/kg] = cp*T1*(PR^((k-1)/k) - 1), PR = P_cond/P_evap."""
    pr = p_sat(tsc) / p_sat(tse)
    return CP_VAP_ISENT * (np.asarray(tse, dtype=float) + 273.15) * (pr ** ((K_ISENT - 1) / K_ISENT) - 1)


def ideal_kw_per_tr(tsc, tse, eta):
    """Specific power [kW/TR] for isentropic efficiency `eta`: 3.517*(dh_is/eta)/(h1 - h4)."""
    return KW_PER_TR * (dh_isentropic(tsc, tse) / eta) / (h_vap(tse) - h_liq(tsc))


# --------------------------------------------------------------------------- #
# Plant simulator (raw controller points only)
# --------------------------------------------------------------------------- #
def _plant_response(t_amb, load, foul, leak, rng, rated_tr=SIM_RATED_TR, supply_v=SIM_SUPPLY_V):
    """Physics response of the plant. foul = extra condenser approach [K]; leak = severity 0..1
    (lowers evaporator saturation temp, cond approach and compressor efficiency)."""
    n = len(t_amb)
    t_cond_in = 14.0 + 0.35 * t_amb + 4.0 + rng.normal(0, 0.25, n)        # tower: wet-bulb + 4 K
    t_cond_out = t_cond_in + 6.2 * load + rng.normal(0, 0.10, n)
    t_evap_out = 6.0 + rng.normal(0, 0.10, n)
    approach = 2.0 + 0.03 * (t_amb - 25) + rng.normal(0, 0.12, n) + foul - 1.6 * leak
    tsc = t_cond_out + approach
    tse = t_evap_out - EVAP_APPROACH_NOM - 3.0 * leak + rng.normal(0, 0.05, n)
    eta = ETA_IS_NOM + rng.normal(0, 0.004, n) - 0.11 * leak
    dh_act = dh_isentropic(tsc, tse) / eta
    m_ref = load * rated_tr * KW_PER_TR / (h_vap(tse) - h_liq(tsc))
    w_kw = m_ref * dh_act
    amps = w_kw * 1000 / (np.sqrt(3) * supply_v * POWER_FACTOR) * (1 + rng.normal(0, 0.004, n))
    t_dis = tsc + (h_vap(tse) + dh_act - h_vap(tsc)) / CP_SUPERHEAT + rng.normal(0, 0.15, n)
    return pd.DataFrame({
        "T_amb": t_amb, "T_cond_in": t_cond_in, "T_cond_out": t_cond_out, "T_evap_out": t_evap_out,
        "P_cond_kPa": p_sat(tsc) * (1 + rng.normal(0, 0.004, n)),
        "P_evap_kPa": p_sat(tse) * (1 + rng.normal(0, 0.004, n)),
        "T_discharge": t_dis, "Compressor_A": amps})


@st.cache_data(show_spinner=False)
def simulate_telemetry(scenario: str, amb_lo: float, amb_hi: float, seed: int = SEED,
                       version: str = APP_VERSION) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = HOURS
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq="h")
    i, hour = np.arange(n), idx.hour.values
    f0 = n - FAULT_H
    mid, amp = (amb_lo + amb_hi) / 2, (amb_hi - amb_lo) / 2
    t_amb = (mid + amp * np.sin(2 * np.pi * (hour - 9) / 24)
             + 0.8 * np.sin(2 * np.pi * i / (24 * 6)) + rng.normal(0, 0.3, n))
    if scenario == "Extreme Weather Spike":                              # heat wave, last 4 days
        t_amb = np.minimum(t_amb + 7.0 * np.clip((i - (n - 96)) / 12, 0, 1), 50.0)
    load = np.clip(0.45 + 0.015 * (t_amb - 20) + rng.normal(0, 0.04, n), 0.35, 1.0)
    foul = 4.0 * np.clip((i - f0) / FAULT_H, 0, 1) if scenario == "Gradual Tube Fouling" else np.zeros(n)
    leak = (i >= f0 + 12).astype(float) if scenario == "Sudden Refrigerant Leak/Pressure Loss" else np.zeros(n)
    out = _plant_response(t_amb, load, foul, leak, rng)
    out.index = pd.Index(idx, name="timestamp")
    return out


def to_rp1043_raw(d: pd.DataFrame, power: str = "Amps", include_amb: bool = True) -> pd.DataFrame:
    """Export canonical metric data as raw BMS / RP-1043 style headers and imperial units."""
    c2f = lambda c: c * 9 / 5 + 32
    kpa2psig = lambda p: p / 6.89476 - 14.696
    out = pd.DataFrame(index=d.index)
    out["TCI"], out["TCO"], out["TEO"] = c2f(d["T_cond_in"]), c2f(d["T_cond_out"]), c2f(d["T_evap_out"])
    out["PRC"], out["PRE"] = kpa2psig(d["P_cond_kPa"]), kpa2psig(d["P_evap_kPa"])
    out["TR_dis"] = c2f(d["T_discharge"])
    if power == "kW":
        out["kW"] = np.sqrt(3) * SIM_SUPPLY_V * d["Compressor_A"] * POWER_FACTOR / 1000
    else:
        out["Amps"] = d["Compressor_A"]
    if include_amb:
        out["OAT_F"] = c2f(d["T_amb"])
    return out


@st.cache_data(show_spinner=False)
def make_benchmark(n_per_class: int = 250, seed: int = 11, version: str = APP_VERSION) -> pd.DataFrame:
    """RP-1043-style steady-state samples in raw lab units (degF, psig) with ground-truth labels.
    Fouling severity spans 1.0-3.5 K so mild cases legitimately sit below the detection limit."""
    rng = np.random.default_rng(seed)
    n = 3 * n_per_class
    labels = np.repeat(CLASSES, n_per_class)
    t_amb = rng.uniform(26, 46, n)
    load = rng.uniform(0.4, 1.0, n)
    foul = np.where(labels == "CF_Fouling", rng.uniform(1.0, 3.5, n), 0.0)
    leak = np.where(labels == "UC_Leak", rng.uniform(0.6, 1.0, n), 0.0)
    raw = to_rp1043_raw(_plant_response(t_amb, load, foul, leak, rng), power="kW", include_amb=False)
    raw["Amps"] = 100.0                                   # steady-state benchmark: machine running
    raw["Label"] = labels
    return raw.sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Telemetry normalisation
# --------------------------------------------------------------------------- #
_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date", "datetimestamp", "timeminutes", "timemin", "timesec", "timeseconds"],
    "T_cond_in": ["tci", "tcondin", "tcondinc", "condenterwatertemp", "condin"],
    "T_cond_out": ["tco", "tcondout", "tcondoutc", "condleavingwatertemp", "condout"],
    "T_evap_in": ["tei", "tevapin", "tevapinc", "evapin"],
    "T_evap_out": ["teo", "tevapout", "tevapoutc", "evapout"],
    "P_cond_kPa": ["prc", "pcond", "pcondkpa", "pcondpsig", "condpressure"],
    "P_evap_kPa": ["pre", "pevap", "pevapkpa", "pevappsig", "evappressure"],
    "T_discharge": ["trdis", "tdis", "tdischarge", "tdisc", "dischargetemp", "tdischargec"],
    "kW": ["kw", "kwpower", "power", "compressorpower", "kwcomp"],
    "Compressor_A": ["amps", "amp", "current", "compressora", "compressorcurrent"],
    "T_amb": ["tamb", "oat", "oatc", "oatf", "outdoortemp", "outsidetemp", "ambienttemp", "ambient"],
    "TR_measured": ["evaptons"],                           # flow-metered lab tonnage (validation only)
    "label": ["label", "faultlabel", "fault", "faulttype", "class", "condition"],
}
_ALIAS_LOOKUP = {alias: canon for canon, lst in _ALIASES.items() for alias in lst}
_TEMP_COLS = ["T_cond_in", "T_cond_out", "T_evap_in", "T_evap_out", "T_discharge", "T_amb"]
_RP_KEYS = {"tci", "tco", "tei", "teo", "prc", "pre"}
UNIT_MODES = ["Auto-detect", "°F + psig (RP-1043)", "°C + kPa abs"]


def _key(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_supply_voltage(amps_max: float) -> float:
    """460 V for commercial/lab chillers (<150 A max), 3300 V for medium-voltage district plants."""
    return 460.0 if amps_max < 150 else 3300.0


def _parse_time_axis(d: pd.DataFrame, samples_per_hour: float, sec_per_unit: float = 1.0):
    """Return (frame indexed by time, kind). kind: 'datetime' | 'elapsed' | 'rows' | 'none'.
    Numeric time columns are RELATIVE SECONDS (never epoch timestamps): index = elapsed hours.
    Integer counters stepping by 1 are row counts: index = row / samples_per_hour."""
    if "timestamp" in d.columns:
        ts = d["timestamp"]
        num = pd.to_numeric(ts, errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(ts):
            d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            return d, "datetime"
        if num.notna().mean() >= 0.9:                                    # numeric time column
            d = d.assign(_t=num).dropna(subset=["_t"]).sort_values("_t")
            t = d["_t"].to_numpy(float)
            if np.nanmedian(t) > 1e9:                                    # genuine epoch seconds
                d.index = pd.to_datetime(t, unit="s")
                return d.drop(columns=["timestamp", "_t"]), "datetime"
            is_counter = len(t) > 2 and np.allclose(t % 1, 0) and np.median(np.diff(t)) == 1
            if is_counter:                                               # sequential row numbers
                hours = np.arange(len(t)) / samples_per_hour
                kind = "rows"
            else:                                                        # relative elapsed seconds
                hours = (t - t.min()) * sec_per_unit / 3600.0
                kind = "elapsed"
            d.index = pd.Index(hours, name="elapsed_h")
            return d.drop(columns=["timestamp", "_t"]), kind
        d["timestamp"] = pd.to_datetime(ts, errors="coerce")
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
        return d, "datetime"
    if isinstance(d.index, pd.DatetimeIndex):
        return d, "datetime"
    d = d.copy()
    d.index = pd.Index(np.arange(len(d)) / samples_per_hour, name="elapsed_h")
    return d, "none"


def normalize_bms_columns(raw: pd.DataFrame, unit_mode: str = UNIT_MODES[0], samples_per_hour: float = 60.0):
    """Map RP-1043 / BMS headers to canonical names, convert units to degC and kPa abs.
       P_kPa_abs = (P_psig + 14.696) * 6.89476.     Returns (dataframe, meta)."""
    d = raw.copy()
    keys = {c: _key(c) for c in d.columns}
    rp_style = bool(_RP_KEYS & set(keys.values()))
    kpa_native = {_ALIAS_LOOKUP[k] for c, k in keys.items() if k in _ALIAS_LOOKUP and "kpa" in k}
    d = d.rename(columns={c: _ALIAS_LOOKUP[k] for c, k in keys.items() if k in _ALIAS_LOOKUP})
    d = d.loc[:, ~d.columns.duplicated()]
    keep = [c for c in set(_ALIASES) if c in d.columns]                  # drop unrelated lab columns
    d = d[[c for c in d.columns if c in keep]]

    num_cols = [c for c in d.columns if c not in ("timestamp", "label")]
    d[num_cols] = d[num_cols].apply(pd.to_numeric, errors="coerce")

    # --- unit decisions
    if unit_mode == UNIT_MODES[1]:
        temp_f = press_psig = True
    elif unit_mode == UNIT_MODES[2]:
        temp_f = press_psig = False
    else:
        temp_f = rp_style or ("T_cond_out" in d and d["T_cond_out"].median() > 60)
        press_psig = rp_style or ("P_cond_kPa" in d and "P_cond_kPa" not in kpa_native
                                  and d["P_cond_kPa"].median() < 400)
    if temp_f:
        for c in _TEMP_COLS:
            if c in d:
                d[c] = (d[c] - 32.0) * 5.0 / 9.0
    if press_psig:
        for c in ("P_cond_kPa", "P_evap_kPa"):
            if c in d:
                d[c] = (d[c] + 14.696) * 6.89476

    # --- supply voltage (only needed when power comes from current); ignore absurd glitches
    has_amps = "Compressor_A" in d and d["Compressor_A"].notna().any()
    supply_v = detect_supply_voltage(float(d["Compressor_A"].quantile(0.99))) if has_amps else None

    # --- ambient: measured value, else entering condenser water temperature (water-cooled files)
    amb_proxy = "T_amb" not in d or d["T_amb"].isna().all()
    if amb_proxy:
        d["T_amb"] = d["T_cond_in"] if "T_cond_in" in d else 25.0

    sec_per_unit = 60.0 if any(k in ("timeminutes", "timemin") for k in keys.values()) else 1.0
    d, kind = _parse_time_axis(d, samples_per_hour, sec_per_unit)
    meta = {"temp": "°F → °C" if temp_f else "°C (no conversion)",
            "press": "psig → kPa abs" if press_psig else "kPa abs (no conversion)",
            "supply_v": supply_v, "amb_proxy": amb_proxy, "time_kind": kind,
            "samples_per_hour": samples_per_hour, "has_tdis": "T_discharge" in d.columns}
    return d, meta


def missing_required(d: pd.DataFrame):
    need = ["T_cond_out", "T_evap_out", "P_cond_kPa", "P_evap_kPa"]
    miss = [c for c in need if c not in d.columns]
    if "kW" not in d.columns and "Compressor_A" not in d.columns:
        miss.append("kW or Compressor current (Amps)")
    return miss


# --------------------------------------------------------------------------- #
# Physics engine
# --------------------------------------------------------------------------- #
def finalize_residuals(d: pd.DataFrame) -> pd.DataFrame:
    d["Approach_Resid"] = d["T_approach"] - d["T_baseline"]
    d["Evap_Resid"] = d["T_evap_approach"] - d["Evap_Base"]
    d["kWTR_Resid_pct"] = (d["kW_per_TR"] / d["kWTR_baseline"] - 1) * 100
    d["Delta_kW"] = (d["kW_per_TR"] - d["kWTR_baseline"]) * d["TR_est"]     # wasted power
    return d


def physics_engine(d: pd.DataFrame, supply_v: float = SIM_SUPPLY_V, smooth_n: int = 1,
                   drive_eff: float = 1.0) -> pd.DataFrame:
    """smooth_n > 1 applies a rolling median to raw sensors (time-ordered logs only).
    drive_eff = shaft/electrical power ratio (1.0 hermetic; ~0.8-0.9 for open-drive motor + coupling)."""
    out = d.copy()
    # sensor sanity: impossible pressures / power glitches become NaN (rows dropped below)
    for c in ("P_cond_kPa", "P_evap_kPa"):
        out.loc[~out[c].between(50, 4000), c] = np.nan
    if "kW" in out:
        ref = out.loc[out["kW"] > 0, "kW"].median()
        out.loc[(out["kW"] <= 0) | (out["kW"] > 20 * ref), "kW"] = np.nan
    if smooth_n > 1:                                          # transient / glitch suppression
        for c in ["P_cond_kPa", "P_evap_kPa", "T_cond_out", "T_evap_out", "T_discharge", "kW", "Compressor_A"]:
            if c in out:
                out[c] = out[c].rolling(smooth_n, min_periods=1).median()

    out["T_sat_cond"], out["T_sat_evap"] = t_sat(out["P_cond_kPa"]), t_sat(out["P_evap_kPa"])
    if "kW" not in out.columns:                               # P = sqrt(3) V I PF
        out["kW"] = np.sqrt(3) * supply_v * out["Compressor_A"] * POWER_FACTOR / 1000

    h1, h4 = h_vap(out["T_sat_evap"]), h_liq(out["T_sat_cond"])
    if "T_discharge" in out.columns:                          # measured discharge state
        h2 = h_vap(out["T_sat_cond"]) + CP_SUPERHEAT * (out["T_discharge"] - out["T_sat_cond"])
    else:                                                     # fallback: nominal efficiency
        h2 = h1 + dh_isentropic(out["T_sat_cond"], out["T_sat_evap"]) / ETA_IS_NOM
    out["dh_actual"] = h2 - h1
    ok = (out["dh_actual"] > 0) & ((h1 - h4) > 0) & (out["kW"] > 0)
    out["m_ref_kg_s"] = out["kW"] * drive_eff / out["dh_actual"]            # shaft work = refrigerant enthalpy rise
    out["TR_est"] = np.where(ok, out["m_ref_kg_s"] * (h1 - h4) / KW_PER_TR, np.nan)
    out = out.dropna(subset=["TR_est"]).copy()
    out["kW_per_TR"] = out["kW"] / out["TR_est"]
    out["Q_cond_kW"] = out["TR_est"] * KW_PER_TR + out["kW"] * drive_eff   # heat rejected = Q_evap + shaft work

    out["T_approach"] = out["T_sat_cond"] - out["T_cond_out"]             # condenser approach
    out["T_evap_approach"] = out["T_evap_out"] - out["T_sat_evap"]        # evaporator approach
    if smooth_n > 1:
        for c in ("T_approach", "T_evap_approach"):
            out[c] = out[c].rolling(smooth_n, min_periods=1).median()

    # "operating under load": compressor current > 5 A, else power above 10 % of its 90th percentile
    if "Compressor_A" in out.columns and out["Compressor_A"].notna().any():
        out["Under_Load"] = (out["Compressor_A"] > MIN_LOAD_AMPS).to_numpy()
    else:
        out["Under_Load"] = (out["kW"] > max(1.0, 0.1 * out["kW"].quantile(0.9))).to_numpy()

    out["Steady"] = True                                  # refined by apply_steady_gate() for fast logs
    out["Eligible"] = out["Under_Load"]                   # loaded (and, for fast logs, steady) rows
    out["Evaluable"] = out["Under_Load"]                  # eligible AND inside the baseline's reference envelope
    out["In_Envelope"] = True                             # refined by apply_baseline()
    out["T_baseline"] = 2.0 + 0.03 * (out["T_amb"] - 25.0)                # ASHRAE G14 weather curve
    out["T_warn"] = out["T_baseline"] + WARN_MARGIN_C
    out["Evap_Base"] = EVAP_APPROACH_NOM
    out["kWTR_baseline"] = ideal_kw_per_tr(out["T_cond_out"] + out["T_baseline"],
                                           out["T_evap_out"] - out["Evap_Base"], ETA_IS_NOM)
    return finalize_residuals(out)


def _robust_lstsq(X, y):
    """Trimmed least squares on a design matrix: fit, drop points beyond 2.5 robust sigma, refit. Keeps
    transients and faults inside the reference data from bending the healthy baseline."""
    keep = np.ones(len(y), dtype=bool)
    for _ in range(3):
        b = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        r = y - X @ b
        sig = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
        new_keep = np.abs(r) <= 2.5 * sig
        if new_keep.sum() < X.shape[1] + 8:
            break
        keep = new_keep
    return np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]


def _rng(v, margin=0.10):
    lo, hi = float(np.quantile(v, 0.02)), float(np.quantile(v, 0.98))
    pad = margin * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _design(p, q, teo, tco, quad):
    """Operating-condition features, clipped to the range seen in the reference data (no wild extrapolation)."""
    qs = (np.clip(q, *p["q_rng"]) - p["q_ref"]) / p["q_ref"]
    cols = [np.ones(len(q)), qs] + ([qs ** 2] if quad else [])
    cols += [np.clip(teo, *p["teo_rng"]) - p["teo_ref"], np.clip(tco, *p["tco_rng"]) - p["tco_ref"]]
    return np.column_stack(cols)


def fit_baseline(d: pd.DataFrame, mask, load_adjust: bool = True):
    """Learn a healthy baseline from reference rows (first N % of a log, or a whole healthy file).
       * condenser approach : weather curve + offset + slope * (heat rejection - reference)
       * evaporator approach: linear in heat rejection, leaving-water and condenser-water temperature
       * kW/TR              : first-principles twin x quadratic part-load / operating-condition correction
       Without load_adjust (or with too few rows) each falls back to a constant offset / scale."""
    m = np.asarray(mask, dtype=bool) & d["Eligible"].to_numpy()
    if m.sum() < 5:
        return None
    w0 = (2.0 + 0.03 * (d["T_amb"] - 25.0)).to_numpy()
    q, teo, tco = d["Q_cond_kW"].to_numpy(), d["T_evap_out"].to_numpy(), d["T_cond_out"].to_numpy()
    y = d["T_approach"].to_numpy() - w0
    p = {"off": float(np.median(y[m])), "slope": 0.0, "q_ref": float(np.median(q[m])),
         "teo_ref": float(np.median(teo[m])), "tco_ref": float(np.median(tco[m])),
         "q_rng": _rng(q[m]), "teo_rng": _rng(teo[m]), "tco_rng": _rng(tco[m]),
         "evap_base": float(np.median(d["T_evap_approach"].to_numpy()[m])), "evap_coef": None,
         "kw_scale": 1.0, "kw_coef": None, "n_ref": int(m.sum())}
    if load_adjust and m.sum() >= 30:                                     # load-aware condenser approach
        c = np.polyfit(q[m] - p["q_ref"], y[m], 1)
        keep = np.abs(y[m] - np.polyval(c, q[m] - p["q_ref"])) <= 2.5 * (1.4826 * np.median(
            np.abs(y[m] - np.polyval(c, q[m] - p["q_ref"]) - np.median(y[m] - np.polyval(c, q[m] - p["q_ref"])))) + 1e-9)
        c = np.polyfit((q[m] - p["q_ref"])[keep], y[m][keep], 1)
        p["slope"], p["off"] = float(c[0]), float(c[1])
    tb = w0 + p["off"] + p["slope"] * (q - p["q_ref"])
    evap_base = np.full(len(q), p["evap_base"])
    if load_adjust and m.sum() >= 60:                                     # condition-aware evaporator approach
        p["evap_coef"] = _robust_lstsq(_design(p, q, teo, tco, False)[m], d["T_evap_approach"].to_numpy()[m])
        evap_base = _design(p, q, teo, tco, False) @ p["evap_coef"]
    twin = ideal_kw_per_tr(tco + tb, teo - evap_base, ETA_IS_NOM)
    ratio = d["kW_per_TR"].to_numpy() / twin
    if load_adjust and m.sum() >= 60:
        p["kw_coef"] = _robust_lstsq(_design(p, q, teo, tco, True)[m], ratio[m])
    else:
        p["kw_scale"] = float(np.median(ratio[m]))
    return p


def apply_baseline(d: pd.DataFrame, p, enforce_envelope: bool = True) -> pd.DataFrame:
    out = d.copy()
    if p is None:
        return out
    q, teo, tco = out["Q_cond_kW"].to_numpy(), out["T_evap_out"].to_numpy(), out["T_cond_out"].to_numpy()
    tb = (2.0 + 0.03 * (out["T_amb"] - 25.0)).to_numpy() + p["off"] + p["slope"] * (q - p["q_ref"])
    evap_base = (_design(p, q, teo, tco, False) @ p["evap_coef"]) if p["evap_coef"] is not None else p["evap_base"]
    out["T_baseline"], out["T_warn"], out["Evap_Base"] = tb, tb + WARN_MARGIN_C, evap_base
    twin = ideal_kw_per_tr(tco + tb, teo - evap_base, ETA_IS_NOM)
    if p["kw_coef"] is not None:
        ratio = np.clip(_design(p, q, teo, tco, True) @ p["kw_coef"], 0.5, 3.0)
    else:
        ratio = p["kw_scale"]
    out["kWTR_baseline"] = twin * ratio
    # applicability domain: is this row inside the operating envelope the baseline was learned on?
    out["In_Envelope"] = ((q >= p["q_rng"][0]) & (q <= p["q_rng"][1]) & (teo >= p["teo_rng"][0])
                          & (teo <= p["teo_rng"][1]) & (tco >= p["tco_rng"][0]) & (tco <= p["tco_rng"][1]))
    # no fault verdicts (and no waste accounting) on operating conditions the baseline has never seen
    out["Evaluable"] = out["Eligible"] & (out["In_Envelope"] if enforce_envelope else True)
    return finalize_residuals(out)


def calibrate_baseline(d: pd.DataFrame, mask, load_adjust: bool = False) -> pd.DataFrame:
    return apply_baseline(d, fit_baseline(d, mask, load_adjust), enforce_envelope=False)


def apply_steady_gate(d: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Steady-state detector for fast-sampled logs (<= 5 min): faults are only evaluated when heat
    rejection is stable, because approach and efficiency lag load changes. Slow (e.g. hourly) BMS
    data cannot resolve transients, so every loaded row stays evaluable."""
    out = d.copy()
    out["Steady"] = True
    hrs = ((out.index - out.index[0]).total_seconds().to_numpy(float) / 3600.0
           if meta["time_kind"] == "datetime" else out.index.to_numpy(float))
    dt = np.diff(hrs)
    pos = dt[dt > 0]
    med = float(np.median(pos)) if len(pos) else 1.0 / meta["samples_per_hour"]
    meta["steady_gate_min"] = None
    if med * 60.0 <= 5.0:
        w = max(5, int(round(STEADY_WINDOW_MIN / 60.0 / med)))
        q = out["Q_cond_kW"]
        roll = q.rolling(w, min_periods=max(3, w // 2))
        out["Steady"] = ((roll.std() / roll.mean()) < STEADY_REL_STD).fillna(False).to_numpy()
        meta["steady_gate_min"] = STEADY_WINDOW_MIN
    out["Eligible"] = out["Under_Load"] & out["Steady"]
    out["Evaluable"] = out["Eligible"]
    return out


# --------------------------------------------------------------------------- #
# ML anomaly detection + fault classification
# --------------------------------------------------------------------------- #
def detect_anomalies(d: pd.DataFrame, train_mask, contamination: float = 0.01, train_df=None) -> pd.DataFrame:
    """Hybrid raw flags (Isolation Forest on baseline-normalised residuals OR physics limits),
    then the persistence filter: >= 3 anomaly flags within the last 5 time-steps.
    train_df: healthy reference frame (already baseline-applied); default = first-N% rows of d."""
    out = d.copy()
    feats = ["Approach_Resid", "kWTR_Resid_pct", "Evap_Resid"]
    ev = out["Evaluable"].to_numpy()
    if train_df is not None:
        train = train_df.loc[train_df["Evaluable"].to_numpy(), feats]
    else:
        train = out.loc[np.asarray(train_mask, dtype=bool) & ev, feats]
    rule_flag = ((out["Approach_Resid"] >= A_COND_RESID_MIN)
                 | (out["Evap_Resid"] >= B_EVAP_RESID_MIN)).to_numpy() & ev
    ml_flag = np.zeros(len(out), dtype=bool)
    if len(train) >= 30:
        model = IsolationForest(n_estimators=300, contamination=contamination, random_state=7).fit(train)
        ml_flag = (model.predict(out[feats]) == -1) & ev & out["In_Envelope"].to_numpy()   # no verdicts on unseen conditions
    out["rule_flag"], out["ml_flag"] = rule_flag, ml_flag
    out["raw_anomaly"] = np.where(ml_flag | rule_flag, -1, 1)             # 1 normal, -1 anomaly
    out["anomaly_count"] = (out["raw_anomaly"] == -1).rolling(PERSIST_WINDOW, min_periods=1).sum()
    out["persistent_alert"] = out["anomaly_count"] >= PERSIST_HITS
    return out


def classify_faults(d: pd.DataFrame, smooth: int = 3, gate: bool = True) -> pd.DataFrame:
    """Relative-signature classifier. gate=True also requires the persistence-filtered alert."""
    out = d.copy()
    sm = (lambda s: s.rolling(smooth, min_periods=1).mean()) if smooth > 1 else (lambda s: s)
    cond_r, evap_r, kw_r = sm(out["Approach_Resid"]), sm(out["Evap_Resid"]), sm(out["kWTR_Resid_pct"])
    load = out["Evaluable"]

    # TYPE B: low-side elevation over its own baseline + condenser approach drop + power spike
    type_b = (evap_r >= B_EVAP_RESID_MIN) & (cond_r <= B_COND_RESID_MAX) & (kw_r > B_KW_MIN_PCT) & load
    # TYPE A: condenser approach >= 1.0 C above weather-adjusted baseline under load (B overrides)
    type_a = (cond_r >= A_COND_RESID_MIN) & load & ~type_b
    if gate:
        type_a &= out["persistent_alert"]
        type_b &= out["persistent_alert"]
    out["fault_type"] = np.select([type_b, type_a], ["B", "A"], default="")
    out["fault_label"] = out["fault_type"].map(LABEL_OF)
    return out


def add_financials(d: pd.DataFrame, tariff: float) -> pd.DataFrame:
    """Waste_kW is SIGNED excess power over the healthy baseline on evaluable rows. Clipping each noisy sample
    at zero would manufacture waste on a healthy chiller, so only totals / smoothed series are floored at 0."""
    out = d.copy()
    out["Waste_kW"] = np.where(out["Evaluable"], out["Delta_kW"], 0.0)
    out["Loss_AED_day"] = out["Waste_kW"] * 24 * tariff                    # Delta_kW * 24 h * tariff
    out["Loss_AED_day_s"] = out["Loss_AED_day"].rolling(6, min_periods=1).mean().clip(lower=0)
    return out


@st.cache_data(show_spinner=False)
def run_live_pipeline(scenario: str, amb_lo: float, amb_hi: float, seed: int = SEED,
                      version: str = APP_VERSION) -> pd.DataFrame:
    raw = simulate_telemetry(scenario, amb_lo, amb_hi, seed, version)
    d, meta = normalize_bms_columns(raw, UNIT_MODES[2])
    d = physics_engine(d, meta["supply_v"] or SIM_SUPPLY_V)
    d = detect_anomalies(d, np.arange(len(d)) < TRAIN_H)
    return classify_faults(d, smooth=3, gate=True)


# --------------------------------------------------------------------------- #
# Insight / status
# --------------------------------------------------------------------------- #
def operational_status(d: pd.DataFrame):
    last = d.iloc[-1]
    if last["fault_type"] == "A":
        return "Fouling Alert", "#e8590c"
    if last["fault_type"] == "B":
        return "Leak Alert", "#c2255c"
    if last["persistent_alert"]:
        return "Watch", "#f08c00"
    return "Normal", "#2b9348"


def build_insight(d: pd.DataFrame):
    last, tail = d.iloc[-1], d.tail(3)
    loss = max(0.0, float(d["Loss_AED_day"].tail(6).mean()))
    cr, er, kr = tail["Approach_Resid"].mean(), tail["Evap_Resid"].mean(), tail["kWTR_Resid_pct"].mean()
    if last["fault_type"] == "A":
        m = min(len(d), 48)
        slope = float(np.polyfit(np.arange(m), d["T_approach"].tail(m).values, 1)[0] * 24)
        days = 14 if slope <= 0.05 else int(np.clip(np.floor((HARD_TRIP_C - tail["T_approach"].mean()) / slope), 1, 14))
        return ("error" if tail["T_approach"].mean() >= HARD_TRIP_C - 1 else "warning"), (
            f"TYPE A - Condenser tube fouling: condenser approach {cr:+.1f} °C above the weather-adjusted baseline "
            f"(evaporator approach {er:+.1f} °C vs its reference, kW/TR {kr:+.1f}%). "
            f"Schedule tube descaling within {days} days to avoid the {HARD_TRIP_C} °C trip and ~AED {loss:,.0f}/day waste.")
    if last["fault_type"] == "B":
        return "error", (
            f"TYPE B - Refrigerant leak / undercharge: evaporator approach {er:+.1f} °C above its reference with "
            f"condenser approach {cr:+.1f} °C vs baseline and kW/TR {kr:+.1f}%. Do NOT descale: check charge/subcooling, "
            f"leak-test joints and seals. Current waste ~AED {loss:,.0f}/day.")
    if last["persistent_alert"]:
        return "info", "Early drift flagged by the model; signature does not match a known fault yet. Keep monitoring."
    return "success", (f"Within weather-adjusted baseline (ambient {last['T_amb']:.0f} °C → expected approach "
                       f"{last['T_baseline']:.2f} °C). Outdoor heat alone does not raise alerts.")


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1200px}
.kpi{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:14px 16px;
     background:rgba(128,128,128,.07);margin-bottom:10px}
.kpi .t{font-size:.74rem;letter-spacing:.05em;text-transform:uppercase;opacity:.72}
.kpi .v{font-size:1.75rem;font-weight:700;line-height:1.2}
.kpi .s{font-size:.82rem;opacity:.78}
@media (max-width:640px){
  .kpi .v{font-size:1.45rem}
  h1{font-size:1.45rem !important}
  .block-container{padding-left:.8rem;padding-right:.8rem}
}
</style>
"""


def kpi(title, value, sub, color):
    return (f"<div class='kpi' style='border-left:6px solid {color}'><div class='t'>{title}</div>"
            f"<div class='v' style='color:{color}'>{value}</div><div class='s'>{sub}</div></div>")


def style_fig(fig, title, ytitle, relative=False):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=360, hovermode="x unified",
                      margin=dict(l=8, r=8, t=45, b=8), font=dict(size=12), yaxis_title=ytitle,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    if relative:                                         # relative operational hours, never epoch dates
        fig.update_xaxes(title_text="Operational time", tickprefix="Hour ", rangemode="tozero")
    return fig


def chart_approach(d, train_n=None, relative=False):
    pts = d[d["persistent_alert"]]                         # red markers ONLY for persistent alerts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["T_approach"], name="T_approach (5-pt median)", mode="lines",
                             line=dict(color="#1f77b4", width=1.5)))
    fig.add_trace(go.Scatter(x=d.index, y=d["T_baseline"], name="Weather-adjusted baseline", mode="lines",
                             line=dict(color="#2b9348", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=d.index, y=d["T_warn"], name="Warning (baseline + 1.5 °C)", mode="lines",
                             line=dict(color="#f08c00", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=pts.index, y=pts["T_approach"], name="Persistent anomaly", mode="markers",
                             marker=dict(color="red", size=6)))
    fig.add_hline(y=HARD_TRIP_C, line_dash="dot", line_color="#d62728")
    if train_n:
        fig.add_vrect(x0=d.index[0], x1=d.index[min(train_n, len(d)) - 1], fillcolor="green",
                      opacity=0.05, line_width=0)
    return style_fig(fig, "Condenser Approach vs Weather-Adjusted Baseline", "°C", relative)


def chart_loss(d):
    fig = go.Figure(go.Scatter(x=d.index, y=d["Loss_AED_day_s"], mode="lines", fill="tozeroy",
                               name="Wasted energy (AED/day)", line=dict(color="#e8590c", width=1.5)))
    return style_fig(fig, "Estimated Daily Financial Loss (6-sample avg)", "AED / day")


def read_table(file, sheet=None) -> pd.DataFrame:
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file, sheet_name=sheet if sheet is not None else 0)
    return pd.read_csv(file)


def pick_sheet(file, key):
    """Excel workbooks (e.g. RP-1043) hold several sheets; default to the first (full data)."""
    if not file.name.lower().endswith((".xlsx", ".xls")):
        return None
    try:
        sheets = pd.ExcelFile(file).sheet_names
    except Exception:
        return 0
    return st.selectbox("Worksheet", sheets, index=0, key=key) if len(sheets) > 1 else sheets[0]


def meta_caption(meta):
    v = f"{meta['supply_v']:.0f} V" if meta["supply_v"] else "n/a (kW supplied directly)"
    notes = [f"temperatures {meta['temp']}", f"pressures {meta['press']}", f"supply {v}"]
    if meta["amb_proxy"]:
        notes.append("ambient reference = entering condenser water (T_cond_in)")
    if not meta["has_tdis"]:
        notes.append("no discharge temperature → nominal-efficiency enthalpy rise")
    tk = meta["time_kind"]
    notes.append({"datetime": "time axis: timestamps", "elapsed": "time axis: relative seconds → hours",
                  "rows": f"time axis: row counter @ {meta['samples_per_hour']:g}/h",
                  "none": f"no time column: {meta['samples_per_hour']:g} samples/h assumed"}[tk])
    if meta.get("steady_gate_min"):
        notes.append(f"steady-state gate: {meta['steady_gate_min']:g}-min window")
    st.caption("Normalisation → " + " · ".join(notes))


# --------------------------------------------------------------------------- #
# Tab 1 - live simulator
# --------------------------------------------------------------------------- #
def tab_live(amb_lo, amb_hi, scenario, tariff):
    banner = st.empty()
    d = run_live_pipeline(scenario, float(amb_lo), float(amb_hi), SEED, APP_VERSION)
    if "Under_Load" not in d.columns:                      # stale cache from an older deployment: self-heal once
        if not st.session_state.get("_cache_cleared"):
            st.session_state["_cache_cleared"] = True
            st.cache_data.clear()
            st.rerun()
        st.error("Cached results are from an older app version. Use menu → Clear cache, then reload.")
        return
    d = add_financials(d, tariff)
    last = d.iloc[-1]
    if last["T_approach"] >= HARD_TRIP_C:                                  # independent of the ML filter
        with banner.container():
            st.error("⚠️ CRITICAL HARD THRESHOLD BREACH: Immediate Physical Inspection Required.")
            st.caption(f"T_approach = {last['T_approach']:.1f} °C ≥ {HARD_TRIP_C} °C ({d.index[-1]:%Y-%m-%d %H:%M}).")

    status, scolor = operational_status(d)
    app_now, base_now, warn_now = d["T_approach"].tail(3).mean(), last["T_baseline"], last["T_warn"]
    kwtr_now, kwtr_base = d["kW_per_TR"].tail(3).mean(), d["kWTR_baseline"].tail(3).mean()
    loss_now = max(0.0, float(d["Loss_AED_day"].tail(6).mean()))
    wasted_kw = max(0.0, float(d["Waste_kW"].tail(6).mean()))

    c1, c2 = st.columns(2)
    c1.markdown(kpi("T_approach vs baseline", f"{app_now:.1f} °C",
                    f"Baseline {base_now:.1f} · Warning {warn_now:.1f} · Δ {app_now - base_now:+.1f} °C",
                    "#d62728" if app_now >= warn_now else "#2b9348"), unsafe_allow_html=True)
    c2.markdown(kpi("Current kW/TR", f"{kwtr_now:.3f}", f"Baseline {kwtr_base:.3f} · {(kwtr_now / kwtr_base - 1) * 100:+.1f}%",
                    "#e8590c" if kwtr_now / kwtr_base > 1.04 else "#2b9348"), unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    c3.markdown(kpi("Operational status", status,
                    f"Ambient {last['T_amb']:.0f} °C · persistent alerts (24 h): {int(d['persistent_alert'].tail(24).sum())}",
                    scolor), unsafe_allow_html=True)
    c4.markdown(kpi("Daily financial waste", f"AED {loss_now:,.0f}/day",
                    f"{wasted_kw:,.0f} kW wasted · {tariff:.2f} AED/kWh",
                    "#e8590c" if loss_now > 100 else "#2b9348"), unsafe_allow_html=True)

    level, msg = build_insight(d)
    getattr(st, level)(msg)
    t = d.tail(3)
    st.caption(f"Signature → condenser resid {t['Approach_Resid'].mean():+.1f} °C · evaporator approach "
               f"{t['Evap_Resid'].mean():+.1f} °C vs reference · kW/TR {t['kWTR_Resid_pct'].mean():+.1f}% · "
               f"est. load {last['TR_est']:,.0f} TR (current + enthalpy balance, no flow meter)")
    cfg = {"displayModeBar": False}
    st.plotly_chart(chart_approach(d, TRAIN_H), config=cfg)
    st.plotly_chart(chart_loss(d), config=cfg)


# --------------------------------------------------------------------------- #
# Tab 2 - benchmark validation
# --------------------------------------------------------------------------- #
def map_label(v):
    k = _key(v)
    if "sl0" in k or "healthy" in k or "normal" in k:
        return "SL0_Healthy"
    if "cf" in k or "foul" in k:
        return "CF_Fouling"
    if "uc" in k or "leak" in k or "undercharge" in k:
        return "UC_Leak"
    return None


def evaluate_benchmark(raw, unit_mode, calibrate):
    d, meta = normalize_bms_columns(raw, unit_mode)
    if "label" not in d.columns:
        return None, meta, "No ground-truth label column found (expected e.g. 'Label' with SL0_Healthy / CF_Fouling / UC_Leak)."
    miss = missing_required(d)
    if miss:
        return None, meta, f"Missing required columns after normalisation: {', '.join(miss)}"
    d["truth"] = d["label"].map(map_label)
    excluded = int(d["truth"].isna().sum())
    d = d.dropna(subset=["truth", "T_cond_out", "T_evap_out", "P_cond_kPa", "P_evap_kPa"])
    d = physics_engine(d, meta["supply_v"] or SIM_SUPPLY_V)               # shuffled samples: no smoothing
    if calibrate:
        d = calibrate_baseline(d, (d["truth"] == "SL0_Healthy").to_numpy())
    d = classify_faults(d, smooth=1, gate=False)                         # steady-state samples: no persistence
    meta["excluded"] = excluded
    return d, meta, None


def tab_benchmark(unit_mode):
    st.markdown("Evaluates the fault-signature classifier (physics rules only) against ground-truth labels.")
    src = st.radio("Data source", ["Built-in RP-1043-style benchmark (simulated)", "Upload labelled RP-1043 / BMS file"],
                   horizontal=True)
    if src.startswith("Built-in"):
        raw = make_benchmark(250, 11, APP_VERSION)
        st.caption("750 steady-state samples in raw lab units (°F, psig, TCI/TCO/TEO/PRC/PRE headers); "
                   "fouling severity 1.0-3.5 °C so mild cases fall below the detection limit by design.")
        calibrate = st.checkbox("Calibrate baseline on healthy-labelled rows", value=False, key="cal_builtin")
    else:
        up = st.file_uploader("Labelled dataset (CSV / Excel)", type=["csv", "xlsx", "xls"], key="bench_up")
        calibrate = st.checkbox("Calibrate baseline on healthy-labelled rows (recommended for real data)",
                                value=True, key="cal_upload")
        if up is None:
            st.info("Upload a file with RP-1043 headers (TCI, TCO, TEO, PRC, PRE, kW, ...) and a label column.")
            return
        try:
            raw = read_table(up, pick_sheet(up, "bench_sheet"))
        except Exception as e:
            st.error(f"Could not read file (legacy .xls needs `pip install xlrd`): {e}")
            return
    d, meta, err = evaluate_benchmark(raw, unit_mode, calibrate)
    if err:
        st.error(err)
        return
    meta_caption(meta)
    if meta.get("excluded"):
        st.caption(f"{meta['excluded']} rows with labels outside SL0/CF/UC were excluded from scoring.")
    y_true, y_pred = d["truth"], d["fault_label"]
    st.metric("Overall accuracy", f"{accuracy_score(y_true, y_pred) * 100:.1f}%", f"{len(d)} samples")
    rep = pd.DataFrame(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES,
                                             output_dict=True, zero_division=0)).T
    st.subheader("Classification report")
    st.dataframe(rep.round(3))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES),
                      index=[f"True: {c}" for c in CLASSES], columns=[f"Pred: {c}" for c in CLASSES])
    st.subheader("Confusion matrix")
    st.dataframe(cm)


# --------------------------------------------------------------------------- #
# Tab 3 - BMS file audit
# --------------------------------------------------------------------------- #
def prepare_log(raw, unit_mode, samples_per_hour, drive_eff):
    """normalise -> clean -> physics -> steady-state gate. Returns (frame, meta, error)."""
    d, meta = normalize_bms_columns(raw, unit_mode, samples_per_hour)
    miss = missing_required(d)
    if miss:
        return None, meta, f"Missing required columns after normalisation: {', '.join(miss)}"
    d = d.dropna(subset=["T_cond_out", "T_evap_out", "P_cond_kPa", "P_evap_kPa"])
    d = physics_engine(d, meta["supply_v"] or SIM_SUPPLY_V, smooth_n=SMOOTH_N, drive_eff=drive_eff)
    if len(d) < 10:
        return None, meta, "Fewer than 10 valid rows after cleaning."
    return apply_steady_gate(d, meta), meta, None


def _block_means(d, rows=30):
    r = d.loc[d["Evaluable"], "Approach_Resid"].to_numpy()
    nb = len(r) // rows
    return r[:nb * rows].reshape(nb, rows).mean(axis=1) if nb >= 3 else np.array([])


def drift_vs_reference(d, dr):
    """Sustained shift of the condenser-approach residual vs a healthy reference file, using block means
    so that a small but persistent offset (e.g. mild tube fouling) is separable from sample noise.
    Returns (shift [C], z-score) or None."""
    a, b = _block_means(d), _block_means(dr)
    if len(a) < 5 or len(b) < 5:
        return None
    shift = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))) + 1e-9
    return shift, shift / se


def analyze_log(raw, unit_mode, calib_pct, samples_per_hour=60.0, drive_eff=1.0, load_adjust=True, ref_raw=None):
    """Baseline source: the first N % of the log (default) or a separate known-healthy reference file."""
    d, meta, err = prepare_log(raw, unit_mode, samples_per_hour, drive_eff)
    if err:
        return None, meta, err
    n = len(d)
    k = max(int(n * calib_pct / 100), min(10, n))                         # reference window: first N %, min 10 rows
    train_mask = np.arange(n) < k
    meta["ref_external"], meta["drift"] = False, None
    if ref_raw is not None:
        dr, _, er = prepare_log(ref_raw, unit_mode, samples_per_hour, drive_eff)
        if er:
            return None, meta, f"Healthy reference file: {er}"
        params = fit_baseline(dr, np.ones(len(dr), dtype=bool), load_adjust)
        if params is None:
            return None, meta, "Healthy reference file has too few loaded, steady records."
        dr, d = apply_baseline(dr, params), apply_baseline(d, params)
        d = detect_anomalies(d, train_mask, train_df=dr)
        meta.update(ref_external=True, drift=drift_vs_reference(d, dr), train_n=0,
                    ml_used=bool(dr["Evaluable"].sum() >= 30))
    else:
        d = apply_baseline(d, fit_baseline(d, train_mask, load_adjust))
        d = detect_anomalies(d, train_mask)
        meta["ml_used"] = bool((train_mask & d["Evaluable"].to_numpy()).sum() >= 30)
        meta["train_n"] = k
    return classify_faults(d, smooth=3, gate=True), meta, None


def time_span(d, meta):
    """Per-row duration [h] for energy integration and the total log span [h]."""
    kind, n = meta["time_kind"], len(d)
    if kind == "datetime":
        hrs = (d.index - d.index[0]).total_seconds().to_numpy(float) / 3600.0
    else:
        hrs = d.index.to_numpy(float)
    dt = np.diff(hrs, append=np.nan)
    pos = dt[np.isfinite(dt) & (dt > 0)]
    med = float(np.median(pos)) if len(pos) else 1.0 / meta["samples_per_hour"]
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, med)
    dt = np.minimum(dt, 5 * med)                                          # gaps do not inflate energy
    if kind == "elapsed":
        total_h = float(hrs.max() - hrs.min())                            # (max(Time) - min(Time)) / 3600
    elif kind in ("rows", "none"):
        total_h = n / meta["samples_per_hour"]                            # rows / sample rate per hour
    else:
        total_h = float(hrs.max() - hrs.min()) + med
    return dt, total_h


def tab_audit(unit_mode, tariff, drive_eff):
    mode = st.radio("Log source", ["Upload my own file", "Demo: hourly UAE BMS log (tube fouling, °F/psig, amps)"],
                    horizontal=True)
    c1, c2 = st.columns(2)
    calib_pct = c1.slider("Healthy reference window at start of log (%)", 5, 40, 15,
                          help="First N % of operating records: calibrates the baselines and trains the anomaly "
                               "model. Assumed healthy. Ignored when a healthy reference file is supplied.")
    sph = c2.number_input("Samples per hour", 1, 7200, 60,
                          help="Only used when the log has no real time axis (no Time column or a row counter).")
    load_adjust = st.checkbox(
        "Load-aware baselines (recommended)", value=True,
        help="Condenser approach and kW/TR depend on heat flux / load. When on, the reference data also fits "
             "them against estimated heat rejection, so normal load swings are not mistaken for faults.")
    ref_raw = None
    with st.expander("Optional: healthy reference file (use when the log may be faulted from its first row)"):
        ref_up = st.file_uploader("Known-healthy log of the same machine (CSV / Excel)", type=["csv", "xlsx", "xls"],
                                  key="ref_up")
        if ref_up is not None:
            try:
                ref_raw = read_table(ref_up, pick_sheet(ref_up, "ref_sheet"))
            except Exception as e:
                st.error(f"Could not read reference file: {e}")
    if mode.startswith("Demo"):
        raw = to_rp1043_raw(simulate_telemetry("Gradual Tube Fouling", 30.0, 44.0, SEED, APP_VERSION),
                             power="Amps").reset_index()
    else:
        up = st.file_uploader("BMS log (CSV / Excel)", type=["csv", "xlsx", "xls"], key="audit_up")
        if up is None:
            st.info("Upload a BMS export or an ASHRAE RP-1043 file (e.g. cf12.xls / normal.xls) with condenser/"
                    "evaporator temperatures & pressures and compressor kW or current.")
            return
        try:
            raw = read_table(up, pick_sheet(up, "audit_sheet"))
        except Exception as e:
            st.error(f"Could not read file (legacy .xls needs `pip install xlrd`, .xlsx needs `openpyxl`): {e}")
            return

    d, meta, err = analyze_log(raw, unit_mode, calib_pct, float(sph), drive_eff, load_adjust, ref_raw)
    if err:
        st.error(err)
        return
    meta_caption(meta)
    if meta["ref_external"]:
        st.caption("Baselines learned from the healthy reference file (reference window slider not used).")
    if not meta["ml_used"]:
        st.caption("Reference data has <30 evaluable rows: Isolation Forest skipped, physics-limit flags only.")
    d = add_financials(d, tariff)
    cover = float(d["Evaluable"].sum()) / max(float(d["Under_Load"].sum()), 1.0)
    if cover < 0.5:
        st.warning(f"Only {cover * 100:.0f}% of loaded records fall inside the operating envelope (load, water "
                   "temperatures) covered by the reference data, so the rest get no fault verdict. Widen the "
                   "reference window or upload a healthy reference file for fuller coverage.")

    # ---- time span & financials (relative seconds are NOT epoch timestamps)
    dt_h, total_hours = time_span(d, meta)
    total_days = max(total_hours / 24.0, 0.01)
    under, ev = d["Under_Load"].to_numpy(), d["Evaluable"].to_numpy()
    ev_h, under_h = float((dt_h * ev).sum()), float((dt_h * under).sum())
    mean_waste_kw = float((d["Waste_kW"].to_numpy() * dt_h).sum() / ev_h) if ev_h > 0 else 0.0
    waste_kwh = max(mean_waste_kw * under_h, 0.0)                         # net excess energy, floored at 0
    total_aed = waste_kwh * tariff
    per_day = total_aed / total_days                                       # AED/day

    n_ev = max(int(ev.sum()), 1)
    counts = d.loc[d["Evaluable"], "fault_type"].value_counts()
    share = {t: counts.get(t, 0) / n_ev for t in ["A", "B"]}
    drift = meta["drift"]
    if counts.get("A", 0) + counts.get("B", 0) < max(3, 0.01 * n_ev):
        if drift and drift[0] >= 0.15 and drift[1] >= 5:
            diagnosis = (f"Sub-threshold condenser drift: +{drift[0]:.2f} °C vs healthy reference "
                         f"(z = {drift[1]:.0f}) - consistent with early / mild fouling")
            dcolor = "#f08c00"
        else:
            diagnosis, dcolor = "No dominant fault - operating healthy", "#2b9348"
    else:
        main = "A" if counts.get("A", 0) >= counts.get("B", 0) else "B"
        active = " · active at end of log" if (d["fault_type"].tail(24) == main).any() else ""
        diagnosis = f"{DIAG_NAME[main]} - {share[main] * 100:.0f}% of evaluable records{active}"
        dcolor = "#e8590c" if main == "A" else "#c2255c"

    trips = int(((d["T_approach"] >= HARD_TRIP_C) & d["Evaluable"]).sum())
    if trips:
        st.error(f"⚠️ CRITICAL HARD THRESHOLD BREACH: {trips} record(s) with T_approach ≥ {HARD_TRIP_C} °C. "
                 "Immediate Physical Inspection Required.")

    span_txt = f"{total_hours:.1f} h ({total_days:.2f} days)" if total_hours < 48 else f"{total_days:.1f} days"
    k1, k2 = st.columns(2)
    k1.markdown(kpi("Total period waste", f"AED {total_aed:,.0f}", f"{waste_kwh:,.1f} kWh over {span_txt} · {tariff:.2f} AED/kWh",
                    "#e8590c" if total_aed > 0 else "#2b9348"), unsafe_allow_html=True)
    k2.markdown(kpi("Average daily waste", f"AED {per_day:,.0f}/day",
                    f"{len(d):,} records · {int(under.sum()):,} under load · {n_ev / max(int(under.sum()), 1) * 100:.0f}% evaluable",
                    "#e8590c" if per_day > 0 else "#2b9348"), unsafe_allow_html=True)
    st.markdown(kpi("Primary dataset diagnosis", diagnosis,
                    f"Type A {share['A'] * 100:.1f}% · Type B {share['B'] * 100:.1f}% · healthy/unclassified "
                    f"{(1 - share['A'] - share['B']) * 100:.1f}% (of evaluable records)", dcolor), unsafe_allow_html=True)
    if total_hours < 24:
        st.caption(f"Log spans only {total_hours:.1f} h: the AED/day figure extrapolates this period to 24 h.")

    # ---- independent check of the flow-meter-free capacity estimate (lab files with flow-metered tonnage)
    if "TR_measured" in d.columns:
        v = d[d["Under_Load"] & (d["TR_measured"] > 10) & (d["TR_measured"] < 500)]
        if len(v) > 20:
            err_pct = float(np.median(np.abs(v["TR_est"] / v["TR_measured"] - 1)) * 100)
            bias = float(np.median(v["TR_est"] / v["TR_measured"] - 1) * 100)
            st.caption(f"Capacity validation vs flow-metered 'Evap Tons' in this file: median |error| {err_pct:.1f}% "
                       f"(bias {bias:+.1f}%, {len(v):,} samples; drive efficiency {drive_eff:.2f}). "
                       "The app itself never uses the flow data.")

    st.plotly_chart(chart_approach(d, meta["train_n"] if meta["train_n"] >= 10 else None,
                                   relative=meta["time_kind"] != "datetime"), config={"displayModeBar": False})
    cols = ["T_amb", "T_approach", "T_baseline", "T_warn", "T_evap_approach", "Evap_Base", "kW", "TR_est",
            "Q_cond_kW", "kW_per_TR", "kWTR_baseline", "Delta_kW", "Loss_AED_day", "Under_Load", "Evaluable",
            "persistent_alert", "fault_label"]
    out_csv = d[cols].round(4)
    out_csv.index.name = "elapsed_hours" if meta["time_kind"] != "datetime" else "timestamp"
    st.download_button("Download scored dataset (CSV)", out_csv.to_csv().encode(),
                       "chiller_audit_scored.csv", "text/csv")
    with st.expander("Scored data preview"):
        st.dataframe(d[cols].round(3).tail(200))


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Chiller FDD - UAE", page_icon="❄️", layout="wide", initial_sidebar_state="auto")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("❄️ Chiller Predictive Maintenance")
    st.caption("Centrifugal chillers · UAE climate · flow-meter-free enthalpy physics · AED energy loss")

    with st.sidebar:
        st.header("Controls")
        tariff = st.slider("DEWA tariff (AED/kWh)", 0.20, 0.80, DEWA_TARIFF, 0.01)
        st.subheader("Tab 1 simulator")
        amb_lo, amb_hi = st.slider("Outdoor ambient range (°C)", 20, 48, (30, 44))
        scenario = st.radio("Scenario", SCENARIOS, index=1)
        st.subheader("Tabs 2-3 files")
        unit_mode = st.selectbox("Input units", UNIT_MODES)
        drive_eff = st.slider("Drive efficiency (shaft / electrical)", 0.60, 1.00, 1.00, 0.01,
                              help="1.00 = hermetic (motor heat enters the refrigerant). Open-drive machines such as "
                                   "the RP-1043 chiller are typically 0.80-0.90; this scales estimated capacity.")

    t1, t2, t3 = st.tabs(["Live UAE Simulator", "RP-1043 Benchmark", "BMS File Audit"])
    with t1:
        tab_live(amb_lo, amb_hi, scenario, tariff)
    with t2:
        tab_benchmark(unit_mode)
    with t3:
        tab_audit(unit_mode, tariff, drive_eff)


if __name__ == "__main__":
    main()
