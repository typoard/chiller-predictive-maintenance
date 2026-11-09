"""
Centrifugal Chiller Predictive Maintenance Platform  -  UAE edition
===================================================================
Run:   streamlit run app.py
Deps:  pip install streamlit pandas numpy scikit-learn plotly openpyxl     (openpyxl only for .xlsx upload)

TABS
  1. Live UAE Operational Simulator  (Normal / Tube Fouling / Leak / Weather Spike)
  2. ASHRAE RP-1043 Benchmark Validation (classification report + confusion matrix)
  3. BMS File Audit Ingestion (CSV/Excel -> waste in AED, primary diagnosis, plot, download)

PHYSICS (no water flow meter anywhere)
  W_elec  = measured kW, or sqrt(3)*V*I*PF from compressor current
  dh      = h_discharge - h_suction_vapour               (measured discharge temperature)
  m_ref   = W_elec / dh                                  (compressor energy balance)
  TR_est  = m_ref * (h_suction_vap - h_liquid) / 3.517   (refrigerating effect)
  kW/TR   = W_elec / TR_est
  T_approach(cond) = T_sat,cond(P_cond) - T_cond,out
  T_approach(evap) = T_evap,out - T_sat,evap(P_evap)
  T_baseline(t)    = 2.0 + 0.03*(T_amb - 25)    (ASHRAE Guideline 14 style weather normalisation)

DUAL-LOOP FAULT SIGNATURES (high side AND low side evaluated together)
  Type A  Condenser fouling : cond approach resid >= +1.5 C, evap approach <= 2.0 C, kW/TR up >= 4 %
  Type B  Refrigerant leak  : evap approach >= 3.0 C, cond approach resid <= +0.5 C, kW/TR elevated
          (B overrides A whenever its low-side signature is present)
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
EVAP_APPROACH_NOM = 1.5                # K, healthy T_evap_out - T_sat_evap
POWER_FACTOR = 0.90
SIM_SUPPLY_V, SIM_RATED_TR = 3300.0, 2500      # simulated MV district-cooling machine

HOURS, TRAIN_H, FAULT_H = 21 * 24, 14 * 24, 5 * 24
WARN_MARGIN_C, HARD_TRIP_C = 1.5, 5.5
PERSIST_WINDOW, PERSIST_HITS = 5, 3
DEWA_TARIFF, SEED = 0.44, 42

# Fault-signature thresholds
A_COND_RESID_MIN = 1.5     # C above weather baseline
A_EVAP_APPROACH_MAX = 2.0  # C
A_KW_MIN_PCT = 4.0         # % above baseline kW/TR (fouling band is +4..+15 %)
A_KW_CAP_PCT = None        # set to 15.0 to enforce the +15 % cap on Type A literally
B_EVAP_APPROACH_MIN = 3.0  # C
B_COND_RESID_MAX = 0.5     # C
COND_LOW_RESID = -1.0      # C, condenser approach this far below baseline is itself an anomaly flag
KW_ELEVATED_PCT = 4.0      # "elevated" kW/TR residual used by the Type B override (>15 % = severe)

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
def simulate_telemetry(scenario: str, amb_lo: float, amb_hi: float, seed: int = SEED) -> pd.DataFrame:
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
def make_benchmark(n_per_class: int = 250, seed: int = 11) -> pd.DataFrame:
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
    raw["Label"] = labels
    return raw.sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Telemetry normalisation
# --------------------------------------------------------------------------- #
_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date", "datetimestamp"],
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


def normalize_bms_columns(raw: pd.DataFrame, unit_mode: str = UNIT_MODES[0]):
    """Map RP-1043 / BMS headers to canonical names, convert units to degC and kPa abs.
       P_kPa_abs = (P_psig + 14.696) * 6.89476.     Returns (dataframe, meta)."""
    d = raw.copy()
    keys = {c: _key(c) for c in d.columns}
    rp_style = bool(_RP_KEYS & set(keys.values()))
    kpa_native = {_ALIAS_LOOKUP[k] for c, k in keys.items() if k in _ALIAS_LOOKUP and "kpa" in k}
    d = d.rename(columns={c: _ALIAS_LOOKUP[k] for c, k in keys.items() if k in _ALIAS_LOOKUP})
    d = d.loc[:, ~d.columns.duplicated()]

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

    # --- supply voltage (only needed when power comes from current)
    has_amps = "Compressor_A" in d and d["Compressor_A"].notna().any()
    supply_v = detect_supply_voltage(float(d["Compressor_A"].max())) if has_amps else None

    # --- ambient: use measured value, else a proxy from condenser inlet water (tower model)
    amb_proxy = "T_amb" not in d or d["T_amb"].isna().all()
    if amb_proxy:
        d["T_amb"] = ((d["T_cond_in"] - 18.0) / 0.35) if "T_cond_in" in d else 30.0

    # --- timestamp
    has_time = False
    if "timestamp" in d:
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
        has_time = True
    elif isinstance(d.index, pd.DatetimeIndex):
        has_time = True
    meta = {"temp": "°F → °C" if temp_f else "°C (no conversion)",
            "press": "psig → kPa abs" if press_psig else "kPa abs (no conversion)",
            "supply_v": supply_v, "amb_proxy": amb_proxy, "has_time": has_time,
            "has_tdis": "T_discharge" in d.columns}
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
    d["kWTR_Resid_pct"] = (d["kW_per_TR"] / d["kWTR_baseline"] - 1) * 100
    d["Delta_kW"] = (d["kW_per_TR"] - d["kWTR_baseline"]) * d["TR_est"]     # wasted power
    return d


def physics_engine(d: pd.DataFrame, supply_v: float = SIM_SUPPLY_V) -> pd.DataFrame:
    out = d.copy()
    out["T_sat_cond"], out["T_sat_evap"] = t_sat(out["P_cond_kPa"]), t_sat(out["P_evap_kPa"])
    if "kW" not in out.columns:                                   # P = sqrt(3) V I PF
        out["kW"] = np.sqrt(3) * supply_v * out["Compressor_A"] * POWER_FACTOR / 1000

    h1, h4 = h_vap(out["T_sat_evap"]), h_liq(out["T_sat_cond"])   # suction vapour / liquid before valve
    if "T_discharge" in out.columns:                              # measured discharge state
        h2 = h_vap(out["T_sat_cond"]) + CP_SUPERHEAT * (out["T_discharge"] - out["T_sat_cond"])
    else:                                                         # fallback: nominal efficiency
        h2 = h1 + dh_isentropic(out["T_sat_cond"], out["T_sat_evap"]) / ETA_IS_NOM
    out["dh_actual"] = h2 - h1                                    # kJ/kg
    ok = (out["dh_actual"] > 0) & ((h1 - h4) > 0) & (out["kW"] > 0)
    out["m_ref_kg_s"] = out["kW"] / out["dh_actual"]
    out["TR_est"] = np.where(ok, out["m_ref_kg_s"] * (h1 - h4) / KW_PER_TR, np.nan)
    out = out.dropna(subset=["TR_est"]).copy()
    out["kW_per_TR"] = out["kW"] / out["TR_est"]

    out["T_approach"] = out["T_sat_cond"] - out["T_cond_out"]             # condenser approach
    out["T_evap_approach"] = out["T_evap_out"] - out["T_sat_evap"]        # evaporator approach

    out["T_baseline"] = 2.0 + 0.03 * (out["T_amb"] - 25.0)                # ASHRAE G14 weather curve
    out["T_warn"] = out["T_baseline"] + WARN_MARGIN_C
    out["kWTR_baseline"] = ideal_kw_per_tr(out["T_cond_out"] + out["T_baseline"],
                                           out["T_evap_out"] - EVAP_APPROACH_NOM, ETA_IS_NOM)
    return finalize_residuals(out)


def calibrate_baseline(d: pd.DataFrame, mask) -> pd.DataFrame:
    """Re-centre baselines on known-healthy rows (needed for real plants whose design point
    differs from the UAE reference curve): additive offset for approach, scale for kW/TR."""
    out = d.copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 5:
        return out
    off = float((out["T_approach"] - out["T_baseline"])[mask].median())
    scale = float((out["kW_per_TR"] / out["kWTR_baseline"])[mask].median())
    out["T_baseline"] += off
    out["T_warn"] += off
    out["kWTR_baseline"] *= scale
    return finalize_residuals(out)


# --------------------------------------------------------------------------- #
# ML anomaly detection + dual-loop fault classification
# --------------------------------------------------------------------------- #
def detect_anomalies(d: pd.DataFrame, train_mask, contamination: float = 0.01) -> pd.DataFrame:
    """Isolation Forest on weather-normalised residuals (+ evaporator approach), then the
    persistence filter: >= 3 anomaly flags in the last 5 time-steps."""
    out = d.copy()
    feats = ["Approach_Resid", "kWTR_Resid_pct", "T_evap_approach"]
    train = out.loc[np.asarray(train_mask, dtype=bool), feats]

    # Physics-limit flags (dynamic, weather-adjusted): Isolation Forest cannot extrapolate beyond
    # its training range, so hard physical limits guarantee that clear excursions are always flagged.
    rule_flag = ((out["Approach_Resid"] >= WARN_MARGIN_C)             # above baseline + 1.5 C warning curve
                 | (out["T_evap_approach"] >= B_EVAP_APPROACH_MIN)    # low-side collapse
                 | (out["Approach_Resid"] <= COND_LOW_RESID)).to_numpy()  # condenser far below baseline
    ml_flag = np.zeros(len(out), dtype=bool)
    if len(train) >= 30:                                              # ML only with enough healthy history
        model = IsolationForest(n_estimators=300, contamination=contamination, random_state=7).fit(train)
        ml_flag = model.predict(out[feats]) == -1
    out["raw_anomaly"] = np.where(ml_flag | rule_flag, -1, 1)         # 1 normal, -1 anomaly
    out["anomaly_count"] = (out["raw_anomaly"] == -1).rolling(PERSIST_WINDOW, min_periods=1).sum()
    out["persistent_alert"] = out["anomaly_count"] >= PERSIST_HITS
    return out


def classify_faults(d: pd.DataFrame, smooth: int = 3, gate: bool = True) -> pd.DataFrame:
    """Dual-loop thermodynamic signature classifier (high side + low side + power).
    gate=True additionally requires the persistence-filtered ML alert (time-series use);
    gate=False is pure signature logic (steady-state benchmark samples)."""
    out = d.copy()
    sm = (lambda s: s.rolling(smooth, min_periods=1).mean()) if smooth > 1 else (lambda s: s)
    cond_r, evap_a, kw_r = sm(out["Approach_Resid"]), sm(out["T_evap_approach"]), sm(out["kWTR_Resid_pct"])

    # TYPE B - leak/undercharge: low-side collapse, condenser normal/low, power elevated
    type_b = (evap_a >= B_EVAP_APPROACH_MIN) & (cond_r <= B_COND_RESID_MAX) & (kw_r > KW_ELEVATED_PCT)
    # TYPE A - fouling: high-side rise, evaporator loop normal, moderate power penalty
    type_a = (cond_r >= A_COND_RESID_MIN) & (evap_a <= A_EVAP_APPROACH_MAX) & (kw_r >= A_KW_MIN_PCT)
    if A_KW_CAP_PCT is not None:
        type_a &= kw_r <= A_KW_CAP_PCT
    type_a &= ~type_b                                                    # override priority: B wins

    if gate:
        type_a &= out["persistent_alert"]
        type_b &= out["persistent_alert"]
    out["fault_type"] = np.select([type_b, type_a], ["B", "A"], default="")
    out["fault_label"] = out["fault_type"].map(LABEL_OF)
    return out


def add_financials(d: pd.DataFrame, tariff: float) -> pd.DataFrame:
    out = d.copy()
    out["Loss_AED_day"] = np.clip(out["Delta_kW"], 0, None) * 24 * tariff   # Delta_kW * 24 h * tariff
    out["Loss_AED_day_s"] = out["Loss_AED_day"].rolling(6, min_periods=1).mean()
    return out


@st.cache_data(show_spinner=False)
def run_live_pipeline(scenario: str, amb_lo: float, amb_hi: float, seed: int = SEED) -> pd.DataFrame:
    raw = simulate_telemetry(scenario, amb_lo, amb_hi, seed)
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
    loss = float(d["Loss_AED_day"].tail(6).mean())
    cr, ea, kr = tail["Approach_Resid"].mean(), tail["T_evap_approach"].mean(), tail["kWTR_Resid_pct"].mean()
    if last["fault_type"] == "A":
        m = min(len(d), 48)
        slope = float(np.polyfit(np.arange(m), d["T_approach"].tail(m).values, 1)[0] * 24)
        days = 14 if slope <= 0.05 else int(np.clip(np.floor((HARD_TRIP_C - tail["T_approach"].mean()) / slope), 1, 14))
        return ("error" if tail["T_approach"].mean() >= HARD_TRIP_C - 1 else "warning"), (
            f"TYPE A - Condenser tube fouling: condenser approach {cr:+.1f} °C above the weather baseline while "
            f"the evaporator loop is normal (approach {ea:.1f} °C) and kW/TR is {kr:+.1f}%. Trend {slope:+.2f} °C/day. "
            f"Schedule tube descaling within {days} days to avoid the {HARD_TRIP_C} °C trip and ~AED {loss:,.0f}/day waste.")
    if last["fault_type"] == "B":
        return "error", (
            f"TYPE B - Refrigerant leak / undercharge: evaporator approach {ea:.1f} °C (nominal {EVAP_APPROACH_NOM}) "
            f"with condenser approach {cr:+.1f} °C vs baseline and kW/TR {kr:+.1f}%"
            f"{' (severe)' if kr > 15 else ''}. Do NOT descale: check charge/subcooling, leak-test joints and "
            f"seals, review evaporator pressure. Current waste ~AED {loss:,.0f}/day.")
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


def style_fig(fig, title, ytitle):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=360, hovermode="x unified",
                      margin=dict(l=8, r=8, t=45, b=8), font=dict(size=12), yaxis_title=ytitle,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    return fig


def chart_approach(d, train_n=None):
    pts = d[d["persistent_alert"]]                         # red markers ONLY for persistent alerts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["T_approach"], name="T_approach", mode="lines",
                             line=dict(color="#1f77b4", width=1.5)))
    fig.add_trace(go.Scatter(x=d.index, y=d["T_baseline"], name="Weather-adjusted baseline", mode="lines",
                             line=dict(color="#2b9348", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=d.index, y=d["T_warn"], name="Warning (baseline + 1.5 °C)", mode="lines",
                             line=dict(color="#f08c00", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=pts.index, y=pts["T_approach"], name="Persistent anomaly", mode="markers",
                             marker=dict(color="red", size=7)))
    fig.add_hline(y=HARD_TRIP_C, line_dash="dot", line_color="#d62728")
    if train_n:
        fig.add_vrect(x0=d.index[0], x1=d.index[min(train_n, len(d)) - 1], fillcolor="green",
                      opacity=0.05, line_width=0)
    return style_fig(fig, "Condenser Approach vs Weather-Adjusted Baseline", "°C")


def chart_loss(d):
    fig = go.Figure(go.Scatter(x=d.index, y=d["Loss_AED_day_s"], mode="lines", fill="tozeroy",
                               name="Wasted energy (AED/day)", line=dict(color="#e8590c", width=1.5)))
    return style_fig(fig, "Estimated Daily Financial Loss (6-sample avg)", "AED / day")


def read_table(file) -> pd.DataFrame:
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def meta_caption(meta):
    v = f"{meta['supply_v']:.0f} V" if meta["supply_v"] else "n/a (kW supplied directly)"
    st.caption(f"Normalisation → temperatures: {meta['temp']} · pressures: {meta['press']} · supply voltage: {v}")
    if meta["amb_proxy"]:
        st.warning("No ambient temperature column found: T_amb is approximated from condenser inlet water "
                   "temperature (cooling-tower model). Supply a real outdoor temperature for site audits.")
    if not meta["has_tdis"]:
        st.warning("No compressor discharge temperature found: enthalpy rise falls back to a nominal "
                   f"isentropic efficiency ({ETA_IS_NOM}), so efficiency-driven capacity effects are less visible.")


# --------------------------------------------------------------------------- #
# Tab 1 - live simulator
# --------------------------------------------------------------------------- #
def tab_live(amb_lo, amb_hi, scenario, tariff):
    banner = st.empty()
    d = add_financials(run_live_pipeline(scenario, float(amb_lo), float(amb_hi)), tariff)
    last = d.iloc[-1]
    if last["T_approach"] >= HARD_TRIP_C:                                  # independent of the ML filter
        with banner.container():
            st.error("⚠️ CRITICAL HARD THRESHOLD BREACH: Immediate Physical Inspection Required.")
            st.caption(f"T_approach = {last['T_approach']:.1f} °C ≥ {HARD_TRIP_C} °C ({d.index[-1]:%Y-%m-%d %H:%M}).")

    status, scolor = operational_status(d)
    app_now, base_now, warn_now = d["T_approach"].tail(3).mean(), last["T_baseline"], last["T_warn"]
    kwtr_now, kwtr_base = d["kW_per_TR"].tail(3).mean(), d["kWTR_baseline"].tail(3).mean()
    loss_now = float(d["Loss_AED_day"].tail(6).mean())
    wasted_kw = float(np.clip(d["Delta_kW"].tail(6), 0, None).mean())

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
    st.caption(f"Dual-loop signature → condenser resid {t['Approach_Resid'].mean():+.1f} °C · evaporator approach "
               f"{t['T_evap_approach'].mean():.1f} °C · kW/TR {t['kWTR_Resid_pct'].mean():+.1f}% · "
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
    d = physics_engine(d, meta["supply_v"] or SIM_SUPPLY_V)
    if calibrate:
        d = calibrate_baseline(d, (d["truth"] == "SL0_Healthy").values)
    d = classify_faults(d, smooth=1, gate=False)                         # steady-state samples: no persistence
    meta["excluded"] = excluded
    return d, meta, None


def tab_benchmark(unit_mode):
    st.markdown("Evaluates the dual-loop classifier (physics signatures only) against ground-truth fault labels.")
    src = st.radio("Data source", ["Built-in RP-1043-style benchmark (simulated)", "Upload labelled RP-1043 / BMS file"],
                   horizontal=True)
    if src.startswith("Built-in"):
        raw = make_benchmark()
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
            raw = read_table(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
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
def analyze_log(raw, unit_mode, calib_pct):
    d, meta = normalize_bms_columns(raw, unit_mode)
    miss = missing_required(d)
    if miss:
        return None, meta, f"Missing required columns after normalisation: {', '.join(miss)}"
    need = ["T_cond_out", "T_evap_out", "P_cond_kPa", "P_evap_kPa", "kW" if "kW" in d.columns else "Compressor_A"]
    d = d.dropna(subset=need)
    d = physics_engine(d, meta["supply_v"] or SIM_SUPPLY_V)
    if len(d) < 10:
        return None, meta, "Fewer than 10 valid rows after cleaning."
    k = int(len(d) * calib_pct / 100)
    train_mask = np.arange(len(d)) < k
    if k >= 10:                                                           # first calib_pct % assumed healthy
        d = calibrate_baseline(d, train_mask)
    d = detect_anomalies(d, train_mask)                                   # ML only if k >= 30; rules always
    d = classify_faults(d, smooth=3, gate=True)
    meta["ml_used"], meta["train_n"] = k >= 30, k
    return d, meta, None


def tab_audit(unit_mode, tariff):
    demo = st.checkbox("Use built-in demo log (tube fouling, RP-1043 headers, °F/psig, 3300 V amps)")
    up = None if demo else st.file_uploader("BMS log (CSV / Excel)", type=["csv", "xlsx", "xls"], key="audit_up")
    calib_pct = st.slider("Healthy reference window at start of log (%)", 5, 40, 15,
                          help="Used to calibrate baselines and train the anomaly model. Assumed healthy.")
    if demo:
        raw = to_rp1043_raw(simulate_telemetry("Gradual Tube Fouling", 30.0, 44.0), power="Amps").reset_index()
    elif up is not None:
        try:
            raw = read_table(up)
        except Exception as e:
            st.error(f"Could not read file (Excel needs `pip install openpyxl`): {e}")
            return
    else:
        st.info("Upload a BMS export with condenser/evaporator temperatures & pressures and compressor kW or current.")
        return

    d, meta, err = analyze_log(raw, unit_mode, calib_pct)
    if err:
        st.error(err)
        return
    meta_caption(meta)
    if not meta["ml_used"]:
        st.warning("Healthy reference window has <30 rows: Isolation Forest skipped, using physics-limit flags only.")
    d = add_financials(d, tariff)

    # sampling interval -> energy
    if meta["has_time"] and len(d) > 1:
        dt_h = float(np.median(np.diff(d.index.values).astype("timedelta64[s]").astype(float))) / 3600
        dt_h = dt_h if dt_h > 0 else 1.0
        span_days = (d.index[-1] - d.index[0]).total_seconds() / 86400 + dt_h / 24
    else:
        dt_h, span_days = 1.0, len(d) / 24
        st.warning("No timestamp column: assuming a 1-hour sampling interval.")
    waste_kwh = float(np.clip(d["Delta_kW"], 0, None).sum() * dt_h)
    total_aed = waste_kwh * tariff
    per_day = total_aed / span_days if span_days > 0 else 0.0

    counts = d["fault_type"].value_counts()
    share = {t: counts.get(t, 0) / len(d) for t in ["A", "B"]}
    if counts.get("A", 0) + counts.get("B", 0) < max(3, 0.01 * len(d)):      # essentially no fault records
        diagnosis, dcolor = "No dominant fault - operating healthy", "#2b9348"
    else:
        main = "A" if counts.get("A", 0) >= counts.get("B", 0) else "B"
        active = " · active at end of log" if (d["fault_type"].tail(24) == main).any() else ""
        diagnosis = f"{DIAG_NAME[main]} - {share[main] * 100:.0f}% of records{active}"
        dcolor = "#e8590c" if main == "A" else "#c2255c"

    trips = int((d["T_approach"] >= HARD_TRIP_C).sum())
    if trips:
        st.error(f"⚠️ CRITICAL HARD THRESHOLD BREACH: {trips} record(s) with T_approach ≥ {HARD_TRIP_C} °C. "
                 "Immediate Physical Inspection Required.")

    c1, c2 = st.columns(2)
    c1.markdown(kpi("Total period waste", f"AED {total_aed:,.0f}", f"{waste_kwh:,.0f} kWh over {span_days:.1f} days · {tariff:.2f} AED/kWh",
                    "#e8590c"), unsafe_allow_html=True)
    c2.markdown(kpi("Average daily waste", f"AED {per_day:,.0f}/day", f"{len(d):,} valid records", "#e8590c"),
                unsafe_allow_html=True)
    st.markdown(kpi("Primary dataset diagnosis", diagnosis,
                    f"Type A {share['A'] * 100:.0f}% · Type B {share['B'] * 100:.0f}% · healthy/unclassified "
                    f"{(1 - share['A'] - share['B']) * 100:.0f}%", dcolor), unsafe_allow_html=True)

    st.plotly_chart(chart_approach(d, meta["train_n"] if meta["train_n"] >= 10 else None), config={"displayModeBar": False})
    cols = ["T_amb", "T_approach", "T_baseline", "T_warn", "T_evap_approach", "kW", "TR_est", "kW_per_TR",
            "kWTR_baseline", "Delta_kW", "Loss_AED_day", "persistent_alert", "fault_label"]
    st.download_button("Download scored dataset (CSV)", d[cols].round(4).to_csv().encode(),
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

    t1, t2, t3 = st.tabs(["Live UAE Simulator", "RP-1043 Benchmark", "BMS File Audit"])
    with t1:
        tab_live(amb_lo, amb_hi, scenario, tariff)
    with t2:
        tab_benchmark(unit_mode)
    with t3:
        tab_audit(unit_mode, tariff)


if __name__ == "__main__":
    main()
