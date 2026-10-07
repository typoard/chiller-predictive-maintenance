"""
District Cooling Centrifugal Chiller - Predictive Maintenance & Anomaly Detection MVP
Run:  streamlit run chiller_predictive_maintenance_mvp.py
Deps: pip install streamlit pandas numpy scikit-learn plotly

Modules (in order):
  1. Synthetic data generator  (30 days hourly telemetry, fouling + bearing wear in last 5 days)
  2. Feature engineering       (condenser approach, dT_evap, kW/TR, COP, lift)
  3. Anomaly engine            (Isolation Forest trained on healthy baseline -> Asset Health Index)
  4. Diagnostics & economics   (load-adjusted baselines, rule-based recommendations, $/day loss)
  5. Streamlit dashboard
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N_HOURS = 30 * 24                  # 30 days hourly
FAULT_HOURS = 5 * 24               # anomalies injected in final 5 days
TRAIN_HOURS = 25 * 24              # healthy baseline used for training
CP_WATER = 4.186                   # kJ/kg.K
KW_PER_TR = 3.517                  # 1 TR = 3.517 kW thermal
DESIGN_DT_EVAP = 5.5               # K at full load
# R-134a saturation fit: ln(P_kPa) = A - B / T_K   (accurate to ~1% for 0-50 C)
SAT_A, SAT_B = 15.428, 2662.6

MODEL_FEATURES = ["Cond_Approach_C", "dT_evap_C", "kW_per_TR", "Vibration_mm_s",
                  "kW_power", "P_cond_kPa", "P_evap_kPa", "Load_pct"]


def p_sat_kpa(t_c):
    """Refrigerant (R-134a) saturation pressure [kPa abs] from temperature [C]."""
    return np.exp(SAT_A - SAT_B / (np.asarray(t_c) + 273.15))


def t_sat_c(p_kpa):
    """Refrigerant (R-134a) saturation temperature [C] from pressure [kPa abs]."""
    return SAT_B / (SAT_A - np.log(np.asarray(p_kpa))) - 273.15


# --------------------------------------------------------------------------- #
# 1. Synthetic data generator
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def generate_telemetry(capacity_tr: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_HOURS
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq="h")
    hour = idx.hour.values
    i = np.arange(n)

    # Cooling load fraction: diurnal profile (peak ~15:00) + smoothed noise
    noise = pd.Series(rng.normal(0, 0.05, n)).rolling(3, min_periods=1).mean().values
    load = np.clip(0.62 + 0.20 * np.sin(2 * np.pi * (hour - 9) / 24) + noise, 0.35, 1.0)
    q_evap_kw = load * capacity_tr * KW_PER_TR

    # Fault ramp (0 -> 1 across final 5 days)
    ramp = np.clip((i - (n - FAULT_HOURS)) / FAULT_HOURS, 0, 1)

    # Evaporator water side
    t_evap_out = 6.0 + rng.normal(0, 0.12, n)
    dt_evap = DESIGN_DT_EVAP * load + rng.normal(0, 0.08, n)
    t_evap_in = t_evap_out + dt_evap

    # Condenser water side (cooling-tower supply follows ambient)
    t_cond_in = (29.5 + 2.5 * np.sin(2 * np.pi * (hour - 9) / 24)
                 + 0.8 * np.sin(2 * np.pi * i / (24 * 9)) + rng.normal(0, 0.3, n))
    t_cond_out = t_cond_in + 6.5 * load + rng.normal(0, 0.1, n)

    # Condenser approach (T_sat - T_cond_out): healthy ~1.5-2.5 K, fouling adds up to +3.2 K
    approach = 1.3 + 1.2 * load + rng.normal(0, 0.12, n) + 3.2 * ramp ** 1.3
    t_cond_sat = t_cond_out + approach
    t_evap_sat = t_evap_out - 1.2 + rng.normal(0, 0.05, n)

    # Compressor efficiency = fraction of Carnot COP (fouling raises lift -> power rises naturally)
    carnot = (t_evap_sat + 273.15) / (t_cond_sat - t_evap_sat)
    eta = 0.60 + 0.10 * load + rng.normal(0, 0.008, n)
    kw = q_evap_kw / (eta * carnot)
    kw *= 1 + 0.02 * ramp                                   # extra bearing friction

    # Refrigerant pressures from saturation temperatures
    p_cond = p_sat_kpa(t_cond_sat) * (1 + rng.normal(0, 0.004, n))
    p_evap = p_sat_kpa(t_evap_sat) * (1 + rng.normal(0, 0.004, n))

    # Bearing vibration: baseline + progressive wear + intermittent spikes in fault window
    vib = 1.8 + 0.9 * load + rng.normal(0, 0.12, n) + 0.9 * ramp
    spikes = (rng.random(n) < 0.10) & (ramp > 0)
    vib = vib + spikes * rng.uniform(1.2, 3.0, n) * (0.5 + ramp)

    return pd.DataFrame({
        "timestamp": idx, "T_evap_in": t_evap_in, "T_evap_out": t_evap_out,
        "T_cond_in": t_cond_in, "T_cond_out": t_cond_out,
        "P_cond": p_cond, "P_evap": p_evap, "kW_power": kw, "Vibration_mm_s": vib,
    }).set_index("timestamp")


# --------------------------------------------------------------------------- #
# 2. Mechanical feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame, capacity_tr: int) -> pd.DataFrame:
    """Adds thermodynamic KPIs.
    Condenser approach is defined conventionally as T_sat,cond - T_cond_out (>= 0 on a healthy
    unit); T_sat,cond is derived from measured condenser refrigerant pressure."""
    out = df.copy()
    design_flow = capacity_tr * KW_PER_TR / (CP_WATER * DESIGN_DT_EVAP)   # kg/s (assumed constant)

    out["T_sat_cond_C"] = t_sat_c(out["P_cond"])
    out["T_sat_evap_C"] = t_sat_c(out["P_evap"])
    out["Cond_Approach_C"] = out["T_sat_cond_C"] - out["T_cond_out"]
    out["dT_evap_C"] = out["T_evap_in"] - out["T_evap_out"]
    out["Lift_K"] = out["T_sat_cond_C"] - out["T_sat_evap_C"]

    out["Q_evap_kW"] = design_flow * CP_WATER * out["dT_evap_C"]
    out["Cooling_TR"] = out["Q_evap_kW"] / KW_PER_TR
    out["Load_frac"] = out["Cooling_TR"] / capacity_tr
    out["Load_pct"] = out["Load_frac"] * 100
    out["kW_per_TR"] = out["kW_power"] / out["Cooling_TR"]
    out["COP"] = out["Q_evap_kW"] / out["kW_power"]
    out.rename(columns={"P_cond": "P_cond_kPa", "P_evap": "P_evap_kPa"}, inplace=True)
    return out


# --------------------------------------------------------------------------- #
# 3. AI anomaly detection engine
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_anomaly_detection(feat: pd.DataFrame, contamination: float = 0.01, seed: int = 7) -> pd.DataFrame:
    """Train Isolation Forest on healthy window, score full dataset.
    Anomaly_Flag: -1 anomaly / 1 normal.  Health_Index: 0-100 %, 50 % == model decision boundary,
    100 % == at/above the median healthy sample."""
    train = feat.iloc[:TRAIN_HOURS][MODEL_FEATURES]
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    model.fit(train)

    out = feat.copy()
    out["Anomaly_Flag"] = model.predict(out[MODEL_FEATURES])
    s = model.decision_function(out[MODEL_FEATURES])          # >0 normal, <0 anomalous
    s_ref = max(float(np.median(model.decision_function(train))), 1e-6)
    out["Anomaly_Score"] = s
    out["Health_Index"] = np.clip(50 + 50 * s / s_ref, 0, 100)
    out["Health_Smooth"] = out["Health_Index"].rolling(6, min_periods=1).mean()
    return out


# ===== NEW: persistence filter + hard fail-safe limits ===================== #
PERSIST_WINDOW = 5          # look-back window (timestamps)
PERSIST_MIN_HITS = 3        # anomalies required inside the window to raise an alert
HARD_LIMIT_APPROACH_C = 5.5
HARD_LIMIT_VIBRATION = 6.5  # mm/s


def apply_persistence_filter(df: pd.DataFrame, window: int = PERSIST_WINDOW,
                             min_hits: int = PERSIST_MIN_HITS) -> pd.DataFrame:
    """Noise reduction: an alert is only 'persistent' when >= min_hits of the last `window`
    readings were flagged by the Isolation Forest. Isolated single anomalies stay False."""
    out = df.copy()
    out["raw_anomaly"] = out["Anomaly_Flag"]                       # 1 = normal, -1 = anomaly
    out["anomaly_count"] = (out["raw_anomaly"] == -1).rolling(window=window, min_periods=1).sum()
    out["persistent_alert"] = out["anomaly_count"] >= min_hits
    return out
# =========================================================================== #


# --------------------------------------------------------------------------- #
# 4. Diagnostics, economics, rule-based recommendations
# --------------------------------------------------------------------------- #
def _fit_baseline(healthy: pd.DataFrame, target: str, cols: list):
    """Least-squares load/ambient-adjusted baseline fitted on healthy data."""
    X = np.column_stack([np.ones(len(healthy))] + [healthy[c].values for c in cols])
    beta = np.linalg.lstsq(X, healthy[target].values, rcond=None)[0]
    sigma = float(np.std(healthy[target].values - X @ beta))

    def predict(d):
        return np.column_stack([np.ones(len(d))] + [d[c].values for c in cols]) @ beta
    return predict, sigma


def analyze(df: pd.DataFrame, tariff: float, cur: str):
    d = df.copy()
    healthy = d.iloc[:TRAIN_HOURS]
    f_app, _ = _fit_baseline(healthy, "Cond_Approach_C", ["Load_frac"])
    f_eff, _ = _fit_baseline(healthy, "kW_per_TR", ["Load_frac", "T_cond_in"])
    f_vib, sig_vib = _fit_baseline(healthy, "Vibration_mm_s", ["Load_frac"])

    d["Approach_Expected"] = f_app(d)
    d["kWTR_Expected"] = f_eff(d)
    d["Vib_Expected"] = f_vib(d)
    d["Approach_Excess"] = d["Cond_Approach_C"] - d["Approach_Expected"]
    d["Excess_kW"] = d["kW_power"] - d["kWTR_Expected"] * d["Cooling_TR"]

    r = d.tail(24)                                           # "current" = last 24 h
    loss_day = float(np.clip(r["Excess_kW"], 0, None).mean() * 24 * tariff)
    app_ex = float(r["Approach_Excess"].mean())
    eff_ex_pct = float((r["kW_per_TR"] / r["kWTR_Expected"] - 1).mean() * 100)
    vib_ex = float((r["Vibration_mm_s"] - r["Vib_Expected"]).mean())
    vib_spikes = int(((r["Vibration_mm_s"] - r["Vib_Expected"]) > 5 * sig_vib).sum())
    vib_peak = float(r["Vibration_mm_s"].max())

    insights = []
    # --- Condenser fouling rule
    if app_ex > 1.0:
        slope_day = float(np.polyfit(np.arange(FAULT_HOURS), d["Approach_Excess"].tail(FAULT_HOURS).values, 1)[0] * 24)
        limit = 3.0
        if app_ex >= limit:
            when = "immediately (within 3 days)"
        elif slope_day > 0.05:
            when = f"within {int(np.clip(np.floor((limit - app_ex) / slope_day), 3, 14))} days"
        else:
            when = "within 14 days"
        sev = "error" if app_ex >= 2.5 else "warning"
        insights.append((sev, f"{'CRITICAL' if sev == 'error' else 'Warning'}: Condenser approach temperature exceeds "
                              f"load-adjusted baseline by +{app_ex:.1f} °C (trend {slope_day:+.2f} °C/day). "
                              f"Schedule condenser tube descaling / brush cleaning {when} to prevent "
                              f"~{cur} {loss_day:,.0f}/day efficiency loss. Verify cooling-tower water treatment."))
    # --- Bearing wear rule
    if vib_ex > 3 * sig_vib or vib_spikes >= 2:
        sev = "error" if vib_peak >= 4.5 or vib_spikes >= 4 else "warning"
        insights.append((sev, f"{'CRITICAL' if sev == 'error' else 'Warning'}: Compressor bearing vibration is "
                              f"+{vib_ex:.2f} mm/s above baseline with {vib_spikes} spike(s) in the last 24 h "
                              f"(peak {vib_peak:.1f} mm/s). Schedule vibration spectrum analysis and oil/bearing "
                              f"inspection within 7 days; plan bearing replacement at next outage if spikes persist."))
    # --- Efficiency rule
    if eff_ex_pct > 4:
        insights.append(("warning", f"Efficiency: kW/TR is {eff_ex_pct:.1f}% worse than the healthy baseline at the same "
                                    f"load and condenser water temperature (avg {r['kW_per_TR'].mean():.3f} vs "
                                    f"{r['kWTR_Expected'].mean():.3f} kW/TR)."))
    if not insights:
        insights.append(("success", "All monitored parameters are within load-adjusted healthy baselines. "
                                    "Continue routine preventive maintenance."))
    return d, {"loss_day": loss_day, "approach_excess": app_ex, "insights": insights}


def alarm_status(df: pd.DataFrame):
    r = df.tail(24)
    h, frac = float(r["Health_Smooth"].mean()), float(r["persistent_alert"].mean())   # CHANGED: persistent only
    if h < 50 or frac > 0.5:
        return "CRITICAL", "#d62728"
    if h < 70 or frac > 0.2:
        return "WARNING", "#ff9900"
    return "NORMAL", "#2ca02c"


# --------------------------------------------------------------------------- #
# 5. Streamlit dashboard
# --------------------------------------------------------------------------- #
def chart_approach(d: pd.DataFrame) -> go.Figure:
    an = d[d["persistent_alert"]]          # CHANGED: isolated anomalies are treated as normal points
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Cond_Approach_C"], name="Condenser Approach",
                             mode="lines", line=dict(color="#1f77b4", width=1.5)))
    fig.add_trace(go.Scatter(x=d.index, y=d["Approach_Expected"] + 1.5, name="Warning limit (baseline +1.5 °C)",
                             mode="lines", line=dict(color="orange", dash="dash", width=1.2)))
    fig.add_trace(go.Scatter(x=an.index, y=an["Cond_Approach_C"], name="Persistent anomaly (≥3 of last 5)", mode="markers",
                             marker=dict(color="red", size=7, symbol="circle")))
    fig.add_vrect(x0=d.index[0], x1=d.index[TRAIN_HOURS - 1], fillcolor="green", opacity=0.06,
                  line_width=0, annotation_text="Training baseline (healthy)", annotation_position="top left")
    fig.update_layout(title="Condenser Approach Temperature", yaxis_title="°C", height=400,
                      hovermode="x unified", margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.15))
    return fig


def chart_power_vib(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d.index, y=d["kW_power"], name="Compressor Power (kW)",
                             line=dict(color="#17becf", width=1.3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=d.index, y=d["Vibration_mm_s"], name="Bearing Vibration (mm/s)",
                             line=dict(color="#d62728", width=1.3)), secondary_y=True)
    an = d[d["persistent_alert"]]          # NEW: same persistence rule as Chart 1
    fig.add_trace(go.Scatter(x=an.index, y=an["Vibration_mm_s"], name="Persistent anomaly", mode="markers",
                             marker=dict(color="red", size=7)), secondary_y=True)
    fig.update_yaxes(title_text="kW", secondary_y=False)
    fig.update_yaxes(title_text="mm/s", secondary_y=True)
    fig.update_layout(title="Compressor Power vs Bearing Vibration", height=400, hovermode="x unified",
                      margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.15))
    return fig


def chart_health(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=d.index, y=d["Health_Smooth"], fill="tozeroy", name="Health Index",
                               line=dict(color="#2ca02c")))
    fig.add_hline(y=70, line_dash="dash", line_color="orange")
    fig.add_hline(y=50, line_dash="dash", line_color="red")
    fig.update_layout(title="Asset Health Index (6 h smoothed)", yaxis=dict(range=[0, 100], title="%"),
                      height=300, margin=dict(t=50, b=20))
    return fig


def main():
    st.set_page_config(page_title="Chiller Predictive Maintenance", page_icon="❄️", layout="wide")
    banner = st.empty()    # NEW: reserved slot at the very top; filled below once data is loaded
    st.title("❄️ Centrifugal Chiller – Predictive Maintenance & Anomaly Detection")
    st.caption("District cooling asset health monitoring · Isolation Forest on thermodynamic features · "
               "Synthetic ASHRAE RP-1043-style telemetry")

    with st.sidebar:
        st.header("Configuration")
        capacity = st.number_input("Chiller capacity (TR)", 300, 3000, 1000, 100)
        cur = st.radio("Currency", ["AED", "USD"], horizontal=True)
        tariff = st.number_input(f"Electricity tariff ({cur}/kWh)", 0.01, 2.0,
                                 0.38 if cur == "AED" else 0.10, 0.01, key=f"tariff_{cur}")
        contamination = st.slider("Model contamination (healthy false-alarm rate)", 0.005, 0.05, 0.01, 0.005)
        seed = st.number_input("Simulation seed", 0, 9999, 42)
        st.markdown("---")
        st.caption(f"Training window: first {TRAIN_HOURS // 24} days (healthy). "
                   f"Faults injected: last {FAULT_HOURS // 24} days.")

    raw = generate_telemetry(int(capacity), int(seed))
    feat = engineer_features(raw, int(capacity))
    scored = run_anomaly_detection(feat, float(contamination))
    scored = apply_persistence_filter(scored)                    # NEW: persistence filtering
    d, rep = analyze(scored, tariff, cur)

    # NEW: hard-threshold fail-safe on the latest telemetry record (independent of the AI model)
    latest = d.iloc[-1]
    if latest["Cond_Approach_C"] > HARD_LIMIT_APPROACH_C or latest["Vibration_mm_s"] > HARD_LIMIT_VIBRATION:
        with banner.container():
            st.error("⚠️ CRITICAL HARD THRESHOLD BREACH: Immediate Physical Inspection Required.")
            st.caption(f"Latest reading ({d.index[-1]:%Y-%m-%d %H:%M}): condenser approach "
                       f"{latest['Cond_Approach_C']:.1f} °C (limit {HARD_LIMIT_APPROACH_C}), bearing vibration "
                       f"{latest['Vibration_mm_s']:.1f} mm/s (limit {HARD_LIMIT_VIBRATION}).")

    status, color = alarm_status(d)
    now = d.iloc[-24:]
    base_kwtr = d.iloc[:TRAIN_HOURS]["kW_per_TR"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Asset Health Index", f"{now['Health_Smooth'].mean():.0f} %",
              delta=f"{now['Health_Smooth'].mean() - d.iloc[:TRAIN_HOURS]['Health_Smooth'].mean():.0f} pts vs baseline")
    k2.metric("Efficiency (kW/TR)", f"{now['kW_per_TR'].mean():.3f}",
              delta=f"{now['kW_per_TR'].mean() - base_kwtr:+.3f} vs healthy avg", delta_color="inverse")
    k3.markdown("**Active Alarm Status**")
    k3.markdown(f"<div style='font-size:2rem;font-weight:700;color:{color}'>● {status}</div>",
                unsafe_allow_html=True)
    k4.metric("Est. Financial Loss", f"{cur} {rep['loss_day']:,.0f} /day",
              delta=f"≈ {cur} {rep['loss_day'] * 30:,.0f} / month", delta_color="off")

    st.plotly_chart(chart_approach(d), use_container_width=True)
    st.plotly_chart(chart_power_vib(d), use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(chart_health(d), use_container_width=True)
    with c2:
        st.subheader("🔧 Actionable Insights")
        for sev, msg in rep["insights"]:
            getattr(st, sev)(msg)
        st.caption(f"Persistent-alert hours in last 24 h: {int(now['persistent_alert'].sum())} / 24 "
                   f"(raw model flags: {int((now['raw_anomaly'] == -1).sum())})")

    with st.expander("Engineered feature table & export"):
        cols = ["T_evap_in", "T_evap_out", "T_cond_in", "T_cond_out", "P_cond_kPa", "P_evap_kPa", "kW_power",
                "Vibration_mm_s", "Cond_Approach_C", "dT_evap_C", "COP", "kW_per_TR", "Load_pct",
                "raw_anomaly", "anomaly_count", "persistent_alert", "Health_Index"]
        st.dataframe(d[cols].round(3).tail(200), use_container_width=True)
        st.download_button("Download full dataset (CSV)", d[cols].round(4).to_csv().encode(),
                           "chiller_telemetry_scored.csv", "text/csv")


if __name__ == "__main__":
    main()
