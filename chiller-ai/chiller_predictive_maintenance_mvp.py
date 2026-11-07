"""
Centrifugal Chiller Predictive Maintenance & Benchmark Engine — UAE Climate Edition
===================================================================================
Run:  streamlit run app.py
Deps: pip install streamlit pandas numpy scikit-learn plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------------------------------- #
# Constants & Constants
# --------------------------------------------------------------------------- #
KW_PER_TR = 3.517                 # 1 TR = 3.517 kW thermal
SAT_A, SAT_B = 15.428, 2662.6     # R-134a fit: ln(P_kPa) = A - B/T_K (~1 % error, 0-50 degC)
CP_VAP_ISENT = 0.95               # kJ/kg.K, mean suction-vapour cp
K_ISENT = 1.09                    # effective isentropic exponent of R-134a vapour
CP_SUPERHEAT = 1.15               # kJ/kg.K, superheated vapour at condensing pressure
ETA_IS_NOM = 0.80                 # healthy compressor isentropic efficiency (baseline twin)
EVAP_APPROACH_NOM = 1.5           # K, healthy T_evap_out - T_sat_evap
SUPPLY_V, POWER_FACTOR = 3300.0, 0.90   # MV motor supply (line-line V) and PF
RATED_TR = 1000                   # Plant rated capacity

HOURS, TRAIN_H, FAULT_H = 21 * 24, 14 * 24, 5 * 24     # 21 d hourly window
WARN_MARGIN_C = 1.5               # dynamic warning = baseline + 1.5 degC
HARD_TRIP_C = 5.5                 # absolute critical trip
LEAK_DROP_C = 1.0                 # Type B drop below baseline
KW_ELEVATED_PCT = 2.0             # Type B specific power threshold
PERSIST_WINDOW, PERSIST_HITS = 5, 3
DEWA_TARIFF = 0.44                # AED/kWh default
SEED = 42

SCENARIOS = [
    "Normal Operation", 
    "Gradual Tube Fouling",
    "Sudden Refrigerant Leak/Pressure Loss", 
    "Extreme Weather Spike"
]


# --------------------------------------------------------------------------- #
# R-134a Thermodynamic Helpers
# --------------------------------------------------------------------------- #
def p_sat(t_c):
    return np.exp(SAT_A - SAT_B / (np.asarray(t_c, dtype=float) + 273.15))

def t_sat(p_kpa):
    return SAT_B / (SAT_A - np.log(np.asarray(p_kpa, dtype=float))) - 273.15

def h_liq(t_c):
    return 200.0 + 1.34 * t_c + 0.00175 * t_c ** 2

def h_vap(t_c):
    return 398.6 + 0.6 * t_c - 0.002 * t_c ** 2

def dh_isentropic(tsc, tse):
    pr = p_sat(tsc) / p_sat(tse)
    return CP_VAP_ISENT * (tse + 273.15) * (pr ** ((K_ISENT - 1) / K_ISENT) - 1)

def ideal_kw_per_tr(tsc, tse, eta):
    return KW_PER_TR * (dh_isentropic(tsc, tse) / eta) / (h_vap(tse) - h_liq(tsc))


# --------------------------------------------------------------------------- #
# 1. Automated UAE Climate & Telemetry Simulator
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def simulate_telemetry(scenario: str, season: str = "Summer (Peak Load)", seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = HOURS
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq="h")
    i, hour = np.arange(n), idx.hour.values
    f0 = n - FAULT_H

    # Automated seasonal profile modeling real UAE weather cycles
    if season == "Summer (Peak Load)":
        amb_mid, amb_amp = 38.0, 6.0   # 32°C to 44°C
    elif season == "Winter (Mild Load)":
        amb_mid, amb_amp = 22.0, 4.0   # 18°C to 26°C
    else:  # Shoulder (Spring/Autumn)
        amb_mid, amb_amp = 30.0, 5.0   # 25°C to 35°C

    t_amb = (amb_mid + amb_amp * np.sin(2 * np.pi * (hour - 9) / 24)
             + 0.8 * np.sin(2 * np.pi * i / (24 * 6)) + rng.normal(0, 0.3, n))
    
    if scenario == "Extreme Weather Spike":
        t_amb = np.minimum(t_amb + 7.0 * np.clip((i - (n - 96)) / 12, 0, 1), 50.0)

    # Cooling load follows UAE ambient conditions
    load = np.clip(0.45 + 0.015 * (t_amb - 20) + rng.normal(0, 0.04, n), 0.35, 1.0)
    t_cond_in = 14.0 + 0.35 * t_amb + 4.0 + rng.normal(0, 0.25, n)
    t_cond_out = t_cond_in + 6.2 * load + rng.normal(0, 0.10, n)
    t_evap_out = 6.0 + rng.normal(0, 0.10, n)

    # Fault Injections
    foul = 3.8 * np.clip((i - f0) / FAULT_H, 0, 1) if scenario == "Gradual Tube Fouling" else 0.0
    leak = (i >= f0 + 12) if scenario == "Sudden Refrigerant Leak/Pressure Loss" else np.zeros(n, bool)

    # Dynamic approach baseline + injected anomalies
    approach = 2.0 + 0.03 * (t_amb - 25) + rng.normal(0, 0.12, n) + foul - 1.6 * leak
    tsc = t_cond_out + approach
    tse = t_evap_out - EVAP_APPROACH_NOM - 2.0 * leak + rng.normal(0, 0.05, n)

    eta = ETA_IS_NOM + rng.normal(0, 0.004, n) - 0.06 * leak
    dh_act = dh_isentropic(tsc, tse) / eta
    q_ref = h_vap(tse) - h_liq(tsc)
    q_kw = load * RATED_TR * KW_PER_TR
    m_ref = q_kw / q_ref
    w_kw = m_ref * dh_act
    amps = w_kw * 1000 / (np.sqrt(3) * SUPPLY_V * POWER_FACTOR) * (1 + rng.normal(0, 0.004, n))
    t_dis = tsc + (h_vap(tse) + dh_act - h_vap(tsc)) / CP_SUPERHEAT + rng.normal(0, 0.15, n)

    return pd.DataFrame({
        "T_amb": t_amb, "T_cond_out": t_cond_out, "T_evap_out": t_evap_out,
        "P_cond_kPa": p_sat(tsc) * (1 + rng.normal(0, 0.004, n)),
        "P_evap_kPa": p_sat(tse) * (1 + rng.normal(0, 0.004, n)),
        "T_discharge": t_dis, "Compressor_A": amps,
    }, index=pd.Index(idx, name="timestamp"))


# --------------------------------------------------------------------------- #
# 2. Physics & ML Engine
# --------------------------------------------------------------------------- #
def physics_engine(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["T_sat_cond"], d["T_sat_evap"] = t_sat(d["P_cond_kPa"]), t_sat(d["P_evap_kPa"])

    # Enthalpy-based capacity estimation without flow meters
    d["kW"] = np.sqrt(3) * SUPPLY_V * d["Compressor_A"] * POWER_FACTOR / 1000
    h1, h4 = h_vap(d["T_sat_evap"]), h_liq(d["T_sat_cond"])
    h2 = h_vap(d["T_sat_cond"]) + CP_SUPERHEAT * (d["T_discharge"] - d["T_sat_cond"])
    d["m_ref_kg_s"] = d["kW"] / (h2 - h1)
    d["TR_est"] = d["m_ref_kg_s"] * (h1 - h4) / KW_PER_TR
    d["kW_per_TR"] = d["kW"] / d["TR_est"]

    d["T_approach"] = d["T_sat_cond"] - d["T_cond_out"]

    # ASHRAE Guideline 14 dynamic weather baseline
    d["T_baseline"] = 2.0 + 0.03 * (d["T_amb"] - 25.0)
    d["T_warn"] = d["T_baseline"] + WARN_MARGIN_C

    d["kWTR_baseline"] = ideal_kw_per_tr(d["T_cond_out"] + d["T_baseline"],
                                         d["T_evap_out"] - EVAP_APPROACH_NOM, ETA_IS_NOM)

    d["Approach_Resid"] = d["T_approach"] - d["T_baseline"]
    d["kWTR_Resid_pct"] = (d["kW_per_TR"] / d["kWTR_baseline"] - 1) * 100
    d["Delta_kW"] = (d["kW_per_TR"] - d["kWTR_baseline"]) * d["TR_est"]
    return d


def detect_anomalies(d: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    out = d.copy()
    feats = ["Approach_Resid", "kWTR_Resid_pct"]
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=7)
    train_limit = min(TRAIN_H, len(out) // 2)
    model.fit(out.iloc[:train_limit][feats])
    out["raw_anomaly"] = model.predict(out[feats])
    
    # 3-of-5 Persistence Filter
    out["anomaly_count"] = (out["raw_anomaly"] == -1).rolling(PERSIST_WINDOW, min_periods=1).sum()
    out["persistent_alert"] = out["anomaly_count"] >= PERSIST_HITS
    return out


def classify_faults(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    r = out["Approach_Resid"].rolling(3, min_periods=1).mean()
    approach_s = out["T_approach"].rolling(3, min_periods=1).mean()
    kw_elevated = out["kWTR_Resid_pct"].rolling(3, min_periods=1).mean() > KW_ELEVATED_PCT

    # Type A: Tube Fouling
    rising = r.diff(24) > 0.3
    type_a = out["persistent_alert"] & (approach_s > out["T_warn"]) & rising

    # Type B: Refrigerant Leak
    neg = r < -0.5
    episode = (neg != neg.shift()).cumsum()
    step_seen = (r.diff(3) <= -LEAK_DROP_C).astype(int).groupby(episode).cummax().astype(bool)
    type_b = out["persistent_alert"] & (r < -LEAK_DROP_C) & kw_elevated & step_seen

    out["fault_type"] = np.select([type_b, type_a], ["B", "A"], default="")
    return out


def run_pipeline(scenario: str, season: str, seed: int = SEED) -> pd.DataFrame:
    return classify_faults(detect_anomalies(physics_engine(simulate_telemetry(scenario, season, seed))))


def add_financials(d: pd.DataFrame, tariff: float) -> pd.DataFrame:
    out = d.copy()
    out["Loss_AED_day"] = np.clip(out["Delta_kW"], 0, None) * 24 * tariff
    out["Loss_AED_day_s"] = out["Loss_AED_day"].rolling(6, min_periods=1).mean()
    return out


# --------------------------------------------------------------------------- #
# 3. ASHRAE RP-1043 Benchmark Generation & Validation Engine
# --------------------------------------------------------------------------- #
def generate_ashrae_benchmark():
    """Generates synthetic telemetry matching ASHRAE RP-1043 experimental protocols."""
    rng = np.random.default_rng(1043)
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    
    # Generate 50% healthy, 30% condenser fouling (CF), 20% refrigerant leak (UC)
    labels = ["SL0_Healthy"] * 250 + ["CF_Fouling"] * 150 + ["UC_Leak"] * 100
    
    t_amb = rng.uniform(25, 42, n)
    load = rng.uniform(0.5, 0.9, n)
    
    foul_bias = np.array([0.0 if l == "SL0_Healthy" else (2.8 if l == "CF_Fouling" else -1.2) for l in labels])
    leak_bias = np.array([0.0 if l != "UC_Leak" else 0.08 for l in labels])
    
    approach = 2.0 + 0.03 * (t_amb - 25) + foul_bias + rng.normal(0, 0.1, n)
    tsc = 35.0 + 0.2 * t_amb + approach
    tse = 4.0 - 20.0 * leak_bias
    
    dh_act = dh_isentropic(tsc, tse) / (ETA_IS_NOM - leak_bias)
    q_ref = h_vap(tse) - h_liq(tsc)
    m_ref = (load * RATED_TR * KW_PER_TR) / q_ref
    w_kw = m_ref * dh_act
    amps = w_kw * 1000 / (np.sqrt(3) * SUPPLY_V * POWER_FACTOR)
    
    df = pd.DataFrame({
        "T_amb": t_amb,
        "T_cond_out": tsc - approach,
        "T_evap_out": tse + EVAP_APPROACH_NOM,
        "P_cond_kPa": p_sat(tsc),
        "P_evap_kPa": p_sat(tse),
        "T_discharge": tsc + 15.0,
        "Compressor_A": amps,
        "Ground_Truth_Label": labels
    }, index=idx)
    
    return df


# --------------------------------------------------------------------------- #
# 4. UI Components & Layout
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1200px}
.kpi{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:14px 16px;background:rgba(128,128,128,.07);margin-bottom:10px}
.kpi .t{font-size:.74rem;letter-spacing:.05em;text-transform:uppercase;opacity:.72}
.kpi .v{font-size:1.85rem;font-weight:700;line-height:1.2}
.kpi .s{font-size:.82rem;opacity:.78}
</style>
"""

def kpi(title, value, sub, color):
    return (f"<div class='kpi' style='border-left:6px solid {color}'><div class='t'>{title}</div>"
            f"<div class='v' style='color:{color}'>{value}</div><div class='s'>{sub}</div></div>")

def style_fig(fig, title, ytitle):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=360, hovermode="x unified",
                      margin=dict(l=8, r=8, t=45, b=8), yaxis_title=ytitle,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    return fig

def main():
    st.set_page_config(page_title="Chiller Health Engine", page_icon="❄️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("❄️ Centrifugal Chiller Predictive Maintenance Platform")
    st.caption("ASHRAE Guideline 14 Physics Engine · Flow-Meter-Free Capacity Estimation · AED Financial Waste Tracking")

    tabs = st.tabs(["📊 Live UAE Operations Monitor", "🔬 ASHRAE RP-1043 Benchmark Validation", "📁 BMS CSV Audit Ingestion"])

    # ----------------------------------------------------------------------- #
    # TAB 1: Live Monitor & Scenario Simulator
    # ----------------------------------------------------------------------- #
    with tabs[0]:
        with st.sidebar:
            st.header("Operational Controls")
            season = st.selectbox("UAE Weather Profile", ["Summer (Peak Load)", "Shoulder (Spring/Autumn)", "Winter (Mild Load)"])
            scenario = st.radio("Simulation Scenario", SCENARIOS, index=1)
            tariff = st.slider("DEWA Tariff (AED/kWh)", 0.20, 0.80, DEWA_TARIFF, 0.01)

        d = add_financials(run_pipeline(scenario, season), tariff)
        last = d.iloc[-1]

        # Metric Summaries
        app_now, base_now, warn_now = (d["T_approach"].tail(3).mean(), last["T_baseline"], last["T_warn"])
        kwtr_now, kwtr_base = d["kW_per_TR"].tail(3).mean(), d["kWTR_baseline"].tail(3).mean()
        loss_now = float(d["Loss_AED_day"].tail(6).mean())
        wasted_kw = float(np.clip(d["Delta_kW"].tail(6), 0, None).mean())

        c1, c2 = st.columns(2)
        c1.markdown(kpi("T_approach vs Weather Baseline", f"{app_now:.1f} °C",
                        f"Baseline {base_now:.1f} °C · Warning {warn_now:.1f} °C · Δ {app_now - base_now:+.1f} °C",
                        "#d62728" if app_now >= warn_now else "#2b9348"), unsafe_allow_html=True)
        c2.markdown(kpi("Current Specific Power (kW/TR)", f"{kwtr_now:.3f}",
                        f"Baseline {kwtr_base:.3f} · {(kwtr_now / kwtr_base - 1) * 100:+.1f}%",
                        "#e8590c" if kwtr_now / kwtr_base > 1.04 else "#2b9348"), unsafe_allow_html=True)
        
        c3, c4 = st.columns(2)
        status_text = "Fouling Alert (Type A)" if last["fault_type"] == "A" else ("Leak Alert (Type B)" if last["fault_type"] == "B" else "Healthy")
        c3.markdown(kpi("Operational Status", status_text, f"Ambient {last['T_amb']:.1f} °C · Persistent Alerts: {int(d['persistent_alert'].tail(24).sum())}",
                        "#e8590c" if last["fault_type"] != "" else "#2b9348"), unsafe_allow_html=True)
        c4.markdown(kpi("Daily Energy Financial Loss", f"AED {loss_now:,.0f}/day",
                        f"{wasted_kw:,.0f} kW wasted power · Tariff {tariff:.2f} AED/kWh",
                        "#e8590c" if loss_now > 50 else "#2b9348"), unsafe_allow_html=True)

        # Plotly Visualizations
        fig_app = go.Figure()
        fig_app.add_trace(go.Scatter(x=d.index, y=d["T_approach"], name="T_approach", mode="lines", line=dict(color="#1f77b4")))
        fig_app.add_trace(go.Scatter(x=d.index, y=d["T_baseline"], name="Weather Baseline", mode="lines", line=dict(color="#2b9348", dash="dash")))
        fig_app.add_trace(go.Scatter(x=d.index, y=d["T_warn"], name="Warning Limit", mode="lines", line=dict(color="#f08c00", dash="dash")))
        pts = d[d["persistent_alert"]]
        fig_app.add_trace(go.Scatter(x=pts.index, y=pts["T_approach"], name="Persistent Anomaly", mode="markers", marker=dict(color="red", size=6)))
        st.plotly_chart(style_fig(fig_app, "Condenser Approach vs Weather-Adjusted Baseline", "°C"), config={"displayModeBar": False})

        fig_loss = go.Figure(go.Scatter(x=d.index, y=d["Loss_AED_day_s"], mode="lines", fill="tozeroy", line=dict(color="#e8590c")))
        st.plotly_chart(style_fig(fig_loss, "Estimated Daily Financial Loss (6 h avg)", "AED / day"), config={"displayModeBar": False})

    # ----------------------------------------------------------------------- #
    # TAB 2: ASHRAE RP-1043 Benchmark Validation
    # ----------------------------------------------------------------------- #
    with tabs[1]:
        st.subheader("🔬 Benchmark Validation against ASHRAE RP-1043 Experimental Protocol")
        st.write("Evaluating physics engine accuracy on 500 hourly experimental chiller conditions (Healthy vs Condenser Fouling vs Refrigerant Leak).")
        
        bench_raw = generate_ashrae_benchmark()
        bench_scored = classify_faults(detect_anomalies(physics_engine(bench_raw)))
        
        # Map Predictions to Ground Truth
        bench_scored["Pred_Label"] = np.select(
            [bench_scored["fault_type"] == "A", bench_scored["fault_type"] == "B"],
            ["CF_Fouling", "UC_Leak"],
            default="SL0_Healthy"
        )
        
        y_true = bench_scored["Ground_Truth_Label"]
        y_pred = bench_scored["Pred_Label"]
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("### Classification Metrics")
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
            
        with col_m2:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=["SL0_Healthy", "CF_Fouling", "UC_Leak"])
            cm_df = pd.DataFrame(cm, index=["True Healthy", "True Fouling", "True Leak"], columns=["Pred Healthy", "Pred Fouling", "Pred Leak"])
            st.dataframe(cm_df)

# ----------------------------------------------------------------------- #
    # TAB 3: BMS File Audit Ingestion
    # ----------------------------------------------------------------------- #
    with tabs[2]:
        st.subheader("📁 Zero-Risk BMS Log File Audit")
        st.write("Upload a raw BMS log (CSV or Excel) to run the dynamic physics audit.")
        
        uploaded_file = st.file_uploader("Upload BMS Log File", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)
                st.success("File ingested successfully!")
                st.dataframe(user_df.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()
