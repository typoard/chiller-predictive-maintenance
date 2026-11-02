"""
Centrifugal Chiller Predictive Maintenance Dashboard  -  UAE climate edition
=============================================================================
Run:   streamlit run app.py
Deps:  pip install streamlit pandas numpy scikit-learn plotly

PHYSICS & ARCHITECTURE
----------------------
* HARDWARE-AGNOSTIC: no water flow meter is used anywhere. Required controller points only:
    T_amb, T_cond_out, T_evap_out, P_cond, P_evap, T_discharge, compressor current (A).
* Cooling capacity (TR) is ESTIMATED with a refrigerant-side enthalpy balance:
      W_elec  = sqrt(3) * V * I * PF                      [kW]   from compressor current
      m_ref   = W_elec / (h2 - h1)                        [kg/s] compressor energy balance
      Q_evap  = m_ref * (h1 - h4)                         [kW]   refrigerating effect
      TR      = Q_evap / 3.517
  h1 = sat. vapour @ T_sat_evap, h4 = sat. liquid @ T_sat_cond (isenthalpic expansion),
  h2 = measured discharge state (T_discharge). Hermetic/semi-hermetic drive assumed, so motor heat
  is rejected into the refrigerant and W_elec equals the enthalpy rise m_ref*(h2-h1).
* Condenser approach  T_approach = T_sat_cond(P_cond) - T_cond_out   [degC]
* Specific power     kW/TR = 3.517 * (h2 - h1) / (h1 - h4)
* WEATHER NORMALISATION (ASHRAE Guideline 14 style, dynamic baseline instead of a fixed limit):
      T_baseline(t) = 2.0 + 0.03 * (T_amb(t) - 25)        Warning = T_baseline + 1.5 degC
  The kW/TR baseline is a first-principles "twin": same enthalpy model evaluated with healthy
  approach and nominal isentropic efficiency at the CURRENT condenser/evaporator conditions.
* ML: Isolation Forest trained on the healthy window, fed with WEATHER-NORMALISED residuals
  (the T_amb-adjusted view of [T_approach, kW/TR, T_amb]). Raw T_amb is deliberately not a tree
  input: an out-of-range heat wave would be "isolated" instantly and cause the very false
  positive the normalisation is meant to remove.
* Persistence filter: >= 3 anomaly flags in the last 5 steps. Hard trip: T_approach >= 5.5 degC.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
KW_PER_TR = 3.517                 # 1 TR = 3.517 kW thermal
SAT_A, SAT_B = 15.428, 2662.6     # R-134a fit: ln(P_kPa) = A - B/T_K  (~1 % error, 0-50 degC)
CP_VAP_ISENT = 0.95               # kJ/kg.K, mean suction-vapour cp for isentropic lift estimate
K_ISENT = 1.09                    # effective isentropic exponent of R-134a vapour
CP_SUPERHEAT = 1.15               # kJ/kg.K, superheated vapour at condensing pressure
ETA_IS_NOM = 0.80                 # healthy compressor isentropic efficiency (baseline twin)
EVAP_APPROACH_NOM = 1.5           # K, healthy T_evap_out - T_sat_evap
SUPPLY_V, POWER_FACTOR = 3300.0, 0.90   # MV motor supply (line-line V) and PF
RATED_TR = 1000                   # used ONLY by the simulator to size the plant

HOURS, TRAIN_H, FAULT_H = 21 * 24, 14 * 24, 5 * 24     # 21 d hourly: 14 d healthy, 5 d fault window
WARN_MARGIN_C = 1.5               # dynamic warning = baseline + 1.5 degC
HARD_TRIP_C = 5.5                 # absolute critical trip
LEAK_DROP_C = 1.0                 # Type B: drop below baseline
KW_ELEVATED_PCT = 2.0             # Type B: specific power above baseline
PERSIST_WINDOW, PERSIST_HITS = 5, 3
DEWA_TARIFF = 0.44                # AED/kWh default
SEED = 42

SCENARIOS = ["Normal Operation", "Gradual Tube Fouling",
             "Sudden Refrigerant Leak/Pressure Loss", "Extreme Weather Spike"]


# --------------------------------------------------------------------------- #
# R-134a thermodynamic helpers (compact correlations, adequate for MVP analytics)
# --------------------------------------------------------------------------- #
def p_sat(t_c):
    """Saturation pressure [kPa abs] from saturation temperature [degC]."""
    return np.exp(SAT_A - SAT_B / (np.asarray(t_c, dtype=float) + 273.15))


def t_sat(p_kpa):
    """Saturation temperature [degC] from pressure [kPa abs] (inverse of p_sat)."""
    return SAT_B / (SAT_A - np.log(np.asarray(p_kpa, dtype=float))) - 273.15


def h_liq(t_c):
    """Saturated-liquid enthalpy [kJ/kg], IIR reference (200 kJ/kg at 0 degC)."""
    return 200.0 + 1.34 * t_c + 0.00175 * t_c ** 2


def h_vap(t_c):
    """Saturated-vapour enthalpy [kJ/kg]."""
    return 398.6 + 0.6 * t_c - 0.002 * t_c ** 2


def dh_isentropic(tsc, tse):
    """Isentropic enthalpy rise [kJ/kg]:  cp * T1 * (PR^((k-1)/k) - 1),  PR = P_cond / P_evap."""
    pr = p_sat(tsc) / p_sat(tse)
    return CP_VAP_ISENT * (tse + 273.15) * (pr ** ((K_ISENT - 1) / K_ISENT) - 1)


def ideal_kw_per_tr(tsc, tse, eta):
    """Specific power [kW/TR] of a compressor with isentropic efficiency `eta` between the
    given saturation temperatures:  kW/TR = 3.517 * (dh_is/eta) / (h1 - h4)."""
    return KW_PER_TR * (dh_isentropic(tsc, tse) / eta) / (h_vap(tse) - h_liq(tsc))


# --------------------------------------------------------------------------- #
# 1. Simulator  (raw controller points only - no flow, no capacity)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def simulate_telemetry(scenario: str, amb_lo: float, amb_hi: float, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = HOURS
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq="h")
    i, hour = np.arange(n), idx.hour.values
    f0 = n - FAULT_H                                            # start of fault window

    # Outdoor dry-bulb: user range as diurnal swing (peak 15:00) + slow weather drift
    mid, amp = (amb_lo + amb_hi) / 2, (amb_hi - amb_lo) / 2
    t_amb = (mid + amp * np.sin(2 * np.pi * (hour - 9) / 24)
             + 0.8 * np.sin(2 * np.pi * i / (24 * 6)) + rng.normal(0, 0.3, n))
    if scenario == "Extreme Weather Spike":                     # heat wave in the last 4 days
        t_amb = np.minimum(t_amb + 7.0 * np.clip((i - (n - 96)) / 12, 0, 1), 50.0)

    # Cooling load follows ambient (UAE: load is weather-driven)
    load = np.clip(0.45 + 0.015 * (t_amb - 20) + rng.normal(0, 0.04, n), 0.35, 1.0)

    # Cooling-tower water: wet-bulb ~ f(dry-bulb) + 4 K tower approach
    t_cond_in = 14.0 + 0.35 * t_amb + 4.0 + rng.normal(0, 0.25, n)
    t_cond_out = t_cond_in + 6.2 * load + rng.normal(0, 0.10, n)
    t_evap_out = 6.0 + rng.normal(0, 0.10, n)

    # Fault injection -------------------------------------------------------
    foul = 3.8 * np.clip((i - f0) / FAULT_H, 0, 1) if scenario == "Gradual Tube Fouling" else 0.0
    leak = (i >= f0 + 12) if scenario == "Sudden Refrigerant Leak/Pressure Loss" else np.zeros(n, bool)

    # Healthy approach follows the specified weather curve; fouling adds, leak steps it down
    approach = 2.0 + 0.03 * (t_amb - 25) + rng.normal(0, 0.12, n) + foul - 1.6 * leak
    tsc = t_cond_out + approach                                              # true T_sat_cond
    tse = t_evap_out - EVAP_APPROACH_NOM - 2.0 * leak + rng.normal(0, 0.05, n)  # true T_sat_evap

    # Compressor: true efficiency (leak = poor charge -> degraded efficiency)
    eta = ETA_IS_NOM + rng.normal(0, 0.004, n) - 0.06 * leak
    dh_act = dh_isentropic(tsc, tse) / eta                                   # h2 - h1
    q_ref = h_vap(tse) - h_liq(tsc)                                          # h1 - h4
    q_kw = load * RATED_TR * KW_PER_TR                                       # true capacity
    m_ref = q_kw / q_ref
    w_kw = m_ref * dh_act                                                    # electrical input
    amps = w_kw * 1000 / (np.sqrt(3) * SUPPLY_V * POWER_FACTOR) * (1 + rng.normal(0, 0.004, n))
    t_dis = tsc + (h_vap(tse) + dh_act - h_vap(tsc)) / CP_SUPERHEAT + rng.normal(0, 0.15, n)

    return pd.DataFrame({
        "T_amb": t_amb, "T_cond_out": t_cond_out, "T_evap_out": t_evap_out,
        "P_cond_kPa": p_sat(tsc) * (1 + rng.normal(0, 0.004, n)),
        "P_evap_kPa": p_sat(tse) * (1 + rng.normal(0, 0.004, n)),
        "T_discharge": t_dis, "Compressor_A": amps,
    }, index=pd.Index(idx, name="timestamp"))


# --------------------------------------------------------------------------- #
# 2. Physics engine: capacity estimate, KPIs, weather-adjusted baselines
# --------------------------------------------------------------------------- #
def physics_engine(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["T_sat_cond"], d["T_sat_evap"] = t_sat(d["P_cond_kPa"]), t_sat(d["P_evap_kPa"])

    # Electrical input from compressor current (3-phase): P = sqrt(3) * V * I * PF
    d["kW"] = np.sqrt(3) * SUPPLY_V * d["Compressor_A"] * POWER_FACTOR / 1000

    # Enthalpy balance -> refrigerant mass flow -> cooling capacity (no flow meter)
    h1, h4 = h_vap(d["T_sat_evap"]), h_liq(d["T_sat_cond"])
    h2 = h_vap(d["T_sat_cond"]) + CP_SUPERHEAT * (d["T_discharge"] - d["T_sat_cond"])
    d["m_ref_kg_s"] = d["kW"] / (h2 - h1)
    d["TR_est"] = d["m_ref_kg_s"] * (h1 - h4) / KW_PER_TR
    d["kW_per_TR"] = d["kW"] / d["TR_est"]

    # Condenser approach: T_approach = T_sat_cond - T_cond_out
    d["T_approach"] = d["T_sat_cond"] - d["T_cond_out"]

    # Dynamic weather-adjusted baseline & warning threshold (Dr. Hamdan / ASHRAE G14)
    d["T_baseline"] = 2.0 + 0.03 * (d["T_amb"] - 25.0)
    d["T_warn"] = d["T_baseline"] + WARN_MARGIN_C

    # Healthy specific-power twin at the CURRENT conditions (healthy approach, nominal efficiency)
    d["kWTR_baseline"] = ideal_kw_per_tr(d["T_cond_out"] + d["T_baseline"],
                                         d["T_evap_out"] - EVAP_APPROACH_NOM, ETA_IS_NOM)

    # Weather-normalised residuals (what the ML model actually sees)
    d["Approach_Resid"] = d["T_approach"] - d["T_baseline"]
    d["kWTR_Resid_pct"] = (d["kW_per_TR"] / d["kWTR_baseline"] - 1) * 100

    # Wasted power: Delta_kW = (actual kW/TR - baseline kW/TR) * estimated TR
    d["Delta_kW"] = (d["kW_per_TR"] - d["kWTR_baseline"]) * d["TR_est"]
    return d


# --------------------------------------------------------------------------- #
# 3. ML anomaly detection + persistence filter + fault disambiguation
# --------------------------------------------------------------------------- #
def detect_anomalies(d: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    out = d.copy()
    feats = ["Approach_Resid", "kWTR_Resid_pct"]
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=7)
    model.fit(out.iloc[:TRAIN_H][feats])                         # healthy baseline only
    out["raw_anomaly"] = model.predict(out[feats])               # 1 = normal, -1 = anomaly

    # Persistence filter: >= 3 flags within the last 5 time-steps
    out["anomaly_count"] = (out["raw_anomaly"] == -1).rolling(PERSIST_WINDOW, min_periods=1).sum()
    out["persistent_alert"] = out["anomaly_count"] >= PERSIST_HITS
    return out


def classify_faults(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    r = out["Approach_Resid"].rolling(3, min_periods=1).mean()           # smoothed residual
    approach_s = out["T_approach"].rolling(3, min_periods=1).mean()
    kw_elevated = out["kWTR_Resid_pct"].rolling(3, min_periods=1).mean() > KW_ELEVATED_PCT

    # TYPE A - scale/fouling: gradual positive trend (>0.3 degC rise over 24 h) above warning limit
    rising = r.diff(24) > 0.3
    type_a = out["persistent_alert"] & (approach_s > out["T_warn"]) & rising

    # TYPE B - refrigerant leak: sudden negative STEP (>1 degC within 3 h) inside a below-baseline
    # episode, residual now >1 degC below baseline, specific power still elevated
    neg = r < -0.5
    episode = (neg != neg.shift()).cumsum()
    step_seen = (r.diff(3) <= -LEAK_DROP_C).astype(int).groupby(episode).cummax().astype(bool)
    type_b = out["persistent_alert"] & (r < -LEAK_DROP_C) & kw_elevated & step_seen

    out["fault_type"] = np.select([type_b, type_a], ["B", "A"], default="")
    return out


@st.cache_data(show_spinner=False)
def run_pipeline(scenario: str, amb_lo: float, amb_hi: float, seed: int = SEED) -> pd.DataFrame:
    return classify_faults(detect_anomalies(physics_engine(simulate_telemetry(scenario, amb_lo, amb_hi, seed))))


# --------------------------------------------------------------------------- #
# 4. Financial engine  (AED, DEWA tariff)
# --------------------------------------------------------------------------- #
def add_financials(d: pd.DataFrame, tariff: float) -> pd.DataFrame:
    out = d.copy()
    # Daily Loss [AED/day] = Delta_kW * 24 h * tariff   (only positive = wasted energy is counted)
    out["Loss_AED_day"] = np.clip(out["Delta_kW"], 0, None) * 24 * tariff
    out["Loss_AED_day_s"] = out["Loss_AED_day"].rolling(6, min_periods=1).mean()
    return out


def operational_status(d: pd.DataFrame):
    last = d.iloc[-1]
    if last["fault_type"] == "A":
        return "Fouling Alert", "#e8590c"
    if last["fault_type"] == "B":
        return "Leak Alert", "#c2255c"
    if last["persistent_alert"]:
        return "Watch", "#f08c00"
    return "Normal", "#2b9348"


def build_insight(d: pd.DataFrame, tariff: float):
    """Rule-based recommendation for the latest state -> (streamlit level, message)."""
    last, tail = d.iloc[-1], d.tail(6)
    loss = float(tail["Loss_AED_day"].mean())
    resid = float(d["Approach_Resid"].tail(3).mean())
    kw_pct = float(d["kWTR_Resid_pct"].tail(3).mean())
    if last["fault_type"] == "A":
        slope = float(np.polyfit(np.arange(48), d["T_approach"].tail(48).values, 1)[0] * 24)  # degC/day
        cur = float(d["T_approach"].tail(3).mean())
        days = 14 if slope <= 0.05 else int(np.clip(np.floor((HARD_TRIP_C - cur) / slope), 1, 14))
        return "error" if cur >= HARD_TRIP_C - 1 else "warning", (
            f"TYPE A - Condenser tube scale / fouling: approach is {resid:+.1f} °C vs the weather-adjusted "
            f"baseline ({last['T_baseline']:.1f} °C at {last['T_amb']:.0f} °C ambient), trending {slope:+.2f} °C/day. "
            f"Schedule tube descaling/brushing within {days} days to avoid reaching the {HARD_TRIP_C} °C trip "
            f"and ~AED {loss:,.0f}/day of wasted energy. Check tower water treatment.")
    if last["fault_type"] == "B":
        return "error", (
            f"TYPE B - Potential refrigerant leak / pressure drop: approach fell {abs(resid):.1f} °C below baseline "
            f"while specific power is {kw_pct:+.1f}% above healthy. Verify refrigerant charge and subcooling, leak-test "
            f"joints and seals, and inspect for low evaporator pressure. Current waste ~AED {loss:,.0f}/day.")
    if last["persistent_alert"]:
        return "info", ("Early drift detected by the model (persistent, below classification limits). "
                        "Continue monitoring; no action required yet.")
    return "success", (f"Operating within the weather-adjusted baseline (ambient {last['T_amb']:.0f} °C → expected "
                       f"approach {last['T_baseline']:.2f} °C). Outdoor heat alone does not raise alerts.")


# --------------------------------------------------------------------------- #
# 5. UI helpers
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1200px}
.kpi{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:14px 16px;
     background:rgba(128,128,128,.07);margin-bottom:10px}
.kpi .t{font-size:.74rem;letter-spacing:.05em;text-transform:uppercase;opacity:.72}
.kpi .v{font-size:1.85rem;font-weight:700;line-height:1.2}
.kpi .s{font-size:.82rem;opacity:.78}
@media (max-width:640px){
  .kpi .v{font-size:1.55rem}
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
                      margin=dict(l=8, r=8, t=45, b=8), font=dict(size=12),
                      yaxis_title=ytitle, legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    return fig


def chart_approach(d):
    pts = d[d["persistent_alert"]]                       # ONLY persistent alerts are drawn in red
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["T_approach"], name="T_approach", mode="lines",
                             line=dict(color="#1f77b4", width=1.6)))
    fig.add_trace(go.Scatter(x=d.index, y=d["T_baseline"], name="Weather-adjusted baseline", mode="lines",
                             line=dict(color="#2b9348", width=1.6, dash="dash")))
    fig.add_trace(go.Scatter(x=d.index, y=d["T_warn"], name="Warning (baseline + 1.5 °C)", mode="lines",
                             line=dict(color="#f08c00", width=1.6, dash="dash")))
    fig.add_trace(go.Scatter(x=pts.index, y=pts["T_approach"], name="Persistent anomaly", mode="markers",
                             marker=dict(color="red", size=7)))
    fig.add_hline(y=HARD_TRIP_C, line_dash="dot", line_color="#d62728")
    fig.add_vrect(x0=d.index[0], x1=d.index[TRAIN_H - 1], fillcolor="green", opacity=0.05, line_width=0)
    return style_fig(fig, "Condenser Approach vs Weather-Adjusted Baseline", "°C")


def chart_loss(d):
    fig = go.Figure(go.Scatter(x=d.index, y=d["Loss_AED_day_s"], mode="lines", fill="tozeroy",
                               name="Wasted energy (AED/day)", line=dict(color="#e8590c", width=1.6)))
    return style_fig(fig, "Estimated Daily Financial Loss (6 h avg)", "AED / day")


# --------------------------------------------------------------------------- #
# 6. Streamlit app
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Chiller Health - UAE", page_icon="❄️", layout="wide",
                       initial_sidebar_state="auto")
    st.markdown(CSS, unsafe_allow_html=True)
    banner = st.empty()                      # reserved slot: critical banner always renders at the very top
    st.title("❄️ Chiller Health Monitor")
    st.caption("Centrifugal chiller · UAE climate · flow-meter-free capacity estimation · AED energy loss")

    with st.sidebar:
        st.header("Controls")
        amb_lo, amb_hi = st.slider("Outdoor ambient range (°C)", 20, 48, (30, 44))
        scenario = st.radio("Simulation scenario", SCENARIOS, index=1)
        tariff = st.slider("DEWA tariff (AED/kWh)", 0.20, 0.80, DEWA_TARIFF, 0.01)
        st.caption(f"Training baseline: first {TRAIN_H // 24} days (healthy). "
                   f"Fault window: last {FAULT_H // 24} days.")

    d = add_financials(run_pipeline(scenario, float(amb_lo), float(amb_hi)), tariff)
    last = d.iloc[-1]

    # Critical hard trip - independent of the ML filter
    if last["T_approach"] >= HARD_TRIP_C:
        with banner.container():
            st.error("⚠️ CRITICAL HARD THRESHOLD BREACH: Immediate Physical Inspection Required.")
            st.caption(f"T_approach = {last['T_approach']:.1f} °C ≥ {HARD_TRIP_C} °C trip limit "
                       f"({d.index[-1]:%Y-%m-%d %H:%M}).")

    status, scolor = operational_status(d)
    app_now, base_now, warn_now = (d["T_approach"].tail(3).mean(), last["T_baseline"], last["T_warn"])
    kwtr_now, kwtr_base = d["kW_per_TR"].tail(3).mean(), d["kWTR_baseline"].tail(3).mean()
    loss_now = float(d["Loss_AED_day"].tail(6).mean())
    wasted_kw = float(np.clip(d["Delta_kW"].tail(6), 0, None).mean())
    app_color = "#d62728" if app_now >= warn_now else "#2b9348"

    c1, c2 = st.columns(2)
    c1.markdown(kpi("T_approach vs baseline", f"{app_now:.1f} °C",
                    f"Baseline {base_now:.1f} °C · Warning {warn_now:.1f} °C · Δ {app_now - base_now:+.1f} °C",
                    app_color), unsafe_allow_html=True)
    c2.markdown(kpi("Current kW/TR", f"{kwtr_now:.3f}",
                    f"Baseline {kwtr_base:.3f} · {(kwtr_now / kwtr_base - 1) * 100:+.1f}%",
                    "#e8590c" if kwtr_now / kwtr_base > 1.04 else "#2b9348"), unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    c3.markdown(kpi("Operational status", status,
                    f"Ambient {last['T_amb']:.0f} °C · persistent alerts (24 h): {int(d['persistent_alert'].tail(24).sum())}",
                    scolor), unsafe_allow_html=True)
    c4.markdown(kpi("Daily financial waste", f"AED {loss_now:,.0f}/day",
                    f"{wasted_kw:,.0f} kW wasted · tariff {tariff:.2f} AED/kWh",
                    "#e8590c" if loss_now > 100 else "#2b9348"), unsafe_allow_html=True)

    level, msg = build_insight(d, tariff)
    getattr(st, level)(msg)

    cfg = {"displayModeBar": False}
    st.plotly_chart(chart_approach(d), config=cfg)
    st.plotly_chart(chart_loss(d), config=cfg)

    st.caption(f"Estimated load {last['TR_est']:,.0f} TR from compressor current "
               f"({last['Compressor_A']:.0f} A) + refrigerant enthalpy balance - no flow meter used.")
    with st.expander("Data & model internals"):
        cols = ["T_amb", "T_approach", "T_baseline", "T_warn", "kW", "TR_est", "kW_per_TR", "kWTR_baseline",
                "Delta_kW", "Loss_AED_day", "raw_anomaly", "anomaly_count", "persistent_alert", "fault_type"]
        st.dataframe(d[cols].round(3).tail(200))
        st.download_button("Download dataset (CSV)", d[cols].round(4).to_csv().encode(),
                           "chiller_scored.csv", "text/csv")


if __name__ == "__main__":
    main()
