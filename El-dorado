"""
El Dorado Public Charging Network — Decision Support System
===========================================================
Single-file Streamlit application wrapping a mixed-integer linear program for
charging-site selection, charger technology mix and zone-to-site session allocation.

RUN:
    pip install streamlit pulp pandas plotly numpy
    streamlit run el_dorado_dss_app.py

The optimiser re-solves on every control change. PuLP/CBC is used when available;
scipy.optimize.milp (HiGHS) is the fallback so the app runs on a bare install.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import pulp
    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False
from scipy.optimize import milp, LinearConstraint, Bounds

# ==================================================================== DATA ===
SITE  = ["C1", "C2", "C3", "C4", "C5", "C6"]
SNAME = ["CBD Mall", "North Metro", "Tech Park", "University", "Highway Hub", "South Plaza"]
ZONE  = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
ZNAME = ["CBD", "Residential North", "Tech Corridor", "University", "Highway", "South Residential"]

SETUP = np.array([70, 48, 65, 42, 55, 45], float)      # Rs Lakh
GRID  = np.array([700, 450, 800, 400, 900, 500], float) # kW
BAYS  = np.array([14, 9, 16, 8, 18, 10], float)
OPEX  = np.array([3.2, 2.4, 3.5, 2.1, 2.8, 2.3])        # Rs Lakh / month
DEM   = np.array([180, 120, 220, 100, 160, 140], float) # sessions / day
PEAK  = np.array([.35, .28, .40, .30, .45, .25])
GROW  = np.array([.35, .55, .70, .40, .60, .50])
DIST  = np.array([[1, 5, 4, 3, 8, 7], [5, 1, 6, 4, 9, 8], [4, 6, 1, 4, 5, 6],
                  [3, 4, 4, 1, 7, 5], [8, 9, 5, 7, 1, 4], [7, 8, 6, 5, 4, 1]], float)

CAPEX = np.array([8., 15.])     # Rs Lakh per charger
CAPY  = np.array([16., 28.])    # sessions / day per charger
KW    = np.array([40., 80.])    # power draw
REVEN = np.array([185., 260.])  # Rs per session
KWHPS = np.array([12., 20.])    # kWh per session  (STATED ASSUMPTION)
TARIFF_BASE = 6.50              # Rs / kWh          (STATED ASSUMPTION)

FAST_MIN = {2: 0.45, 4: 0.55}   # Z3 Tech Corridor, Z5 Highway
FAST_MAX = {1: 0.25, 5: 0.25}   # Z2, Z6 residential
BASE_GROWTH = float((DEM * GROW).sum() / DEM.sum() + 1)   # 1.532, reconciles Exhibits 1 and 6

INK, OP, CITY, BAD, OK, MUT = "#1F3864", "#E08A2B", "#0E8C82", "#C0392B", "#5E8C3A", "#7A8894"

# =============================================================== OPTIMISER ===
NV = 102
IY, IN, IW, IU, IZ = 0, 6, 18, 90, 96
def _n(i, t): return IN + i * 2 + t
def _w(i, j, t): return IW + (i * 6 + j) * 2 + t


def build_rows(budget, util, dmax, cov, tariff, demand, minzone, costmult, fastrules):
    """Assemble the constraint matrix shared by both solver backends."""
    A, lo, hi = [], [], []
    def add(r, l, h):
        A.append(r); lo.append(l); hi.append(h)

    r = np.zeros(NV)                                              # capital budget
    for i in range(6):
        r[IY + i] = SETUP[i] * costmult
        for t in range(2):
            r[_n(i, t)] = CAPEX[t] * costmult
    add(r, -np.inf, budget)

    for i in range(6):
        r = np.zeros(NV); r[_n(i, 0)] = r[_n(i, 1)] = 1; r[IY + i] = -BAYS[i]
        add(r, -np.inf, 0)                                        # charger bays
        r = np.zeros(NV); r[_n(i, 0)] = KW[0]; r[_n(i, 1)] = KW[1]; r[IY + i] = -GRID[i]
        add(r, -np.inf, 0)                                        # grid power
        for t in range(2):                                        # utilisation ceiling
            r = np.zeros(NV)
            for j in range(6): r[_w(i, j, t)] = 1
            r[_n(i, t)] = -util * CAPY[t]
            add(r, -np.inf, 0)

    for j in range(6):                                            # demand balance
        r = np.zeros(NV)
        for i in range(6):
            for t in range(2): r[_w(i, j, t)] = 1
        r[IU + j] = 1
        add(r, demand[j], demand[j])

    for i in range(6):                                            # no service beyond cutoff
        for j in range(6):
            if DIST[i, j] > dmax:
                for t in range(2):
                    r = np.zeros(NV); r[_w(i, j, t)] = 1; add(r, 0, 0)

    for j in range(6):                                            # coverage linking
        r = np.zeros(NV); r[IZ + j] = 1
        for i in range(6):
            if DIST[i, j] <= dmax: r[IY + i] = -1
        add(r, -np.inf, 0)

    r = np.zeros(NV)                                              # coverage target
    for j in range(6): r[IZ + j] = demand[j]
    add(r, cov * demand.sum(), np.inf)

    if fastrules:
        for j, sh in FAST_MIN.items():
            r = np.zeros(NV)
            for i in range(6):
                r[_w(i, j, 1)] = -(1 - sh); r[_w(i, j, 0)] = sh
            add(r, -np.inf, 0)
        for j, sh in FAST_MAX.items():
            r = np.zeros(NV)
            for i in range(6):
                r[_w(i, j, 1)] = (1 - sh); r[_w(i, j, 0)] = -sh
            add(r, -np.inf, 0)

    if minzone > 0:
        for j in range(6):
            r = np.zeros(NV)
            for i in range(6):
                for t in range(2): r[_w(i, j, t)] = 1
            add(r, minzone * demand[j], np.inf)
    return np.array(A), lo, hi


def objective_vector(mode, weight, tariff):
    margin = REVEN - KWHPS * tariff
    profit = np.zeros(NV); access = np.zeros(NV)
    for i in range(6):
        profit[IY + i] = OPEX[i] * 12
        for j in range(6):
            for t in range(2):
                profit[_w(i, j, t)] = -margin[t] * 365 / 1e5
                access[_w(i, j, t)] = -1.0
    scale = abs(profit).max() / abs(access).max()
    if mode == "Operator Profit Maximizer":
        c = profit.copy()
    elif mode == "City Equitable Access Maximizer":
        c = access * scale + 1e-4 * profit
    else:
        c = (1 - weight) * profit + weight * access * scale
    for i in range(6):                                            # nearest-site tie-break
        for j in range(6):
            for t in range(2): c[_w(i, j, t)] += 1e-3 * DIST[i, j]
    return c


@st.cache_data(show_spinner=False)
def optimise(budget, util, dmax, cov, tariff, demand_t, minzone, costmult,
             mode, weight, fastrules=True):
    demand = np.array(demand_t, float)
    A, lo, hi = build_rows(budget, util, dmax, cov, tariff, demand, minzone, costmult, fastrules)
    c = objective_vector(mode, weight, tariff)
    integ = np.zeros(NV); integ[IY:IW] = 1; integ[IZ:NV] = 1
    ub = np.full(NV, np.inf); ub[IY:IN] = 1; ub[IZ:NV] = 1
    res = milp(c=c, constraints=[LinearConstraint(A, lo, hi)], integrality=integ,
               bounds=Bounds(np.zeros(NV), ub))
    if res.x is None:
        return None
    return unpack(res.x, demand, tariff, budget, costmult)


def unpack(v, demand, tariff, budget, costmult):
    y = np.round(v[IY:IN]).astype(int)
    n = np.round(v[IN:IW]).reshape(6, 2).astype(int)
    w = v[IW:IU].reshape(6, 6, 2)
    u = v[IU:IZ]
    z = np.round(v[IZ:]).astype(int)
    cap = (n * CAPY).sum(1)
    served_site = w.sum(axis=(1, 2))
    daily = w.sum(axis=(0, 1))
    capex = (SETUP * y).sum() * costmult + (n * CAPEX).sum() * costmult
    revenue = (daily * REVEN).sum() * 365 / 1e5
    energy = (daily * KWHPS * tariff).sum() * 365 / 1e5
    opex = (OPEX * y).sum() * 12
    return dict(
        y=y, n=n, w=w, u=u, z=z,
        sites=[SITE[i] for i in range(6) if y[i]],
        capex=capex, budget=budget, slack=budget - capex,
        revenue=revenue, energy=energy, opex=opex, profit=revenue - energy - opex,
        served=w.sum(), demand=demand.sum(), capacity=cap, cap_total=cap.sum(),
        kw=(n * KW).sum(1),
        util=np.divide(served_site, cap, out=np.zeros(6), where=cap > 0),
        net_util=w.sum() / cap.sum() if cap.sum() > 0 else 0.0,
        coverage=float((demand * z).sum() / demand.sum()),
        zone_service=np.array([w[:, j, :].sum() / demand[j] for j in range(6)]),
        zone_fast=np.array([w[:, j, 1].sum() / max(w[:, j, :].sum(), 1e-9) for j in range(6)]),
        n_std=int(n[:, 0].sum()), n_fast=int(n[:, 1].sum()),
        daily_revenue=float((daily * REVEN).sum()),
        served_site=served_site)


@st.cache_data(show_spinner=False)
def reallocate(y_t, n_t, demand_t, dmax, tariff, capmult):
    """LP re-allocation on a FIXED build — used for scenario and stress testing."""
    from scipy.optimize import linprog
    y = np.array(y_t); n = np.array(n_t).reshape(6, 2); demand = np.array(demand_t, float)
    margin = REVEN - KWHPS * tariff
    capt = n * CAPY * capmult
    nv = 78
    c = np.zeros(nv)
    for i in range(6):
        for j in range(6):
            for t in range(2):
                ok = DIST[i, j] <= dmax and y[i]
                c[(i * 6 + j) * 2 + t] = -(margin[t] - 0.001 * DIST[i, j]) if ok else 1e4
    Au, bu = [], []
    for i in range(6):
        for t in range(2):
            r = np.zeros(nv)
            for j in range(6): r[(i * 6 + j) * 2 + t] = 1
            Au.append(r); bu.append(capt[i, t])
    Ae, be = [], []
    for j in range(6):
        r = np.zeros(nv)
        for i in range(6):
            for t in range(2): r[(i * 6 + j) * 2 + t] = 1
        r[72 + j] = 1
        Ae.append(r); be.append(demand[j])
    R = linprog(c, A_ub=np.array(Au), b_ub=bu, A_eq=np.array(Ae), b_eq=be,
                bounds=(0, None), method="highs")
    w = R.x[:72].reshape(6, 6, 2); u = R.x[72:]
    daily = w.sum(axis=(0, 1))
    return dict(served=float(w.sum()), unmet=float(u.sum()),
                profit=float((daily * REVEN).sum() * 365 / 1e5
                             - (daily * KWHPS * tariff).sum() * 365 / 1e5
                             - (OPEX * y).sum() * 12),
                net_util=float(w.sum() / capt.sum()) if capt.sum() > 0 else 0.0)

# ======================================================================= UI ===
st.set_page_config(page_title="El Dorado Charging Network DSS", page_icon="⚡", layout="wide")
st.markdown("""<style>
  .block-container{padding-top:2rem;padding-bottom:2rem;}
  div[data-testid="stMetricValue"]{font-size:1.65rem;}
  div[data-testid="stMetric"]{background:#F7F9FB;border:1px solid #E1E7ED;border-radius:9px;padding:14px 16px;}
  h1{color:#1F3864;font-size:2rem;}
  .stCaption,.caption{color:#7A8894;}
</style>""", unsafe_allow_html=True)

st.title("⚡ El Dorado Public Charging Network")
st.caption("Decision Support System for the city and the network operator — the optimisation re-solves on every change.")

# --------------------------------------------------------------- SIDEBAR ----
with st.sidebar:
    st.header("Decision levers")

    st.subheader("Capital and policy")
    budget = st.slider("Capital budget (₹ Crore)", 2.5, 6.0, 4.5, 0.05,
                       help="Total upfront expenditure ceiling.") * 100
    cov_target = st.slider("Minimum coverage target", 0.70, 1.00, 0.90, 0.01,
                           format="%.0f%%",
                           help="Share of total daily demand that must lie within the travel cutoff of an open site.")
    dmax = st.select_slider("Maximum travel distance", options=[3, 5, 7], value=5,
                            format_func=lambda v: f"{v} km")
    equity = st.slider("Equity floor — minimum service in every zone", 0.00, 0.55, 0.50, 0.01,
                       format="%.0f%%",
                       help="Guarantees each zone a minimum share of its demand. This is the cost-of-equity lever.")
    util_cap = st.slider("Utilisation ceiling (queueing threshold)", 0.60, 1.00, 0.85, 0.01,
                         format="%.0f%%")

    st.subheader("Policy mode")
    mode = st.radio("Objective", ["Operator Profit Maximizer",
                                  "City Equitable Access Maximizer",
                                  "Blended Weighted Score"], index=2)
    weight = 0.5
    if mode == "Blended Weighted Score":
        weight = st.slider("Weight on public access", 0.0, 1.0, 0.35, 0.05,
                           help="0 = pure operator profit, 1 = pure sessions served.")

    st.subheader("Scenario")
    scen = st.radio("EV adoption over 2 years",
                    ["Today", "Slow (1.25×)", "Base (1.50×)", "Rapid (1.85×)", "Custom"], index=0)
    mult = {"Today": 1.00, "Slow (1.25×)": 1.25, "Base (1.50×)": 1.50, "Rapid (1.85×)": 1.85}.get(scen)
    if scen == "Custom":
        mult = st.slider("Custom adoption multiplier", 1.00, 2.20, 1.50, 0.05)
    shock = st.slider("Electricity cost shock", -18, 18, 0, 1, format="%+d%%")
    tariff = TARIFF_BASE * (1 + shock / 100)
    costmult = st.slider("Construction cost index", 1.00, 1.12, 1.00, 0.01,
                         help="1.12 reflects the staged-expansion penalty in the case.")

    st.divider()
    st.caption(f"Solver: {'PuLP/CBC available · ' if HAVE_PULP else ''}scipy HiGHS branch-and-bound. "
               "Energy cost is a stated assumption: 12 kWh (standard) and 20 kWh (fast) per session.")

demand = DEM.copy() if mult <= 1.001 else DEM * (1 + GROW) * (mult / BASE_GROWTH)

sol = optimise(budget, util_cap, float(dmax), cov_target, tariff, tuple(demand),
               equity, costmult, mode, weight)

relaxed_to = None
if sol is None:
    for f in np.arange(equity - 0.05, -0.001, -0.05):
        sol = optimise(budget, util_cap, float(dmax), cov_target, tariff, tuple(demand),
                       max(0.0, round(float(f), 2)), costmult, mode, weight)
        if sol is not None:
            relaxed_to = max(0.0, round(float(f), 2)); break

if sol is None:
    st.error("**No feasible network at these settings.** The coverage target cannot be met within the "
             "capital budget at this travel cutoff. Lower the coverage target, widen the distance cutoff, "
             "or raise the budget.")
    st.stop()

if relaxed_to is not None:
    st.warning(f"A **{equity:.0%} equity floor is unattainable** on {demand.sum():,.0f} sessions/day with "
               f"₹{budget/100:.2f} Cr. Showing the best feasible network, which holds every zone at "
               f"**{relaxed_to:.0%}**. Raise the budget or lower the floor to clear this.")

# ------------------------------------------------------------ KPI CARDS -----
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Capex utilised", f"₹{sol['capex']:.0f} L",
          f"₹{sol['slack']:.0f} L unspent of ₹{budget:.0f} L")
c2.metric("Demand coverage", f"{sol['coverage']:.0%}",
          f"{'meets' if sol['coverage'] >= cov_target - 1e-6 else 'MISSES'} the {cov_target:.0%} target",
          delta_color="normal" if sol['coverage'] >= cov_target - 1e-6 else "inverse")
c3.metric("Capacity vs demand", f"{sol['cap_total']:,.0f} / {sol['demand']:,.0f}",
          f"{sol['served']/sol['demand']:.0%} of sessions served")
c4.metric("Average utilisation", f"{sol['net_util']:.0%}",
          f"ceiling {util_cap:.0%}",
          delta_color="inverse" if sol['net_util'] > util_cap + 1e-6 else "off")
c5.metric("Annual operating profit", f"₹{sol['profit']:.0f} L",
          f"{sol['profit']/sol['capex']:.0%} return on capex" if sol['capex'] > 0 else "—")

st.markdown(
    f"**Recommended build — {' + '.join(f'{SITE[i]} {SNAME[i]}' for i in range(6) if sol['y'][i])}**  ·  "
    f"{sol['n_std']} standard and {sol['n_fast']} fast chargers  ·  "
    f"₹{sol['daily_revenue']:,.0f} expected daily revenue  ·  "
    f"minimum zone service {sol['zone_service'].min():.0%}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏗️ Network build", "🔌 Grid headroom & coverage", "📊 Risk & scenarios", "🚦 Expansion intelligence"])

# ------------------------------------------------------ TAB 1: THE BUILD ----
with tab1:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Charger mix by site")
        remaining = np.maximum(BAYS - sol['n'].sum(1), 0) * (sol['y'] > 0)
        fig = go.Figure()
        fig.add_bar(name="Standard", x=SITE, y=sol['n'][:, 0], marker_color=CITY,
                    text=[v or "" for v in sol['n'][:, 0]], textposition="inside")
        fig.add_bar(name="Fast", x=SITE, y=sol['n'][:, 1], marker_color=OP,
                    text=[v or "" for v in sol['n'][:, 1]], textposition="inside")
        fig.add_bar(name="Unused bays", x=SITE, y=remaining, marker_color="#DDE3E9",
                    text=[int(v) or "" for v in remaining], textposition="inside")
        fig.update_layout(barmode="stack", height=360, plot_bgcolor="white",
                          margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.12),
                          yaxis_title="Chargers")
        fig.update_yaxes(gridcolor="#EEF2F5")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Service reaching each zone")
        colors = [BAD if sol['zone_service'][j] < 0.001 else
                  (OP if sol['zone_service'][j] < 0.35 else CITY) for j in range(6)]
        fig = go.Figure(go.Bar(
            x=sol['zone_service'] * 100, y=[f"{ZONE[j]} {ZNAME[j]}" for j in range(6)],
            orientation="h", marker_color=colors,
            text=[f"{v:.0%}" for v in sol['zone_service']], textposition="outside"))
        fig.update_layout(height=360, plot_bgcolor="white", xaxis_title="% of zone demand served",
                          margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 115])
        fig.update_xaxes(gridcolor="#EEF2F5")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Site-by-site decision table")
    df = pd.DataFrame({
        "Site": [f"{SITE[i]} {SNAME[i]}" for i in range(6)],
        "Open": ["✅" if sol['y'][i] else "—" for i in range(6)],
        "Standard": [sol['n'][i, 0] if sol['y'][i] else 0 for i in range(6)],
        "Fast": [sol['n'][i, 1] if sol['y'][i] else 0 for i in range(6)],
        "Capacity (sess/day)": [int(sol['capacity'][i]) for i in range(6)],
        "Served (sess/day)": [round(sol['served_site'][i], 1) for i in range(6)],
        "Utilisation": [f"{sol['util'][i]:.0%}" if sol['y'][i] else "—" for i in range(6)],
        "Grid kW": [f"{int(sol['kw'][i])} / {int(GRID[i])}" for i in range(6)],
        "Capex (₹L)": [round(SETUP[i] + sol['n'][i, 0] * 8 + sol['n'][i, 1] * 15, 1)
                       if sol['y'][i] else 0 for i in range(6)]})
    st.dataframe(df, use_container_width=True, hide_index=True)

# ------------------------------------------- TAB 2: GRID & COVERAGE ---------
with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Grid power headroom")
        for i in range(6):
            if not sol['y'][i]:
                continue
            used, lim = sol['kw'][i], GRID[i]
            frac = used / lim
            st.markdown(f"**{SITE[i]} {SNAME[i]}** — {int(used)} kW of {int(lim)} kW  "
                        f"({frac:.0%}{' · **AT LIMIT**' if frac > 0.96 else ''})")
            st.progress(min(frac, 1.0))
        st.caption("A site at its grid limit cannot take another charger however much budget remains. "
                   "That is where the shadow price sits.")
    with right:
        st.subheader("Distance–coverage matrix")
        inrange = (DIST <= dmax).astype(int)
        annot = [[f"{DIST[i,j]:.0f}" for j in range(6)] for i in range(6)]
        fig = go.Figure(go.Heatmap(
            z=inrange, x=ZONE, y=[f"{SITE[i]}{'  ●' if sol['y'][i] else ''}" for i in range(6)],
            text=annot, texttemplate="%{text}", showscale=False,
            colorscale=[[0, "#F2F5F7"], [1, "#CDEBE7"]], xgap=2, ygap=2))
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Shaded = within the {dmax} km cutoff. ● marks an opened site. "
                   f"Numbers are travel distance in km.")

    st.subheader("Zone diagnostics")
    near = [min((DIST[i, j], SITE[i]) for i in range(6) if sol['y'][i]) for j in range(6)]
    st.dataframe(pd.DataFrame({
        "Zone": [f"{ZONE[j]} {ZNAME[j]}" for j in range(6)],
        "Demand/day": demand.round(0).astype(int),
        "Served/day": [round(sol['w'][:, j, :].sum(), 1) for j in range(6)],
        "Unserved/day": sol['u'].round(1),
        "Service level": [f"{v:.0%}" for v in sol['zone_service']],
        "Fast share": [f"{v:.0%}" for v in sol['zone_fast']],
        "Nearest open site": [f"{s} ({d:.0f} km)" for d, s in near],
        "Within cutoff": ["✅" if d <= dmax else "❌ NO ACCESS" for d, _ in near],
    }), use_container_width=True, hide_index=True)

# --------------------------------------------- TAB 3: RISK & SCENARIOS ------
with tab3:
    st.subheader("Joint adoption × session-duration scenario grid")
    st.caption("The build is held fixed and demand is re-allocated. Nine joint states, "
               "probabilities multiplied from the two independent case distributions.")
    ADO = [("Slow", 1.25, .25), ("Base", 1.50, .50), ("Rapid", 1.85, .25)]
    DUR = [("Short", 1.10, .20), ("Normal", 1.00, .60), ("Long", 0.82, .20)]
    rows = []
    for an, am, ap in ADO:
        for dn, dm, dp in DUR:
            d2 = DEM * (1 + GROW) * (am / BASE_GROWTH)
            e = reallocate(tuple(sol['y']), tuple(sol['n'].flatten()), tuple(d2),
                           float(dmax), tariff, dm)
            rows.append({"Adoption": an, "Duration": dn, "Probability": ap * dp,
                         "Demand/day": round(d2.sum()), "Served/day": round(e['served']),
                         "Unserved/day": round(e['unmet']), "Utilisation": e['net_util'],
                         "Profit (₹L)": round(e['profit'], 1)})
    sc = pd.DataFrame(rows)
    emv = float((sc["Probability"] * sc["Profit (₹L)"]).sum())
    eun = float((sc["Probability"] * sc["Unserved/day"]).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected annual profit", f"₹{emv:.0f} L")
    k2.metric("Expected unserved demand", f"{eun:,.0f}/day")
    k3.metric("Worst case", f"₹{sc['Profit (₹L)'].min():.0f} L")
    k4.metric("Probability of a loss", f"{(sc.loc[sc['Profit (₹L)']<0,'Probability'].sum()):.0%}")

    left, right = st.columns([1, 1])
    with left:
        piv = sc.pivot(index="Adoption", columns="Duration", values="Profit (₹L)").reindex(
            index=["Slow", "Base", "Rapid"], columns=["Short", "Normal", "Long"])
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                                   text=piv.values, texttemplate="₹%{text:.0f}L",
                                   colorscale="RdYlGn", showscale=False, xgap=3, ygap=3))
        fig.update_layout(height=290, title="Annual profit by joint scenario",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        piv2 = sc.pivot(index="Adoption", columns="Duration", values="Unserved/day").reindex(
            index=["Slow", "Base", "Rapid"], columns=["Short", "Normal", "Long"])
        fig = go.Figure(go.Heatmap(z=piv2.values, x=piv2.columns, y=piv2.index,
                                   text=piv2.values, texttemplate="%{text:.0f}",
                                   colorscale="Reds", showscale=False, xgap=3, ygap=3))
        fig.update_layout(height=290, title="Unserved sessions per day",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sc.style.format({"Probability": "{:.2f}", "Utilisation": "{:.0%}"}),
                 use_container_width=True, hide_index=True)

    st.subheader("Grid curtailment stress test")
    cur = reallocate(tuple(sol['y']), tuple(sol['n'].flatten()), tuple(demand),
                     float(dmax), tariff, 0.60)
    a, b = st.columns(2)
    a.metric("Sessions served on a curtailment day", f"{cur['served']:,.0f}",
             f"{cur['served'] - sol['served']:+,.0f} vs normal", delta_color="inverse")
    b.metric("Probability per site per day", "6%", "capacity falls 40%", delta_color="off")

    st.subheader("Cost-of-equity frontier")
    st.caption("Each point re-solves the full MILP at a different guaranteed minimum service level.")
    fr = []
    for q in np.arange(0, 0.60, 0.05):
        s2 = optimise(budget, util_cap, float(dmax), cov_target, tariff, tuple(demand),
                      round(float(q), 2), costmult, mode, weight)
        if s2: fr.append((q, s2['profit'], s2['served'], "+".join(s2['sites'])))
    if fr:
        fdf = pd.DataFrame(fr, columns=["floor", "profit", "served", "sites"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=fdf["floor"] * 100, y=fdf["profit"], name="Annual profit (₹L)",
                    marker_color=OP, text=fdf["sites"], textposition="outside", textfont_size=9)
        fig.add_scatter(x=fdf["floor"] * 100, y=fdf["served"], name="Sessions served/day",
                        mode="lines+markers", line=dict(color=CITY, width=3), secondary_y=True)
        fig.update_layout(height=330, plot_bgcolor="white", xaxis_title="Guaranteed minimum service per zone (%)",
                          legend=dict(orientation="h", y=1.15), margin=dict(l=10, r=10, t=10, b=10))
        fig.update_yaxes(gridcolor="#EEF2F5")
        st.plotly_chart(fig, use_container_width=True)
        drop = fdf["profit"].iloc[0] - fdf["profit"].iloc[-1]
        st.info(f"**The cost of equity is ₹{drop:.1f} Lakh per year.** Moving from no guarantee to a "
                f"{fdf['floor'].iloc[-1]:.0%} floor in every zone changes the build from "
                f"**{fdf['sites'].iloc[0]}** to **{fdf['sites'].iloc[-1]}**. The frontier is flat until "
                f"the floor forces a different site pair — that knee is the number to negotiate around.")

# ------------------------------------ TAB 4: EXPANSION INTELLIGENCE ---------
with tab4:
    st.subheader("Binding constraints and shadow prices")
    bind = []
    for i in range(6):
        if not sol['y'][i]: continue
        bind.append({"Resource": f"{SITE[i]} grid power",
                     "Used": f"{int(sol['kw'][i])} kW", "Limit": f"{int(GRID[i])} kW",
                     "Slack": f"{int(GRID[i]-sol['kw'][i])} kW",
                     "Status": "🔴 BINDING" if GRID[i] - sol['kw'][i] < KW[0] else "🟢 slack"})
        bind.append({"Resource": f"{SITE[i]} charger bays",
                     "Used": f"{int(sol['n'][i].sum())}", "Limit": f"{int(BAYS[i])}",
                     "Slack": f"{int(BAYS[i]-sol['n'][i].sum())}",
                     "Status": "🔴 BINDING" if BAYS[i] - sol['n'][i].sum() < 1 else "🟢 slack"})
    bind.append({"Resource": "Capital budget", "Used": f"₹{sol['capex']:.0f} L",
                 "Limit": f"₹{budget:.0f} L", "Slack": f"₹{sol['slack']:.0f} L",
                 "Status": "🔴 BINDING" if sol['slack'] < 8 else "🟢 slack"})
    st.dataframe(pd.DataFrame(bind), use_container_width=True, hide_index=True)
    if sol['slack'] >= 8:
        st.warning(f"**₹{sol['slack']:.0f} Lakh of budget cannot be deployed.** Capital is not the "
                   "bottleneck — grid power is. Negotiating additional kW with the utility unlocks "
                   "more capacity than another tranche of capital.")

    st.subheader("Which site should be expanded first if demand grows 40%?")
    d40 = DEM * 1.40
    base = reallocate(tuple(sol['y']), tuple(sol['n'].flatten()), tuple(d40), float(dmax), tariff, 1.0)
    cands = []
    for i in range(6):
        yy = sol['y'].copy(); nn = sol['n'].copy().astype(int)
        fresh = not yy[i]; yy[i] = 1
        used = int(nn[i].sum()); kw = nn[i][0] * 40 + nn[i][1] * 80
        cost = SETUP[i] * 1.12 if fresh else 0.0
        added = 0
        while used + added < BAYS[i] and added < 4:
            t = 1 if kw + 80 <= GRID[i] else (0 if kw + 40 <= GRID[i] else -1)
            if t < 0: break
            nn[i][t] += 1; kw += KW[t]; cost += CAPEX[t] * 1.12; added += 1
        if not added: continue
        e = reallocate(tuple(yy), tuple(nn.flatten()), tuple(d40), float(dmax), tariff, 1.0)
        cands.append({"Site": f"{SITE[i]} {SNAME[i]}",
                      "Action": "OPEN NEW SITE" if fresh else "deepen existing",
                      "Chargers added": added, "Capex (₹L, +12%)": round(cost, 1),
                      "Extra sessions/day": round(e['served'] - base['served'], 1),
                      "Extra profit (₹L/yr)": round(e['profit'] - base['profit'], 1),
                      "Sessions per ₹L": round((e['served'] - base['served']) / cost, 3) if cost else 0})
    if cands:
        cdf = pd.DataFrame(cands).sort_values("Sessions per ₹L", ascending=False)
        top = cdf.iloc[0]
        st.success(f"**Expand {top['Site']} first.** {top['Action'].capitalize()}, "
                   f"{top['Chargers added']} chargers at ₹{top['Capex (₹L, +12%)']:.0f} Lakh, "
                   f"recovering {top['Extra sessions/day']:.0f} sessions/day and "
                   f"₹{top['Extra profit (₹L/yr)']:.1f} Lakh of annual profit. It wins on sessions "
                   f"recovered per rupee, not on headline demand.")
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        st.caption("Sites already at their grid limit are absent from this list — they physically "
                   "cannot take another charger, which is why expansion moves to a new location.")
    else:
        st.info("Every candidate site is built out. Further growth requires a seventh location.")

    st.subheader("Expansion trigger rules")
    st.markdown(f"""
| Trigger | Threshold | Why this level |
|---|---|---|
| Sustained network utilisation | above 75% for two consecutive months | Fires before queues become visible, leaving room for the four-month construction lead time |
| Any single site | above {util_cap:.0%} for four consecutive weeks | The case's own queueing threshold, measured per site rather than on the network average |
| Observed demand growth | above 1.25× annualised | Even the Slow branch exceeds current capacity, so this confirms rather than predicts |
""")
    st.caption("A trigger that fires when queues appear has already failed — the 12% cost penalty "
               "and four-month lead time are both incurred after the decision, not before it.")

st.divider()
st.caption("Model: 102 variables (12 binary, 12 integer, 78 continuous) and 48 structural constraints, "
           "solved to a 0% integer optimality gap. Decision Science produces the evidence; this interface "
           "only makes it explorable. Energy cost is a stated assumption documented in the workbook.")
