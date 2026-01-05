import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit.components.v1 import html

st.set_page_config(page_title="IoT–Edge–Cloud Continuum Simulator", layout="wide")

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.running = False
    st.session_state.interval_sec = 1.5
    st.session_state.step = 0
    st.session_state.service_location = "edge"
    st.session_state.scenarios = {
        "latency_spike": False,
        "edge_failure": False,
        "llm_load": False,
        "energy_saving": False,
    }
    st.session_state.events = []
    st.session_state.llm_history = []
    st.session_state.active_edges = []
    st.session_state.active_edges_t = 0
    st.session_state.llm_intent = None
    st.session_state.llm_plan = None
    st.session_state.nodes_status = {
        "Cloud Orchestrator": "ok",
        "LLM Intent": "ok",
        "LLM Reason": "ok",
        "Management UI": "ok",
        "AUTH": "ok",
        "gRPC Hub": "ok",
        "Monitoring": "ok",
        "Migration": "ok",
        "Service Registry": "ok",
        "Data Storage": "ok",
        "Knowledge": "ok",
        "Extreme A": "ok",
        "Extreme B": "ok",
        "Extreme C": "ok",
    }
    st.session_state.metrics = pd.DataFrame([
        {
            "t": 0,
            "cpu_cloud": 35.0,
            "cpu_edge": 45.0,
            "cpu_ext": 55.0,
            "latency": 60.0,
            "throughput": 100.0,
            "energy_kwh": 1.2,
            "sla": 99.0,
        }
    ])


def log_event(msg):
    st.session_state.events.append({"t": st.session_state.step, "event": msg})


def set_active_edges(edge_list):
    canonical = []
    for a, b in edge_list:
        if (a, b) in edges:
            canonical.append((a, b))
        elif (b, a) in edges:
            canonical.append((b, a))
    st.session_state.active_edges = canonical
    st.session_state.active_edges_t = st.session_state.step


def migrate_service(target):
    if st.session_state.service_location != target:
        log_event(f"Service migration: {st.session_state.service_location} -> {target}")
        st.session_state.service_location = target
        path = [("Cloud Orchestrator", "Migration"), ("Cloud Orchestrator", "Service Registry"), ("Cloud Orchestrator", "gRPC Hub")]
        if target == "ext":
            path.append(("Cloud Orchestrator", "Extreme B"))
        elif target == "cloud":
            path = [("Management UI", "Cloud Orchestrator"), ("LLM Reason", "Cloud Orchestrator")]
        set_active_edges(path)


def simulate_step():
    df = st.session_state.metrics
    last = df.iloc[-1]
    step = st.session_state.step + 1
    random.seed(step)
    np.random.seed(step)
    prev_loc = st.session_state.service_location

    cpu_cloud = max(5, min(98, float(last.cpu_cloud + np.random.normal(0, 2))))
    cpu_edge = max(5, min(98, float(last.cpu_edge + np.random.normal(0, 2))))
    cpu_ext = max(5, min(98, float(last.cpu_ext + np.random.normal(0, 3))))
    latency = max(10, min(800, float(last.latency + np.random.normal(0, 5))))
    throughput = max(10, float(last.throughput + np.random.normal(0, 2)))
    energy = max(0.1, float(last.energy_kwh + np.random.normal(0, 0.05)))
    sla = max(80, min(100, float(last.sla + np.random.normal(0, 0.05))))

    if st.session_state.scenarios["latency_spike"]:
        latency += 80 + abs(np.random.normal(0, 25))
        sla -= 0.2
    if st.session_state.scenarios["llm_load"]:
        cpu_cloud += 12 + abs(np.random.normal(0, 6))
        throughput += 8
    if st.session_state.scenarios["edge_failure"]:
        st.session_state.nodes_status["Extreme B"] = "down"
        cpu_edge += 6
        cpu_ext += 10
        latency += 40
        sla -= 0.5
    else:
        st.session_state.nodes_status["Extreme B"] = "ok"
    if st.session_state.scenarios["energy_saving"]:
        cpu_ext -= 6
        energy -= 0.1

    loc = st.session_state.service_location
    if loc == "ext":
        cpu_ext += 8
        latency += 10
        energy += 0.05
    elif loc == "edge":
        cpu_edge += 6
        latency -= 10
        energy += 0.03
    elif loc == "cloud":
        cpu_cloud += 5
        latency += 5
        energy += 0.02

    if st.session_state.scenarios["edge_failure"] and st.session_state.service_location == "ext":
        migrate_service("edge")
    if (cpu_ext > 85 or latency > 200) and st.session_state.service_location == "ext":
        migrate_service("edge")
    if (cpu_edge > 88 or st.session_state.scenarios["energy_saving"]) and st.session_state.service_location == "edge":
        if cpu_cloud < 85:
            migrate_service("cloud")
    if (cpu_edge < 55 and latency < 120) and st.session_state.service_location == "cloud" and not st.session_state.scenarios["llm_load"]:
        migrate_service("edge")
    if (cpu_ext < 60 and latency < 100) and st.session_state.service_location == "edge" and not st.session_state.scenarios["latency_spike"]:
        migrate_service("ext")

    row = {
        "t": step,
        "cpu_cloud": round(cpu_cloud, 2),
        "cpu_edge": round(cpu_edge, 2),
        "cpu_ext": round(cpu_ext, 2),
        "latency": round(latency, 2),
        "throughput": round(throughput, 2),
        "energy_kwh": round(energy, 3),
        "sla": round(sla, 3),
    }
    st.session_state.metrics = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    st.session_state.step = step
    # Pulse coordination edges if no migration occurred this step
    if st.session_state.service_location == prev_loc:
        base_edges = [("Cloud Orchestrator", "Monitoring"), ("Cloud Orchestrator", "gRPC Hub")]
        if st.session_state.service_location == "ext":
            base_edges.append(("Cloud Orchestrator", "Extreme B"))
        set_active_edges(base_edges)


def classify_intent(text: str):
    t = (text or "").lower()
    intent = {"objective": "balance_cost_latency", "priority": "normal"}
    if any(k in t for k in ["latency", "fast", "quick", "response"]):
        intent["objective"] = "minimize_latency"
    elif any(k in t for k in ["fail", "outage", "down", "recover", "resilience"]):
        intent["objective"] = "failover"
    elif any(k in t for k in ["llm", "inference", "model", "gpt", "reasoning"]):
        intent["objective"] = "scale_llm"
    elif any(k in t for k in ["energy", "battery", "power", "consumption"]):
        intent["objective"] = "reduce_energy"
    return intent


def reason_plan(intent: dict, last_row: pd.Series):
    obj = (intent or {}).get("objective", "balance_cost_latency")
    plan = {"target_placement": None, "scenario_toggles": {}, "steps": []}
    if obj == "minimize_latency":
        plan["steps"] = ["Optimize for latency via near-device placement."]
        target = "ext" if float(last_row.latency) > 100 and float(last_row.cpu_ext) < 80 else "edge"
        plan["target_placement"] = target
    elif obj == "failover":
        plan["steps"] = ["Handle failure and shift workloads to available tier."]
        target = "cloud" if st.session_state.nodes_status.get("Extreme B") == "down" else "edge"
        plan["target_placement"] = target
        plan["scenario_toggles"] = {"edge_failure": True}
    elif obj == "scale_llm":
        plan["steps"] = ["Scale LLM on cloud and allocate capacity."]
        plan["target_placement"] = "cloud"
        plan["scenario_toggles"] = {"llm_load": True}
    elif obj == "reduce_energy":
        plan["steps"] = ["Reduce energy via cloud consolidation and duty cycling."]
        plan["target_placement"] = "cloud"
        plan["scenario_toggles"] = {"energy_saving": True}
    else:
        plan["steps"] = ["Balance cost and latency via edge placement."]
        plan["target_placement"] = "edge"
    return plan


layer_map = {
    "Cloud Orchestrator": "cloud",
    "LLM Intent": "cloud",
    "LLM Reason": "cloud",
    "Management UI": "cloud",
    "AUTH": "cloud",
    "gRPC Hub": "cloud",
    "Monitoring": "cloud",
    "Migration": "edge",
    "Service Registry": "cloud",
    "Data Storage": "cloud",
    "Knowledge": "cloud",
    "Extreme A": "ext",
    "Extreme B": "ext",
    "Extreme C": "ext",
}

positions = {
    "LLM Intent": (-1.8, 3.7),
    "LLM Reason": (-0.6, 3.7),
    "Management UI": (1.6, 3.7),
    "AUTH": (2.7, 3.7),
    "Cloud Orchestrator": (0.0, 3.0),
    "gRPC Hub": (0.0, 2.2),
    "Monitoring": (1.6, 2.5),
    "Migration": (-1.6, 2.5),
    "Service Registry": (-2.6, 2.5),
    "Data Storage": (2.7, 2.5),
    "Knowledge": (-3.5, 2.5),
    "Extreme A": (-1.6, 0.2),
    "Extreme B": (0.0, 0.2),
    "Extreme C": (1.6, 0.2),
}

edges = [
    ("LLM Intent", "Cloud Orchestrator"),
    ("LLM Reason", "Cloud Orchestrator"),
    ("Management UI", "Cloud Orchestrator"),
    ("AUTH", "Management UI"),
    ("Cloud Orchestrator", "gRPC Hub"),
    ("Cloud Orchestrator", "Monitoring"),
    ("Cloud Orchestrator", "Migration"),
    ("Cloud Orchestrator", "Service Registry"),
    ("Cloud Orchestrator", "Data Storage"),
    ("Cloud Orchestrator", "Knowledge"),
    ("Cloud Orchestrator", "Extreme A"),
    ("Cloud Orchestrator", "Extreme B"),
    ("Cloud Orchestrator", "Extreme C"),
]

colors = {
    "cloud": "#1f77b4",
    "edge": "#2ca02c",
    "ext": "#ff7f0e",
    "down": "#d62728",
}


def architecture_figure():
    x_edges = []
    y_edges = []
    hx_edges = []
    hy_edges = []
    highlight_ok = (st.session_state.step - st.session_state.get("active_edges_t", 0)) <= 2
    active = set(st.session_state.active_edges if highlight_ok else [])
    for s, t_ in edges:
        x0, y0 = positions[s]
        x1, y1 = positions[t_]
        if (s, t_) in active:
            hx_edges += [x0, x1, None]
            hy_edges += [y0, y1, None]
        else:
            x_edges += [x0, x1, None]
            y_edges += [y0, y1, None]
    edge_trace = go.Scatter(x=x_edges, y=y_edges, mode="lines", line=dict(color="#888", width=1.5), hoverinfo="none")
    edge_trace_h = go.Scatter(x=hx_edges, y=hy_edges, mode="lines", line=dict(color="#FFD700", width=3.5), hoverinfo="none")

    x_nodes = []
    y_nodes = []
    text = []
    marker_colors = []
    marker_sizes = []
    symbols = []
    for n, (x, y) in positions.items():
        x_nodes.append(x)
        y_nodes.append(y)
        stt = st.session_state.nodes_status.get(n, "ok")
        layer = layer_map[n]
        base_color = colors[layer]
        if stt == "down":
            base_color = colors["down"]
        marker_colors.append(base_color)
        sz = 22
        symb = "circle"
        if st.session_state.service_location == "cloud" and n == "Cloud Orchestrator":
            sz = 28
            symb = "star"
        if st.session_state.service_location == "ext" and n == "Extreme B":
            sz = 28
            symb = "star"
        marker_sizes.append(sz)
        symbols.append(symb)
        text.append(n)
    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=text,
        textposition="top center",
        marker=dict(size=marker_sizes or 24, color=marker_colors, symbol=symbols, line=dict(width=2, color="#FFFFFF")),
        hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, edge_trace_h, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False), height=560)
    return fig


def kpi_metrics():
    df = st.session_state.metrics.tail(60)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud CPU %", f"{df.cpu_cloud.iloc[-1]:.1f}", f"{df.cpu_cloud.iloc[-1]-df.cpu_cloud.iloc[0]:+.1f}")
    c2.metric("Edge CPU %", f"{df.cpu_edge.iloc[-1]:.1f}", f"{df.cpu_edge.iloc[-1]-df.cpu_edge.iloc[0]:+.1f}")
    c3.metric("Extreme CPU %", f"{df.cpu_ext.iloc[-1]:.1f}", f"{df.cpu_ext.iloc[-1]-df.cpu_ext.iloc[0]:+.1f}")
    c4.metric("Latency ms", f"{df.latency.iloc[-1]:.0f}", f"{df.latency.iloc[-1]-df.latency.iloc[0]:+.0f}")


with st.sidebar:
    st.title("Controls")
    col_a, col_b = st.columns(2)
    if col_a.button("Start", use_container_width=True):
        st.session_state.running = True
    if col_b.button("Stop", use_container_width=True):
        st.session_state.running = False
    if st.button("Step once", use_container_width=True):
        simulate_step()
    st.session_state.interval_sec = st.slider("Update interval (s)", 0.5, 5.0, st.session_state.interval_sec, 0.5)
    st.divider()
    st.subheader("Scenarios")
    st.session_state.scenarios["latency_spike"] = st.checkbox("Latency spike", value=st.session_state.scenarios["latency_spike"]) 
    st.session_state.scenarios["edge_failure"] = st.checkbox("Edge failure", value=st.session_state.scenarios["edge_failure"]) 
    st.session_state.scenarios["llm_load"] = st.checkbox("LLM load surge", value=st.session_state.scenarios["llm_load"]) 
    st.session_state.scenarios["energy_saving"] = st.checkbox("Energy-saving mode", value=st.session_state.scenarios["energy_saving"]) 
    st.divider()
    st.caption("Service placement")
    placement = st.radio("", ["ext", "edge", "cloud"], horizontal=True, index=["ext", "edge", "cloud"].index(st.session_state.service_location))
    if placement != st.session_state.service_location:
        migrate_service(placement)
    st.caption("Reset simulation")
    if st.button("Reset", type="secondary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

st.title("IoT–Edge–Cloud Continuum Dashboard")

if st.session_state.running:
    simulate_step()
    time.sleep(st.session_state.interval_sec)
    st.experimental_rerun()

tabs = st.tabs(["Overview", "LLM Inference", "Live Metrics", "Scenarios & Log", "ROI & Business"])

with tabs[0]:
    kpi_metrics()
    st.plotly_chart(architecture_figure(), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Service location", st.session_state.service_location.upper())
    df = st.session_state.metrics
    c2.metric("Throughput", f"{df.throughput.iloc[-1]:.1f} rps")
    c3.metric("SLA", f"{df.sla.iloc[-1]:.2f}%")

with tabs[1]:
    st.subheader("LLM Inference Demo")
    examples = {
        "Reduce latency for robots at the edge": "Reduce latency to serve robot navigation queries",
        "Simulate edge failure and recover to cloud": "An edge node is failing; ensure continuity",
        "Scale LLM inference for peak traffic": "Scale LLM inference for high traffic",
        "Optimize energy usage": "Reduce energy usage while maintaining SLA",
    }
    ex = st.selectbox("Examples", list(examples.keys()), index=0)
    prompt = st.text_area("User request", value="", placeholder=examples[ex], height=100)
    b1, b2, b3 = st.columns(3)
    if b1.button("Classify Intent", use_container_width=True):
        use_text = prompt.strip() or examples[ex]
        intent = classify_intent(use_text)
        st.session_state.llm_intent = intent
        st.session_state.llm_history.append({"t": st.session_state.step, "type": "intent", "input": use_text, "output": intent})
        set_active_edges([("Management UI", "Cloud Orchestrator"), ("LLM Intent", "Cloud Orchestrator")])
        log_event("LLM Intent classified")
    if b2.button("Reason Plan", use_container_width=True):
        if st.session_state.get("llm_intent"):
            plan = reason_plan(st.session_state.llm_intent, st.session_state.metrics.iloc[-1])
            st.session_state.llm_plan = plan
            st.session_state.llm_history.append({"t": st.session_state.step, "type": "plan", "input": st.session_state.llm_intent, "output": plan})
            path = [("Management UI", "Cloud Orchestrator"), ("LLM Intent", "Cloud Orchestrator"), ("LLM Reason", "Cloud Orchestrator"), ("Cloud Orchestrator", "Migration"), ("Cloud Orchestrator", "Service Registry")]
            target = plan.get("target_placement")
            if target == "edge":
                path.append(("Cloud Orchestrator", "Extreme B"))
            elif target == "ext":
                path.append(("Cloud Orchestrator", "Extreme B"))
            set_active_edges(path)
            log_event(f"LLM Reason produced plan -> target={target}")
        else:
            st.warning("Run Classify Intent first.")
    if b3.button("Execute Plan", use_container_width=True):
        plan = st.session_state.get("llm_plan")
        if plan:
            for scen, val in plan.get("scenario_toggles", {}).items():
                if scen in st.session_state.scenarios:
                    st.session_state.scenarios[scen] = val
            if plan.get("target_placement"):
                migrate_service(plan["target_placement"])
            log_event("LLM plan executed")
        else:
            st.warning("No plan to execute. Run Reason Plan first.")
    cL, cR = st.columns(2)
    with cL:
        st.caption("Intent (structured)")
        st.json(st.session_state.get("llm_intent") or {})
    with cR:
        st.caption("Plan")
        st.json(st.session_state.get("llm_plan") or {})
    st.caption("LLM history")
    if len(st.session_state.llm_history) == 0:
        st.info("No LLM interactions yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.llm_history), use_container_width=True, height=240)

with tabs[2]:
    df = st.session_state.metrics.tail(200)
    c1, c2 = st.columns(2)
    fig_cpu = go.Figure()
    fig_cpu.add_trace(go.Scatter(x=df.t, y=df.cpu_cloud, name="Cloud", mode="lines"))
    fig_cpu.add_trace(go.Scatter(x=df.t, y=df.cpu_edge, name="Edge", mode="lines"))
    fig_cpu.add_trace(go.Scatter(x=df.t, y=df.cpu_ext, name="Extreme", mode="lines"))
    fig_cpu.update_layout(title="CPU Utilization %", xaxis_title="t", yaxis_title="%", height=320, margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig_cpu, use_container_width=True)

    fig_lat = go.Figure()
    fig_lat.add_trace(go.Scatter(x=df.t, y=df.latency, name="Latency", mode="lines"))
    fig_lat.update_layout(title="Latency (ms)", xaxis_title="t", yaxis_title="ms", height=320, margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(fig_lat, use_container_width=True)

    fig_th = go.Figure()
    fig_th.add_trace(go.Scatter(x=df.t, y=df.throughput, name="Throughput", mode="lines"))
    fig_th.update_layout(title="Throughput (rps)", xaxis_title="t", yaxis_title="rps", height=320, margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig_th, use_container_width=True)

    fig_en = go.Figure()
    fig_en.add_trace(go.Scatter(x=df.t, y=df.energy_kwh, name="Energy", mode="lines"))
    fig_en.update_layout(title="Energy Consumption (kWh)", xaxis_title="t", yaxis_title="kWh", height=320, margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(fig_en, use_container_width=True)

with tabs[3]:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Scenario Toggles")
        for k in ["latency_spike", "edge_failure", "llm_load", "energy_saving"]:
            st.write(f"{k.replace('_',' ').title()}: {'ON' if st.session_state.scenarios[k] else 'OFF'}")
        st.subheader("Node Status")
        for n, s_ in st.session_state.nodes_status.items():
            st.write(f"{n}: {s_.upper()}")
    with c2:
        st.subheader("Event Log")
        if len(st.session_state.events) == 0:
            st.info("No events yet. Trigger scenarios or start the simulation.")
        else:
            st.dataframe(pd.DataFrame(st.session_state.events).tail(100), use_container_width=True, height=480)

with tabs[4]:
    st.subheader("Investor View: ROI Projection")
    col1, col2, col3 = st.columns(3)
    baseline_cloud_cost = col1.number_input("Baseline cloud cost ($/hr)", 2.0, 200.0, 25.0, 1.0)
    edge_energy_cost = col2.number_input("Edge energy cost ($/kWh)", 0.05, 1.0, 0.2, 0.01)
    automation_eff = col3.slider("Operational efficiency gain (%)", 0, 50, 15, 1)
    col4, col5, col6 = st.columns(3)
    downtime_cost = col4.number_input("Downtime cost ($/hr)", 10.0, 10000.0, 500.0, 10.0)
    downtime_reduction = col5.slider("Downtime avoided (hrs/mo)", 0, 200, 12, 1)
    months = col6.slider("Projection months", 3, 36, 12, 1)

    dfm = st.session_state.metrics.tail(1)
    energy_use = float(dfm.energy_kwh.iloc[-1]) * 24 * 30
    monthly_energy_cost = energy_use * edge_energy_cost
    monthly_cloud_baseline = baseline_cloud_cost * 24 * 30
    savings_eff = monthly_cloud_baseline * (automation_eff / 100.0)
    savings_downtime = downtime_cost * downtime_reduction
    monthly_benefit = savings_eff + savings_downtime - monthly_energy_cost
    deployment_cost = 15000.0

    timeline = np.arange(1, months + 1)
    cumulative_benefit = np.cumsum(np.full(months, monthly_benefit)) - deployment_cost
    breakeven_month = next((i + 1 for i, v in enumerate(cumulative_benefit) if v >= 0), None)
    roi_pct = (cumulative_benefit[-1] / deployment_cost) * 100.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly benefit", f"${monthly_benefit:,.0f}")
    c2.metric("Monthly energy cost", f"${monthly_energy_cost:,.0f}")
    c3.metric("Break-even", f"{breakeven_month if breakeven_month else 'N/A'} mo")
    c4.metric("ROI (end)", f"{roi_pct:,.1f}%")

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=timeline, y=cumulative_benefit, mode="lines+markers", name="Cumulative ROI"))
    fig_roi.add_hline(y=0, line_dash="dash")
    fig_roi.update_layout(title="Cumulative ROI over time", xaxis_title="Month", yaxis_title="$", height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_roi, use_container_width=True)
