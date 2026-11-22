"""Streamlit dashboard: Agentic AI orchestrated cloud resource management (decision-support prototype).

Run:  streamlit run dashboard/app.py
Needs the results of `python -m src.app run` (the dashboard tells you if they are missing).
No real cloud resource is ever changed; every recommendation needs human approval.
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(os.environ.get("AGENTIC_ROOT") or PROJECT)

import pandas as pd
import streamlit as st

import charts
from src import dashboard_data as dd
from src.explainability.explanation import check_explanation
from src.forecasting.evaluation import BASELINE_NOTE, NAIVE, PERSISTENCE, VARIANT_NAMES, mae_rmse, verdict
from src.orchestration.decision_log import verify_trace
from src.service import MachineStore, analyze_machine

STRATEGIES = ["original", "improved", "cost_aware"]
STRATEGY_LABELS = {"original": "Original", "improved": "Improved (conflict-aware)", "cost_aware": "Cost-Aware"}
STRATEGY_HELP = {
    "original": "Follows any scale-up vote from the Resource or Performance Agent; ignores cost.",
    "improved": "Needs corroboration for scale-up, refuses to act on unresolved conflicts, Security has the final say.",
    "cost_aware": "Improved decision plus the (simulated) budget: refuses over-budget scale-ups unless the forecast is critical.",
}
ACTION_ICON = {"SCALE_UP": "🔺", "SCALE_DOWN": "🔻", "MAINTAIN": "⏸️"}
ACTION_COLOR = {"SCALE_UP": "orange", "SCALE_DOWN": "green", "MAINTAIN": "violet"}
PAGES = ["Dashboard Overview", "Data Explorer", "Forecasting", "Agent Monitoring",
         "Orchestration & Negotiation", "Explainability", "Evaluation"]

st.set_page_config(page_title="Agentic Cloud Resource Management", page_icon="☁️", layout="wide")


# ----------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading machine history...")
def get_store(root):
    return MachineStore.from_project(root)


@st.cache_data(show_spinner=False)
def load_predictions(root):
    return dd.load_test_predictions(root)


@st.cache_data(show_spinner=False)
def load_table(name, root):
    if name in STRATEGIES:
        return dd.load_decision_table(name, root)
    if name == "forecast":
        return dd.load_forecast_metrics(root)
    if name == "per_machine":
        return dd.load_per_machine_metrics(root)
    if name == "comparison":
        return dd.load_strategy_comparison(root)
    return dd.load_eval_table(name, root)


@st.cache_data(show_spinner="Running the four agents and three coordinators...")
def run_analysis(root, machine, index):
    return analyze_machine(get_store(root), machine, index)


def action_badge(action):
    return f":{ACTION_COLOR.get(action, 'gray')}[**{ACTION_ICON.get(action, '')} {action}**]"


def pct(x):
    return f"{x:.1f}%"


def utilisation_delta(pred, cur):
    """(text, colour) for a utilisation change. Rising use is the bad direction (red); a tiny change shows no arrow."""
    d = pred - cur
    if abs(d) < 0.05:
        return None, "off"
    return f"{d:+.1f} pts", "inverse"


def show_table(df, index_col):
    """Small tables with long text: st.table wraps the text instead of cutting it off like st.dataframe does."""
    st.table(df.set_index(index_col))


def window_controls(total, default=600, step=50, key=""):
    """(start, n) for a 'points to show' window. Sliders are skipped when the range is empty (short histories)."""
    if total <= 100:
        return 0, total
    top = min(5000, total)
    n = st.slider("Points to show", 100, top, min(default, top), step=step, key=key + "n") if top > 100 else total
    start = st.slider("Start position", 0, total - n, 0, step=step, key=key + "s") if total - n >= step else 0
    return start, n


# ----------------------------------------------------------------- sidebar
def sidebar(store):
    st.sidebar.title("☁️ Agentic Cloud Resource Management")
    st.sidebar.caption("Decision-support prototype. Costs are SIMULATED. Nothing here changes a real cloud resource.")
    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.divider()
    strategy = st.sidebar.radio("Coordinator strategy", STRATEGIES, format_func=STRATEGY_LABELS.get, index=1,
                                help="Used on the Overview, Orchestration and Explainability pages.")
    st.sidebar.caption(STRATEGY_HELP[strategy])
    st.sidebar.divider()
    machine = st.sidebar.selectbox("Machine", store.machines())
    n = store.n_observations(machine)
    latest = st.sidebar.checkbox("Use the latest observation", value=True)
    index = n - 1 if latest else st.sidebar.slider("Observation (row in the machine's history)", 3, n - 1, n - 1)
    return page, strategy, machine, index


# ----------------------------------------------------------------- pages
def page_overview(strategy, root):
    st.header("Dashboard Overview")
    st.caption(f"Saved run of the pipeline, coordinator: **{STRATEGY_LABELS[strategy]}**. "
               "Each machine's most recent test-period observation.")
    t = load_table(strategy, root)
    summary = dd.load_run_summary(root)
    c = st.columns(4)
    c[0].metric("Machines analysed", len(t))
    c[1].metric("Avg current CPU", pct(t["current_cpu"].mean()))
    dtxt, dcol = utilisation_delta(t["predicted_cpu"].mean(), t["current_cpu"].mean())
    c[2].metric("Avg predicted CPU", pct(t["predicted_cpu"].mean()), dtxt, delta_color=dcol)
    c[3].metric("Avg current memory", pct(t["current_memory"].mean()))
    c = st.columns(4)
    dtxt, dcol = utilisation_delta(t["predicted_memory"].mean(), t["current_memory"].mean())
    c[0].metric("Avg predicted memory", pct(t["predicted_memory"].mean()), dtxt, delta_color=dcol)
    counts = t["final_action"].value_counts()
    c[1].metric("SCALE_UP", int(counts.get("SCALE_UP", 0)))
    c[2].metric("SCALE_DOWN", int(counts.get("SCALE_DOWN", 0)))
    c[3].metric("MAINTAIN", int(counts.get("MAINTAIN", 0)))

    st.subheader("Decisions per machine")
    view = t[["machine_id", "current_cpu", "predicted_cpu", "current_memory", "predicted_memory",
              "resource_vote", "performance_vote", "cost_vote", "security_vote", "final_action", "n_conflicts"]]
    st.dataframe(view, hide_index=True, column_config={
        "current_cpu": st.column_config.NumberColumn("CPU now", format="%.1f"),
        "predicted_cpu": st.column_config.NumberColumn("CPU pred", format="%.1f"),
        "current_memory": st.column_config.NumberColumn("Mem now", format="%.1f"),
        "predicted_memory": st.column_config.NumberColumn("Mem pred", format="%.1f"),
        "n_conflicts": st.column_config.NumberColumn("Conflicts")})
    dist = pd.DataFrame({s: load_table(s, root)["final_action"].value_counts() for s in STRATEGIES}).fillna(0).astype(int)
    dist.index.name = "Action"
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.action_bar(dist, "Final actions: all three strategies"))
    right.markdown("**How to read this**")
    right.write("Each machine gets one decision per strategy. Where the bars differ, the strategies resolve "
                "the agents' disagreements differently: Original follows any scale-up vote, Improved needs "
                "corroboration, and Cost-Aware also enforces the simulated budget.")
    st.caption(f"Run `{summary['run_id']}` finished {summary['finished_utc']} UTC. Pricing: {summary['pricing_source']}.")


def page_data(store, machine):
    st.header("Data Explorer")
    st.caption("Alibaba Cluster Trace 2018, the 20 machines with the most observations (10-second sampling).")
    c = st.columns(3)
    c[0].metric("Observations", f"{len(store.usage):,}")
    c[1].metric("Machines", len(store.machines()))
    c[2].metric("Time span (s)", f"{int(store.usage['time_stamp'].max() - store.usage['time_stamp'].min()):,}")
    with st.expander("Dataset preview", expanded=True):
        st.dataframe(store.usage.head(200), hide_index=True)
        st.dataframe(store.usage[["cpu_util_percent", "mem_util_percent"]].describe().T)
    g = store.series(machine)
    lo, hi = int(g["time_stamp"].min()), int(g["time_stamp"].max())
    window = st.slider(f"Time window for {machine} (seconds)", lo, hi, (lo, hi)) if hi > lo else (lo, hi)
    gw = g[g["time_stamp"].between(*window)]
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.usage_figure(gw, "cpu_util_percent", "CPU"))
    c2.plotly_chart(charts.usage_figure(gw, "mem_util_percent", "Memory"))
    st.subheader("Workload across machines over time")
    st.plotly_chart(charts.workload_heatmap(store.usage))
    st.caption("Memory stays near 91% on every machine, so CPU is what separates busy machines from idle ones.")


def page_forecast(machine, root):
    st.header("Forecasting")
    pred = load_predictions(root)
    d = pred[pred["machine_id"] == machine].sort_values("time_stamp")
    comp = load_table("forecast", root)
    st.caption("Random Forest models predict the next CPU and memory value. The system uses the 'current value' model; the "
               "original notebook-feature model is shown for comparison. Evaluated on each machine's last 20% of time.")
    start, n = window_controls(len(d), key="fc_")
    dv = d.iloc[start:start + n]
    show_base = st.checkbox("Show persistence baselines", value=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.forecast_figure(dv, "CPU", show_base))
    c2.plotly_chart(charts.forecast_figure(dv, "Memory", show_base))

    st.subheader(f"Error metrics for {machine} (test period) and overall")
    rows = []
    for res, tgt, rf, naive, cur in [("CPU", "cpu_target", "cpu_pred", "cpu_naive", "cpu_util_percent"),
                                     ("Memory", "mem_target", "mem_pred", "mem_naive", "mem_util_percent")]:
        for model, col in [(VARIANT_NAMES["current_value"], f"{rf}_current_value"), (VARIANT_NAMES["notebook"], f"{rf}_notebook"),
                           (NAIVE, naive), (PERSISTENCE, cur)]:
            mae, rmse = mae_rmse(d[tgt], d[col])
            ov = comp[(comp["Resource"] == res) & (comp["Model"] == model)].iloc[0]
            rows.append({"Resource": res, "Model": model, f"MAE {machine}": mae, f"RMSE {machine}": rmse,
                         "MAE (all machines)": ov["MAE"], "RMSE (all machines)": ov["RMSE"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.metrics_bar(comp, "MAE"))
    c2.plotly_chart(charts.metrics_bar(comp, "RMSE"))
    st.markdown("**Findings (computed from the metrics):**")
    for line in verdict(comp):
        st.write("• " + line)
    st.caption(BASELINE_NOTE)


def page_agents(machine, index, root, strategy):
    st.header("Agent Monitoring")
    res = run_analysis(root, machine, index)
    o = res["observation"]
    st.caption(f"Machine **{machine}** at t = {o['timestamp']} s. The forecast and the four agents run live for this observation.")
    c = st.columns(4)
    c[0].metric("CPU now", pct(o["current_cpu"]))
    dtxt, dcol = utilisation_delta(o["predicted_cpu"], o["current_cpu"])
    c[1].metric("CPU predicted (next)", pct(o["predicted_cpu"]), dtxt, delta_color=dcol)
    c[2].metric("Memory now", pct(o["current_memory"]))
    dtxt, dcol = utilisation_delta(o["predicted_memory"], o["current_memory"])
    c[3].metric("Memory predicted (next)", pct(o["predicted_memory"]), dtxt, delta_color=dcol)
    if "actual_next_cpu" in o:
        st.caption(f"Actual next values: CPU {o['actual_next_cpu']:.0f}%, memory {o['actual_next_memory']:.0f}% "
                   f"(forecast error {abs(o['predicted_cpu'] - o['actual_next_cpu']):.1f} / "
                   f"{abs(o['predicted_memory'] - o['actual_next_memory']):.1f} points).")
    if "data_warning" in o:
        st.warning(o["data_warning"])

    rec = res["decisions"][strategy]
    agents = rec["agents"]
    cols = st.columns(2)
    for i, key in enumerate(["resource", "performance", "cost", "security"]):
        with cols[i % 2]:
            a = agents.get(key)
            with st.container(border=True):
                if not a:
                    st.subheader(key.title() + " Agent")
                    st.error("Unavailable: " + next((e["error"] for e in rec["agent_errors"] if e["agent"] == key), "no output"))
                    continue
                st.subheader(a["agent"])
                if key == "security":
                    st.markdown(f"**Verdict:** {a['verdict']}  |  **Risk:** {a['risk_level']}")
                    st.markdown(f"**Allowed actions:** {', '.join(a['allowed_actions'])}")
                    if a["alerts"]:
                        st.markdown("**Alerts:** " + ", ".join(a["alerts"]))
                    show_table(pd.DataFrame(a["policy_checks"]).assign(
                        result=lambda x: x["passed"].map({True: "pass", False: "FAIL"}))[["check", "result", "detail"]], "check")
                else:
                    st.markdown(f"**Recommendation:** {action_badge(a['vote'])}  |  **Status:** {a['status']}  |  **Risk:** {a['risk_level']}")
                if key == "cost":
                    st.markdown(f"**Pricing:** {a['pricing_source']} (not real cloud prices)")
                    m = a["metrics"]
                    st.markdown(f"Current {a['current_cost']:.2f}/h  |  after scale-up {m['cost_scale_up']:.2f}/h  |  "
                                f"after scale-down {m['cost_scale_down']:.2f}/h  |  budget {m['budget']:.2f}/h")
                    st.markdown(f"Scale-up within budget: **{'yes' if a['scale_up_within_budget'] else 'no'}**")
                st.write(a["reason"])
                with st.expander("Metrics"):
                    st.json(a["metrics"])


def page_orchestration(machine, index, strategy, root):
    st.header("Orchestration & Negotiation")
    res = run_analysis(root, machine, index)
    rec = res["decisions"][strategy]
    st.caption(f"Strategy: **{STRATEGY_LABELS[strategy]}**. {STRATEGY_HELP[strategy]}")

    st.subheader("1. The four agent outputs")
    rows = []
    for key in ["resource", "performance", "cost", "security"]:
        a = rec["agents"].get(key)
        rows.append({"Agent": a["agent"] if a else key, "Vote": (a["vote"] if a else "UNAVAILABLE"),
                     "Status": a["status"] if a else "-", "Risk": a["risk_level"] if a else "-",
                     "Reason": a["reason"] if a else "-"})
    show_table(pd.DataFrame(rows), "Agent")

    st.subheader("2. Conflicts and constraints")
    if rec["conflicts"]:
        for c in rec["conflicts"]:
            st.warning(f"**{c['type']}**: {c['detail']}")
    else:
        st.success("No conflicts between the agents.")

    st.subheader("3. Final decision")
    c = st.columns([1, 3])
    c[0].markdown(f"## {action_badge(rec['final_action'])}")
    with c[1]:
        for line in rec["rationale"]:
            st.write("• " + line)
    if rec["rejected"]:
        with st.expander("Recommendations not followed"):
            for x in rec["rejected"]:
                st.write(f"• **{x['agent']}** ({x['vote']}): {x['reason']}")

    st.subheader("4. Compare the three strategies on this observation")
    cols = st.columns(3)
    for col, s in zip(cols, STRATEGIES):
        r = res["decisions"][s]
        with col, st.container(border=True):
            st.markdown(f"**{STRATEGY_LABELS[s]}**")
            st.markdown(f"## {action_badge(r['final_action'])}")
            if s != strategy and r["final_action"] != rec["final_action"]:
                st.caption(f"Differs from {STRATEGY_LABELS[strategy]}: {r['rationale'][-1]}")
            elif s != strategy:
                st.caption("Same action.")
    st.subheader("5. Human approval")
    st.info("This is a recommendation only. Approving here records your choice in this session; **no cloud resource is changed**.")
    approvals = st.session_state.setdefault("approvals", {})
    key = f"{machine}|{index}|{strategy}"
    c = st.columns(3)
    if c[0].button("Approve (simulation)", key="approve_" + key):
        approvals[key] = "APPROVED"
    if c[1].button("Reject", key="reject_" + key):
        approvals[key] = "REJECTED"
    c[2].write(f"Status: **{approvals.get(key, 'PENDING')}**" + (" (a reviewer should look at this one)" if rec["requires_review"] else ""))


def page_explain(machine, index, strategy, root):
    st.header("Explainability")
    res = run_analysis(root, machine, index)
    rec = res["decisions"][strategy]
    d = rec["explanation_detail"]
    st.markdown(f"### {action_badge(rec['final_action'])} for {machine}")
    st.write(d["headline"])
    st.write(d["inputs"])
    problems = check_explanation(rec) + verify_trace(rec)
    if problems:
        st.error("Explanation check failed: " + "; ".join(problems))
    else:
        st.success("Checked: this explanation matches the decision record, and the decision traces back to its input.")

    st.subheader("Agent votes and reasoning")
    reasoning = pd.DataFrame(d["agent_reasoning"])
    reasoning["followed"] = reasoning["followed"].map({True: "yes", False: "no"}).fillna("-")
    show_table(reasoning.rename(columns={"followed": "final decision consistent with it?"}), "agent")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Conflicts")
        for x in d["conflicts"] or ["None between agents."]:
            st.write("• " + x)
        st.subheader("How the decision was reached")
        for x in d["decision_path"]:
            st.write("• " + x)
        if d["not_followed"]:
            st.subheader("Recommendations not followed")
            for x in d["not_followed"]:
                st.write("• " + x)
    with c2:
        st.subheader("Cost impact")
        ci = d["cost_impact"]
        if ci:
            m = st.columns(3)
            m[0].metric("Current cost / h", f"{ci['current_cost']:.2f}")
            m[1].metric(f"After {rec['final_action']}", f"{ci['proposed_cost']:.2f}",
                        f"{ci['difference']:+.2f}" if abs(ci["difference"]) >= 0.005 else None,
                        delta_color="inverse" if abs(ci["difference"]) >= 0.005 else "off")
            m[2].metric("Budget / h", f"{ci['budget']:.2f}")
            st.caption(f"{ci['budget_status']}, {ci['pricing_source']} pricing (not real cloud prices).")
        else:
            st.info("Cost assessment unavailable.")
        st.subheader("Security checks")
        sec = d["security"]
        if sec:
            st.write(f"Verdict **{sec['verdict']}**: {sec['checks_passed']} of {sec['checks_total']} checks passed. "
                     f"Allowed actions: {', '.join(sec['allowed_actions'])}.")
            show_table(pd.DataFrame(sec["checks"]).assign(result=lambda x: x["passed"].map({True: "pass", False: "FAIL"}))[
                ["check", "result", "detail"]], "check")
        else:
            st.info("Security assessment unavailable.")
    st.info(d["human_approval"])
    with st.expander("Full explanation text"):
        st.write(rec["explanation"])


def page_eval(root, machine):
    st.header("Evaluation")
    t1, t2, t3, t4 = st.tabs(["Forecasting", "Coordinators", "Threshold sensitivity", "Budget sensitivity"])
    with t1:
        comp = load_table("forecast", root)
        st.dataframe(comp.round(4), hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.metrics_bar(comp, "MAE"))
        c2.plotly_chart(charts.metrics_bar(comp, "RMSE"))
        st.markdown("**Findings (computed from the metrics):**")
        for line in verdict(comp):
            st.write("• " + line)
        st.caption(BASELINE_NOTE)
        pm = load_table("per_machine", root)
        wins = {}
        for v, label in [("current_value", "RF + current value"), ("notebook", "RF (notebook features)")]:
            for res in ("CPU", "Memory"):
                rf = pm[f"{res}_RF_{v}_MAE"]
                wins[f"{label}: {res} vs Naive (notebook baseline)"] = int((rf < pm[f"{res}_Naive_MAE"]).sum())
                wins[f"{label}: {res} vs Persistence (current value)"] = int((rf < pm[f"{res}_Current_MAE"]).sum())
        st.write("**Machines (out of %d) where the Random Forest has the lower MAE:**" % len(pm))
        show_table(pd.DataFrame({"Comparison": list(wins), "Machines won": list(wins.values())}), "Comparison")
        with st.expander("Per-machine MAE"):
            st.dataframe(pm.round(3), hide_index=True)
        d = load_predictions(root)
        d = d[d["machine_id"] == machine].sort_values("time_stamp").head(600)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.forecast_figure(d, "CPU"))
        c2.plotly_chart(charts.forecast_figure(d, "Memory"))
    with t2:
        dist = pd.DataFrame({s: load_table(s, root)["final_action"].value_counts() for s in STRATEGIES}).fillna(0).astype(int)
        dist.index.name = "Action"
        samp = load_table("decision_distribution_sample", root)
        samp = samp.set_index(samp.columns[0])
        samp.index.name = "Action"
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.action_bar(dist, "Latest observation of the 20 machines"))
        c2.plotly_chart(charts.action_bar(samp, "Sample of 2,000 observations across the test period"))
        st.caption("The 2,000-observation sample is spread over time but neighbouring observations of a machine are correlated.")
        st.subheader("Agreement between strategies (20 machines)")
        st.dataframe(load_table("strategy_agreement", root).assign(agreement=lambda x: (x["agreement"] * 100).round(0).astype(int).astype(str) + "%"), hide_index=True)
        st.subheader("Strategy comparison per machine")
        st.dataframe(load_table("comparison", root), hide_index=True)
        st.subheader("How often each agent voted for each action (Improved run)")
        st.dataframe(load_table("agent_votes", root), hide_index=True)
    with t3:
        sens = load_table("threshold_sensitivity", root)
        st.caption("Vary the Resource Agent's pressure thresholds; all other agents keep their defaults. "
                   f"Based on {int(sens['n'].iloc[0]):,} observations spread across the test period.")
        s = st.selectbox("Strategy", STRATEGIES, index=1, format_func=STRATEGY_LABELS.get, key="sens_strategy")
        c1, c2 = st.columns(2)
        mem = c1.selectbox("Memory threshold held at", sorted(sens["memory_high"].unique()), index=1)
        cpu = c2.selectbox("CPU threshold held at", sorted(sens["cpu_high"].unique()), index=2)
        c1.plotly_chart(charts.sensitivity_figure(sens, s, mem))
        c2.plotly_chart(charts.sensitivity_memory_figure(sens, s, cpu))
        st.caption("CPU is rarely above 70% on these machines, so the CPU threshold barely changes anything; "
                   "memory sits near 91%, so the memory threshold changes a lot.")
        with st.expander("Full table"):
            st.dataframe(sens, hide_index=True)
    with t4:
        b = load_table("budget_sensitivity", root)
        st.plotly_chart(charts.budget_figure(b))
        st.caption("A scale-up costs 6.40 per hour in the simulated pricing. Budgets below that stop the Cost-Aware strategy "
                   "from scaling up unless the forecast is critical. The budget is an assumption, not a measured value.")
        st.dataframe(b, hide_index=True)


# ----------------------------------------------------------------- main
def main():
    st.title("Agentic AI Orchestrated Cloud Resource Management")
    st.caption("Multi-agent negotiation and explainable decisions on the Alibaba Cluster Trace 2018. "
               "Decision-support prototype: recommendations need human approval; costs are simulated.")
    try:
        store = get_store(ROOT)
        page, strategy, machine, index = sidebar(store)
        if page == "Dashboard Overview":
            page_overview(strategy, ROOT)
        elif page == "Data Explorer":
            page_data(store, machine)
        elif page == "Forecasting":
            page_forecast(machine, ROOT)
        elif page == "Agent Monitoring":
            page_agents(machine, index, ROOT, strategy)
        elif page == "Orchestration & Negotiation":
            page_orchestration(machine, index, strategy, ROOT)
        elif page == "Explainability":
            page_explain(machine, index, strategy, ROOT)
        else:
            page_eval(ROOT, machine)
    except FileNotFoundError as e:   # includes dd.ResultsNotFound
        st.error(f"Results are missing: {e}")
        st.code("python -m src.app run", language="bash")
        st.stop()


main()
