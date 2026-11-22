"""Plotly figure builders for the dashboard. Pure functions (DataFrame in, figure out), so they can be tested."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ACTION_COLORS = {"SCALE_UP": "#d95f02", "SCALE_DOWN": "#1b9e77", "MAINTAIN": "#7570b3"}
ACTION_ORDER = ["SCALE_UP", "SCALE_DOWN", "MAINTAIN"]
MODEL_COLORS = {"Random Forest + current value": "#1f77b4", "Random Forest (notebook features)": "#ff7f0e",
                "Naive Persistence": "#aaaaaa", "Persistence (current value)": "#2ca02c"}


def _layout(fig, title=None, height=380):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def downsample(df, max_points=4000):
    """Keep at most max_points rows (evenly spaced) so charts stay fast."""
    if len(df) <= max_points:
        return df
    return df.iloc[np.linspace(0, len(df) - 1, max_points).astype(int)]


def usage_figure(g, column, label):
    g = downsample(g)
    fig = go.Figure(go.Scattergl(x=g["time_stamp"], y=g[column], mode="lines", line=dict(width=1)))
    fig.update_layout(xaxis_title="Timestamp (seconds)", yaxis_title=f"{label} utilisation (%)")
    return _layout(fig, f"{label} utilisation over time")


def workload_heatmap(usage, bins=48):
    """Average CPU per machine per time bin."""
    u = usage[["machine_id", "time_stamp", "cpu_util_percent"]].copy()
    u["bin"] = pd.cut(u["time_stamp"], bins=bins, labels=False)
    pivot = u.groupby(["machine_id", "bin"])["cpu_util_percent"].mean().unstack()
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(x="Time bin", y="Machine", color="CPU %"))
    return _layout(fig, "Average CPU utilisation per machine over time", height=430)


def forecast_figure(d, resource, show_baselines=True):
    """Actual vs predicted for one machine. d needs the test_predictions columns."""
    key, tgt, cur, naive = {"CPU": ("cpu", "cpu_target", "cpu_util_percent", "cpu_naive"),
                            "Memory": ("mem", "mem_target", "mem_util_percent", "mem_naive")}[resource]
    d = downsample(d, 3000)
    fig = go.Figure()
    fig.add_scatter(x=d["time_stamp"], y=d[tgt], name="Actual (next value)", line=dict(color="#222", width=2))
    fig.add_scatter(x=d["time_stamp"], y=d[f"{key}_pred_current_value"], name="Random Forest + current value",
                    line=dict(color=MODEL_COLORS["Random Forest + current value"], width=1.2))
    if show_baselines:
        fig.add_scatter(x=d["time_stamp"], y=d[f"{key}_pred_notebook"], name="Random Forest (notebook features)",
                        line=dict(color=MODEL_COLORS["Random Forest (notebook features)"], width=1))
        fig.add_scatter(x=d["time_stamp"], y=d[cur], name="Persistence (current value)",
                        line=dict(color=MODEL_COLORS["Persistence (current value)"], width=1, dash="dash"))
        fig.add_scatter(x=d["time_stamp"], y=d[naive], name="Naive Persistence (lag 1)",
                        line=dict(color=MODEL_COLORS["Naive Persistence"], width=1), opacity=0.6)
    fig.update_layout(xaxis_title="Timestamp (seconds)", yaxis_title=f"{resource} utilisation (%)")
    return _layout(fig, f"{resource}: actual vs predicted")


def metrics_bar(comp, metric="MAE"):
    fig = px.bar(comp, x="Resource", y=metric, color="Model", barmode="group", color_discrete_map=MODEL_COLORS, text_auto=".3f")
    fig.update_layout(yaxis_title=f"{metric} (percentage points)")
    return _layout(fig, f"Forecast error ({metric}, lower is better)")


def action_bar(dist, title="Final actions by coordinator strategy"):
    """dist: rows = actions, columns = strategies."""
    long = dist.reindex(ACTION_ORDER).fillna(0).reset_index().melt(id_vars=dist.index.name or "index",
                                                                    var_name="Strategy", value_name="Decisions")
    long = long.rename(columns={dist.index.name or "index": "Action"})
    fig = px.bar(long, x="Strategy", y="Decisions", color="Action", barmode="group", color_discrete_map=ACTION_COLORS,
                 category_orders={"Action": ACTION_ORDER}, text_auto=True)
    return _layout(fig, title)


def sensitivity_figure(sens, strategy, memory_high):
    d = sens[(sens["strategy"] == strategy) & (sens["memory_high"] == memory_high)]
    fig = go.Figure()
    for a in ACTION_ORDER:
        fig.add_scatter(x=d["cpu_high"], y=d[f"{a}_pct"], name=a, mode="lines+markers", line=dict(color=ACTION_COLORS[a]))
    fig.update_layout(xaxis_title="Resource Agent CPU-high threshold (%)", yaxis_title="Share of decisions (%)")
    return _layout(fig, f"Sensitivity to the CPU threshold ({strategy}, memory threshold {memory_high}%)")


def sensitivity_memory_figure(sens, strategy, cpu_high):
    d = sens[(sens["strategy"] == strategy) & (sens["cpu_high"] == cpu_high)]
    fig = go.Figure()
    for a in ACTION_ORDER:
        fig.add_scatter(x=d["memory_high"], y=d[f"{a}_pct"], name=a, mode="lines+markers", line=dict(color=ACTION_COLORS[a]))
    fig.update_layout(xaxis_title="Resource Agent memory-high threshold (%)", yaxis_title="Share of decisions (%)")
    return _layout(fig, f"Sensitivity to the memory threshold ({strategy}, CPU threshold {cpu_high}%)")


def budget_figure(b):
    long = b.melt(id_vars=["budget_per_hour", "n"], value_vars=ACTION_ORDER, var_name="Action", value_name="Decisions")
    fig = px.bar(long, x="budget_per_hour", y="Decisions", color="Action", barmode="stack", color_discrete_map=ACTION_COLORS,
                 category_orders={"Action": ACTION_ORDER})
    fig.update_layout(xaxis_title="Simulated budget (per machine-hour)", xaxis=dict(type="category"))
    return _layout(fig, "Cost-Aware strategy: effect of the simulated budget")
