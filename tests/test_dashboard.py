import os
from pathlib import Path

import pandas as pd
import pytest
from streamlit.proto.Metric_pb2 import Metric
from streamlit.testing.v1 import AppTest

import sys

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import charts  # noqa: E402

PAGES = ["Dashboard Overview", "Data Explorer", "Forecasting", "Agent Monitoring",
         "Orchestration & Negotiation", "Explainability", "Evaluation"]


def start(root):
    os.environ["AGENTIC_ROOT"] = str(root)
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    return at


def go(at, page):
    at.sidebar.radio[0].set_value(page).run()
    return at


def texts(at):
    out = []
    for kind in (at.markdown, at.subheader, at.header, at.caption, at.success, at.warning, at.info, at.error):
        out += [e.value for e in kind]
    return " ".join(out)


@pytest.fixture(scope="module")
def at(project):
    app_test = start(project[0])
    yield app_test
    os.environ.pop("AGENTIC_ROOT", None)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(at, page):
    go(at, page)
    assert not at.exception, [e.value for e in at.exception]
    assert page.split(" &")[0].split()[0] in texts(at) or at.header


def test_overview_shows_machines_and_action_counts(at):
    go(at, "Dashboard Overview")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Machines analysed"] == "3"
    assert {"SCALE_UP", "SCALE_DOWN", "MAINTAIN"} <= set(metrics)
    assert sum(int(metrics[k]) for k in ("SCALE_UP", "SCALE_DOWN", "MAINTAIN")) == 3


def test_every_strategy_selectable_on_overview(at):
    go(at, "Dashboard Overview")
    for s in ["original", "improved", "cost_aware"]:
        at.sidebar.radio[1].set_value(s).run()
        assert not at.exception
        assert sum(int(m.value) for m in at.metric if m.label in ("SCALE_UP", "SCALE_DOWN", "MAINTAIN")) == 3


def test_agent_monitoring_shows_four_agent_cards(at):
    go(at, "Agent Monitoring")
    subheaders = " ".join(s.value for s in at.subheader)
    for name in ("Resource Agent", "Performance Agent", "Cost Agent", "Security Agent"):
        assert name in subheaders
    assert "SIMULATED" in " ".join(m.value for m in at.markdown)


def test_orchestration_shows_final_decision_and_three_strategies(at):
    go(at, "Orchestration & Negotiation")
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    for s in ("Original", "Improved", "Cost-Aware"):
        assert s in body
    assert any(a in body for a in ("SCALE_UP", "SCALE_DOWN", "MAINTAIN"))
    assert "no cloud resource is changed" in " ".join(i.value for i in at.info)


def test_orchestration_approval_buttons_record_choice_in_session_only(at):
    go(at, "Orchestration & Negotiation")
    at.button[0].click().run()
    assert not at.exception
    assert "APPROVED" in " ".join(m.value for m in at.markdown) or "APPROVED" in str(at.session_state["approvals"])
    assert "APPROVED" in at.session_state["approvals"].values()


def test_explainability_page_verifies_explanation(at):
    go(at, "Explainability")
    assert not at.exception
    assert any("Checked:" in s.value for s in at.success)
    assert not at.error


def test_machine_and_observation_selection_changes_output(at):
    go(at, "Agent Monitoring")
    first = {m.label: m.value for m in at.metric}
    at.sidebar.selectbox[0].set_value("m_2").run()
    second = {m.label: m.value for m in at.metric}
    assert not at.exception and first != second
    at.sidebar.checkbox[0].uncheck().run()
    assert not at.exception and len(at.sidebar.slider) == 1
    at.sidebar.slider[0].set_value(100).run()
    assert not at.exception


def test_evaluation_tabs_present(at):
    go(at, "Evaluation")
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Forecasting", "Coordinators", "Threshold sensitivity", "Budget sensitivity"]
    assert "Persistence (current value)" in " ".join(str(d.value) for d in at.dataframe) or at.warning


def test_forecasting_page_shows_computed_findings_for_both_resources(at):
    go(at, "Forecasting")
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Findings (computed from the metrics)" in body
    assert "CPU: lowest MAE is" in body and "Memory: lowest MAE is" in body
    assert any("one-step baseline" in c.value for c in at.caption)


def test_friendly_error_when_results_are_missing(tmp_path):
    a = start(tmp_path)
    assert not a.exception
    assert any("Results are missing" in e.value for e in a.error)
    os.environ.pop("AGENTIC_ROOT", None)


# ---------- chart builders ----------
def test_chart_builders_return_figures(project):
    from src import dashboard_data as dd
    root = project[0]
    pred = dd.load_test_predictions(root)
    d = pred[pred["machine_id"] == "m_1"]
    assert charts.forecast_figure(d, "CPU").data and charts.forecast_figure(d, "Memory", False).data
    usage = dd.load_usage(root)
    assert charts.usage_figure(usage[usage.machine_id == "m_1"], "cpu_util_percent", "CPU").data
    assert charts.workload_heatmap(usage, bins=10).data
    assert charts.metrics_bar(dd.load_forecast_metrics(root)).data
    dist = dd.load_eval_table("decision_distribution", root).set_index("Unnamed: 0")
    dist.index.name = "Action"
    assert charts.action_bar(dist).data
    sens = dd.load_eval_table("threshold_sensitivity", root)
    assert charts.sensitivity_figure(sens, "improved", 95).data and charts.sensitivity_memory_figure(sens, "improved", 80).data
    assert charts.budget_figure(dd.load_eval_table("budget_sensitivity", root)).data


def test_downsample_keeps_endpoints_and_limit():
    df = pd.DataFrame({"x": range(100_000)})
    s = charts.downsample(df, 1000)
    assert len(s) == 1000 and s["x"].iloc[0] == 0 and s["x"].iloc[-1] == 99_999
    assert len(charts.downsample(df.head(10), 1000)) == 10


def test_utilisation_deltas_are_coloured_by_direction_not_sign(at):
    """Rising utilisation is the bad direction (red), falling is good (green); no change shows no badge (found in the
    browser check: Streamlit's default colours made a memory DROP red)."""
    go(at, "Agent Monitoring")
    seen = 0
    for m in at.metric:
        if m.label in ("CPU predicted (next)", "Memory predicted (next)"):
            if m.delta in (None, ""):
                continue
            seen += 1
            assert m.delta.strip().startswith(("+", "-")), m.delta
            expected = Metric.MetricColor.RED if m.delta.strip().startswith("+") else Metric.MetricColor.GREEN
            assert m.color == expected, (m.label, m.delta, m.color)
    assert seen >= 1, "expected at least one coloured utilisation delta"


def test_overview_table_shows_the_final_action_column(at):
    go(at, "Dashboard Overview")
    cols = " ".join(str(c) for d in at.dataframe for c in d.value.columns)
    assert "final_action" in cols and "security_vote" in cols
