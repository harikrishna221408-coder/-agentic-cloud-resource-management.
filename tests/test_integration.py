import copy
import json

import numpy as np
import pandas as pd
import pytest

from src import app, dashboard_data as dd
from src.evaluation import decision_eval as ev
from src.orchestration.orchestrator import Orchestrator, observations_from_predictions
from src.service import MachineStore, analyze_machine
from src.utils.config import load_config
from tests.helpers import make_project

STRATEGIES = ("original", "improved", "cost_aware")


# ---------- full run ----------
def test_full_run_produces_all_outputs(project):
    root, _, summary = project
    for rel in ["models/current_value/cpu_random_forest.joblib", "models/current_value/memory_random_forest.joblib",
                "models/notebook/cpu_random_forest.joblib", "models/notebook/memory_random_forest.joblib",
                "results/metrics/forecasting_comparison.csv", "results/metrics/test_predictions.csv",
                "results/metrics/decision_distribution.csv", "results/metrics/threshold_sensitivity.csv",
                "results/metrics/budget_sensitivity.csv", "results/metrics/run_summary.json",
                "results/decisions/decisions_original.jsonl", "results/decisions/decisions_improved.csv",
                "results/decisions/decisions_cost_aware.jsonl", "results/decisions/decision_log.jsonl",
                "results/decisions/strategy_comparison.csv", "results/graphs/decision_distribution.png",
                "results/graphs/threshold_sensitivity.png"]:
        assert (root / rel).exists(), rel
    assert summary["machines"] == 3 and summary["explanation_or_trace_problems"] == 0
    assert summary["pricing_source"] == "SIMULATED"


def test_every_machine_gets_exactly_one_decision_per_strategy_with_explanation(project):
    root, _, _ = project
    for s in STRATEGIES:
        recs = dd.load_decisions(s, root)
        ids = [r["machine_id"] for r in recs]
        assert sorted(ids) == ["m_1", "m_2", "m_3"] and len(set(ids)) == 3
        assert all(r["explanation"] and r["final_action"] in ("SCALE_UP", "SCALE_DOWN", "MAINTAIN") for r in recs)
        assert all(r["human_approval_required"] for r in recs)


def test_forecast_metrics_include_both_random_forests_and_both_baselines(project):
    root, _, _ = project
    m = dd.load_forecast_metrics(root)
    assert set(m["Model"]) == {"Random Forest + current value", "Random Forest (notebook features)",
                               "Naive Persistence", "Persistence (current value)"}
    pred = dd.load_test_predictions(root)
    for c in ("cpu_pred", "mem_pred", "cpu_pred_current_value", "cpu_pred_notebook", "mem_pred_current_value", "mem_pred_notebook"):
        assert c in pred.columns
    assert (pred["cpu_pred"] == pred["cpu_pred_current_value"]).all()      # the active variant feeds the agents
    assert set(dd.load_per_machine_metrics(root)["machine_id"]) == {"m_1", "m_2", "m_3"}


def test_rerun_appends_to_log_not_overwrites(project):
    root, cfg, summary = project
    before = len(dd.load_decision_log(root))
    app.run_all(root=root, cfg=cfg, skip_data=True, skip_train=True, n_sample=30)
    after = dd.load_decision_log(root)
    assert len(after) == before + 9                      # 3 machines x 3 strategies
    assert len({r["run_id"] for r in after}) >= 2


def test_decisions_trace_to_test_predictions(project):
    root, _, _ = project
    pred = dd.load_test_predictions(root)
    latest = pred.sort_values("time_stamp").groupby("machine_id").tail(1).set_index("machine_id")
    for r in dd.load_decisions("improved", root):
        row = latest.loc[r["machine_id"]]
        assert r["current_cpu"] == pytest.approx(row["cpu_util_percent"])
        assert r["predicted_cpu"] == pytest.approx(row["cpu_pred"])
        assert r["input"]["timestamp"] == int(row["time_stamp"])


def test_missing_raw_data_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        app.pipeline.run(tmp_path, load_config())


# ---------- single-machine workflow ----------
def test_machine_store_observation_alignment(project):
    root, _, _ = project
    store = MachineStore.from_project(root)
    assert store.machines() == ["m_1", "m_2", "m_3"]
    g = store.series("m_1")
    o = store.observation("m_1", 10)
    assert o["current_cpu"] == g.loc[10, "cpu_util_percent"] and o["timestamp"] == int(g.loc[10, "time_stamp"])
    assert o["actual_next_cpu"] == g.loc[11, "cpu_util_percent"]       # the value the forecast is aimed at
    assert 0 <= o["predicted_cpu"] <= 100 and o["cpu_num"] == 96 and o["status"] == "USING"


def test_machine_store_latest_has_no_actual_next(project):
    store = MachineStore.from_project(project[0])
    o = store.observation("m_2")
    assert "actual_next_cpu" not in o and o["timestamp"] == int(store.series("m_2")["time_stamp"].iloc[-1])


def test_machine_store_errors(project):
    store = MachineStore.from_project(project[0])
    with pytest.raises(KeyError):
        store.series("m_999")
    with pytest.raises(IndexError):
        store.observation("m_1", 1)
    with pytest.raises(ValueError):
        MachineStore(pd.DataFrame({"machine_id": ["m_1"]}))


def test_machine_store_warns_on_large_time_gap(project):
    root, _, _ = project
    store = MachineStore.from_project(root)
    u = store.usage.copy()
    u.loc[u["machine_id"] == "m_1", "time_stamp"] += np.where(
        u.loc[u["machine_id"] == "m_1", "time_stamp"] >= 500, 1000, 0)
    gappy = MachineStore(u, model_dir=root / "models")
    assert "data_warning" in gappy.observation("m_1", 50)       # row 50 follows the 1010 s gap
    assert "data_warning" not in gappy.observation("m_1", 51)   # back to normal 10 s steps


def test_analyze_machine_runs_all_three_strategies(project):
    store = MachineStore.from_project(project[0])
    res = analyze_machine(store, "m_3")
    assert set(res["decisions"]) == set(STRATEGIES)
    for rec in res["decisions"].values():
        assert rec["explanation"] and rec["input"]["machine_id"] == "m_3" and set(rec["agents"]) == {
            "resource", "performance", "cost", "security"}


def test_cli_analyze_and_machines(project, capsys):
    root = project[0]
    assert app.main(["machines"], root=root) == 0
    assert "m_1" in capsys.readouterr().out
    assert app.main(["analyze", "m_1"], root=root) == 0
    out = capsys.readouterr().out
    assert "[original]" in out and "[improved]" in out and "[cost_aware]" in out


# ---------- decision evaluation ----------
def obs_grid():
    return [dict(machine_id=f"x{i}", timestamp=i, current_cpu=c, predicted_cpu=c, current_memory=91, predicted_memory=91)
            for i, c in enumerate(range(10, 96, 5))]


def test_sensitivity_scale_up_never_increases_with_higher_cpu_threshold():
    sens = ev.threshold_sensitivity(obs_grid(), cpu_high_values=(70, 80, 90), memory_high_values=(95,))
    for s in ev.STRATEGIES:
        up = sens[sens["strategy"] == s].sort_values("cpu_high")["SCALE_UP"].tolist()
        assert up == sorted(up, reverse=True), (s, up)
    assert (sens["SCALE_UP"] + sens["SCALE_DOWN"] + sens["MAINTAIN"] == sens["n"]).all()


def test_sensitivity_changes_original_strategy_results():
    sens = ev.threshold_sensitivity(obs_grid(), cpu_high_values=(60, 95), memory_high_values=(95,), strategies=("original",))
    assert sens["SCALE_UP"].nunique() > 1


def test_budget_sensitivity_cost_aware_scale_ups_grow_with_budget():
    o = [dict(machine_id=f"y{i}", timestamp=i, current_cpu=80, predicted_cpu=84, current_memory=91, predicted_memory=91)
         for i in range(3)]
    b = ev.budget_sensitivity(o, budgets=(6.0, 7.0)).set_index("budget_per_hour")
    assert b.loc[6.0, "SCALE_UP"] == 0 and b.loc[7.0, "SCALE_UP"] == 3


def test_distribution_agreement_and_votes():
    by = Orchestrator(explain=False).compare_strategies(obs_grid(), ev.STRATEGIES)
    dist = ev.decision_distribution(by)
    assert list(dist.index) == ["SCALE_UP", "SCALE_DOWN", "MAINTAIN"] and (dist.sum() == len(obs_grid())).all()
    ag = ev.strategy_agreement(by)
    assert len(ag) == 3 and ag["agreement"].between(0, 1).all()
    assert len(ev.agent_vote_summary(by["improved"])) == 3


def test_sample_observations_are_unique_and_reproducible(project):
    pred = dd.load_test_predictions(project[0])
    a = ev.sample_observations(pred, n=50, seed=1)
    b = ev.sample_observations(pred, n=50, seed=1)
    assert a == b and len({o["machine_id"] for o in a}) == 50
    assert all("@" in o["machine_id"] for o in a)


def test_bulk_mode_skips_explanations_but_normal_mode_does_not():
    o = obs_grid()[0]
    assert Orchestrator(explain=False).decide(o)["explanation"] == ""
    assert Orchestrator().decide(o)["explanation"]


# ---------- dashboard data loading ----------
@pytest.mark.parametrize("loader", [dd.load_forecast_metrics, dd.load_test_predictions, dd.load_strategy_comparison,
                                    dd.load_run_summary, dd.load_usage, lambda root=None: dd.load_decisions("improved", root)])
def test_loaders_give_helpful_error_when_results_missing(loader, tmp_path):
    with pytest.raises(dd.ResultsNotFound, match="Run"):
        loader(tmp_path)


def test_unknown_strategy_rejected(tmp_path):
    with pytest.raises(ValueError):
        dd.load_decisions("random", tmp_path)


def test_loaders_read_a_finished_run(project):
    root, _, summary = project
    assert dd.load_run_summary(root)["run_id"] == summary["run_id"] or dd.load_run_summary(root)["machines"] == 3
    assert set(dd.load_eval_table("threshold_sensitivity", root)["strategy"]) == set(STRATEGIES)
    assert len(dd.load_decision_table("cost_aware", root)) == 3
    assert dd.load_usage(root, machines=["m_1"])["machine_id"].unique().tolist() == ["m_1"]
    assert any(p.name == "decision_distribution.png" for p in dd.list_graphs(root))
    assert dd.list_graphs(root / "nope") == []
