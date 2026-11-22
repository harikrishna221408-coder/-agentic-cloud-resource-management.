"""Robustness: malformed input, randomised invariants, determinism, monotonicity, and real-result acceptance checks."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.agents.base import InvalidObservation, validate_observation
from src.explainability.explanation import check_explanation
from src.orchestration.decision_log import verify_trace
from src.orchestration.orchestrator import Orchestrator

STRATEGIES = ("original", "improved", "cost_aware")
ACTIONS = {"SCALE_UP", "SCALE_DOWN", "MAINTAIN"}
BASE = dict(machine_id="m_1", current_cpu=40, predicted_cpu=40, current_memory=91, predicted_memory=91)
RANK = {"SCALE_DOWN": 0, "MAINTAIN": 1, "SCALE_UP": 2}


# ---------- malformed input: rejected explicitly, never silently accepted ----------
REJECTED = {
    "bool as number": dict(BASE, current_cpu=True),
    "numeric string": dict(BASE, current_cpu="50"),
    "string with spaces": dict(BASE, current_cpu=" 50 "),
    "inf": dict(BASE, predicted_cpu=float("inf")),
    "nan": dict(BASE, predicted_memory=float("nan")),
    "list value": dict(BASE, current_cpu=[1, 2]),
    "above 100": dict(BASE, current_cpu=100.01),
    "below 0": dict(BASE, current_memory=-0.01),
    "empty machine id": dict(BASE, machine_id=""),
    "blank machine id": dict(BASE, machine_id="   "),
    "None machine id": dict(BASE, machine_id=None),
    "bool machine id": dict(BASE, machine_id=True),
    "list machine id": dict(BASE, machine_id=["m_1"]),
    "cpu_num zero": dict(BASE, cpu_num=0),
    "cpu_num negative": dict(BASE, cpu_num=-5),
    "cpu_num string": dict(BASE, cpu_num="many"),
    "cpu_num NaN": dict(BASE, cpu_num=float("nan")),
    "status not a string": dict(BASE, status=5),
    "empty dict": {}, "list": [1, 2, 3], "None": None, "string": "m_1",
}
ACCEPTED = {
    "boundary 0": dict(BASE, current_cpu=0),
    "boundary 100": dict(BASE, current_cpu=100, predicted_cpu=100),
    "negative zero": dict(BASE, current_cpu=-0.0),
    "numpy float": dict(BASE, current_cpu=np.float32(55.5)),
    "numpy int": dict(BASE, current_cpu=np.int64(55)),
    "int machine id": dict(BASE, machine_id=7),
    "unicode machine id": dict(BASE, machine_id="m_é☁"),
    "unknown extra field": dict(BASE, foo="bar"),
    "cpu_num given": dict(BASE, cpu_num=48),
    "cpu_num None means absent": dict(BASE, cpu_num=None),
}


@pytest.mark.parametrize("name", REJECTED)
def test_malformed_input_rejected_by_validator(name):
    with pytest.raises(InvalidObservation):
        validate_observation(REJECTED[name])


@pytest.mark.parametrize("name", REJECTED)
def test_malformed_input_gives_safe_maintain_decision(name):
    r = Orchestrator().decide(REJECTED[name])
    assert r["decision_status"] == "INVALID_INPUT" and r["final_action"] == "MAINTAIN"
    assert r["requires_review"] is True and r["human_approval_required"] is True
    assert "Input rejected" in r["explanation"] and r["explanation"]
    assert verify_trace(r) == []
    json.dumps(r, default=str)                      # the record can always be saved


@pytest.mark.parametrize("name", ACCEPTED)
def test_valid_but_unusual_input_is_accepted(name):
    r = Orchestrator().decide(ACCEPTED[name])
    assert r["decision_status"] == "OK" and r["final_action"] in ACTIONS
    assert check_explanation(r) == [] and verify_trace(r) == []
    cost = r["agents"]["cost"]
    assert np.isfinite(cost["current_cost"]) and cost["current_cost"] > 0


# ---------- randomised invariants (fixed seed, so the test is reproducible) ----------
def random_observations(n, seed):
    rng = np.random.default_rng(seed)
    edge = [0, 100, 25, 30, 70, 80, 85, 93, 95, 96, 98, 24.999, 80.0001, 94.9999]

    def pick():
        return float(rng.choice(edge)) if rng.random() < 0.35 else float(rng.uniform(0, 100))
    for i in range(n):
        o = dict(machine_id=f"f{i}", current_cpu=pick(), predicted_cpu=pick(), current_memory=pick(), predicted_memory=pick())
        if rng.random() < 0.15:
            o["status"] = str(rng.choice(["USING", "IMPORT_INSTALLING", "OTHER"]))
        if rng.random() < 0.15:
            o["cpu_num"] = int(rng.choice([1, 8, 96, 512]))
        yield o


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_invariants_hold_on_random_observations(strategy):
    orch = Orchestrator()
    for o in random_observations(400, seed=11):
        snapshot = copy.deepcopy(o)
        r = orch.decide(o, strategy)
        assert o == snapshot, "input was modified"
        assert r["final_action"] in ACTIONS and r["decision_status"] == "OK" and not r["agent_errors"]
        sec = r["agents"]["security"]
        assert r["final_action"] in sec["allowed_actions"], (o, r["final_action"], sec["allowed_actions"])
        assert check_explanation(r) == [], (o, check_explanation(r))
        assert verify_trace(r) == [], o
        assert all(x["vote"] != r["final_action"] for x in r["rejected"])
        # a scale-down is never chosen while the Resource or Performance Agent asks for more capacity
        if r["final_action"] == "SCALE_DOWN":
            assert all(r["agents"][k]["vote"] != "SCALE_UP" for k in ("resource", "performance")), o


def test_random_observations_exercise_every_action_and_conflicts():
    orch = Orchestrator(explain=False)
    seen, conflicts = {s: set() for s in STRATEGIES}, 0
    for o in random_observations(600, seed=3):
        for s in STRATEGIES:
            r = orch.decide(o, s)
            seen[s].add(r["final_action"])
            conflicts += bool(r["conflicts"]) and s == "improved"
    assert all(v == ACTIONS for v in seen.values()), seen        # the stress test really reaches all outcomes
    assert conflicts > 50


def test_decisions_are_deterministic():
    orch = Orchestrator()
    for o in random_observations(40, seed=5):
        for s in STRATEGIES:
            a, b = orch.decide(o, s), orch.decide(o, s)
            for k in ("final_action", "explanation", "rationale", "conflicts", "rejected", "requires_review"):
                assert a[k] == b[k]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_more_cpu_never_means_less_capacity(strategy):
    """Sweeping CPU from 0 to 100 (memory typical): the action only moves DOWN -> MAINTAIN -> UP, never backwards."""
    orch = Orchestrator(explain=False)
    seq = [RANK[orch.decide(dict(BASE, current_cpu=c, predicted_cpu=c), strategy)["final_action"]]
           for c in np.arange(0, 100.5, 0.5)]
    assert seq == sorted(seq) and seq[0] == 0 and seq[-1] == 2


def test_security_restriction_is_never_overridden_by_any_strategy():
    for o in random_observations(300, seed=21):
        o["status"] = "IMPORT_INSTALLING"
        for s in STRATEGIES:
            assert Orchestrator(explain=False).decide(o, s)["final_action"] == "MAINTAIN"


# ---------- acceptance checks on the REAL results (skipped if the full run has not been done) ----------
REAL = Path(__file__).resolve().parents[1]
needs_real = pytest.mark.skipif(not (REAL / "results" / "decisions" / "decisions_improved.jsonl").exists(),
                                reason="real run results not present")


@needs_real
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_real_results_every_machine_has_one_explained_traceable_decision(strategy):
    from src import dashboard_data as dd
    recs = dd.load_decisions(strategy, REAL)
    ids = [r["machine_id"] for r in recs]
    selected = sorted(dd.load_per_machine_metrics(REAL)["machine_id"])
    assert sorted(ids) == selected and len(selected) == 20            # every selected machine, exactly once
    for r in recs:
        assert r["explanation"] and r["final_action"] in ACTIONS
        assert r["human_approval_required"] and r["approval_status"] == "PENDING"
        assert check_explanation(r) == [] and verify_trace(r) == []
        assert set(r["agents"]) == {"resource", "performance", "cost", "security"}
        assert r["agents"]["cost"]["pricing_source"] == "SIMULATED"


@needs_real
def test_real_results_decisions_trace_to_forecast_rows():
    from src import dashboard_data as dd
    pred = dd.load_test_predictions(REAL).set_index(["machine_id", "time_stamp"])
    for r in dd.load_decisions("original", REAL):
        row = pred.loc[(r["machine_id"], r["input"]["timestamp"])]
        assert r["current_cpu"] == pytest.approx(row["cpu_util_percent"])
        assert r["predicted_cpu"] == pytest.approx(row["cpu_pred"])
        assert r["predicted_memory"] == pytest.approx(row["mem_pred"])


@needs_real
def test_real_forecast_metrics_match_a_fresh_recomputation():
    from src import dashboard_data as dd
    from src.forecasting.evaluation import mae_rmse
    pred, comp = dd.load_test_predictions(REAL), dd.load_forecast_metrics(REAL)
    mae, rmse = mae_rmse(pred["cpu_target"], pred["cpu_pred"])
    row = comp[(comp["Resource"] == "CPU") & (comp["Model"] == "Random Forest + current value")].iloc[0]
    assert row["MAE"] == pytest.approx(mae) and row["RMSE"] == pytest.approx(rmse)
    mae, _ = mae_rmse(pred["mem_target"], pred["mem_util_percent"])
    row = comp[(comp["Resource"] == "Memory") & (comp["Model"] == "Persistence (current value)")].iloc[0]
    assert row["MAE"] == pytest.approx(mae)


@needs_real
def test_real_split_is_chronological_per_machine():
    import pandas as pd
    from src.data.preprocessing import chronological_split
    df = pd.read_csv(REAL / "data" / "processed" / "final_forecasting_dataset.csv")
    train, test = chronological_split(df, 0.8)
    last_train, first_test = train.groupby("machine_id")["time_stamp"].max(), test.groupby("machine_id")["time_stamp"].min()
    assert (last_train < first_test).all() and len(train) + len(test) == len(df)
    assert abs(len(train) / len(df) - 0.8) < 0.001


@needs_real
def test_performance_margins_match_the_active_models_saved_rmse():
    """The Performance Agent adds the forecast error to predictions; the config must hold the RMSE of the model in use."""
    from src import dashboard_data as dd
    from src.agents.base import load_agent_config
    from src.forecasting.evaluation import VARIANT_NAMES
    from src.forecasting.predict import active_variant
    comp = dd.load_forecast_metrics(REAL).set_index(["Resource", "Model"])
    name = VARIANT_NAMES[active_variant()]
    cfg = load_agent_config()["performance_agent"]
    assert cfg["cpu_margin"] == pytest.approx(comp.loc[("CPU", name), "RMSE"], abs=0.01)
    assert cfg["memory_margin"] == pytest.approx(comp.loc[("Memory", name), "RMSE"], abs=0.01)
