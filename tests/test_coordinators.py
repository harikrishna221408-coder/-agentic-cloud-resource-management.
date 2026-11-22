import pytest

from src.agents.base import load_agent_config
from src.orchestration.coordinators import (CostAwareCoordinator, ImprovedCoordinator, OriginalCoordinator,
                                            get_coordinator)
from src.orchestration.orchestrator import Orchestrator, comparison_table
from src.orchestration.scenarios import SCENARIOS
from tests.test_orchestrator import Fixed, fake_sec, obs, Boom

STRATEGIES = ("original", "improved", "cost_aware")


# ---------- the six required scenarios, with expected outcomes defined in advance ----------
@pytest.mark.parametrize("sc", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_scenario_expected_outcome(sc, strategy):
    rec = Orchestrator().decide(sc["obs"], strategy)
    assert rec["final_action"] == sc["expected"][strategy], (rec["explanation"], sc["why"])
    assert rec["explanation"] and rec["human_approval_required"]


def test_scenarios_cover_all_three_actions_and_all_strategies_differ_somewhere():
    outcomes = {a for s in SCENARIOS for a in s["expected"].values()}
    assert outcomes == {"SCALE_UP", "SCALE_DOWN", "MAINTAIN"}
    assert any(len(set(s["expected"].values())) > 1 for s in SCENARIOS)


def test_security_violation_flags_review():
    sc = [s for s in SCENARIOS if s["name"] == "security_policy_violation"][0]
    for st in STRATEGIES:
        assert Orchestrator().decide(sc["obs"], st)["requires_review"] is True


# ---------- Improved coordinator ----------
def run(strategy, res, perf, cost, sec, perf_extra=None):
    perf_agent = Fixed(perf, **(perf_extra or {"cpu_pressure": "LOW", "memory_pressure": "LOW"}))
    perf_agent.out["metrics"] = {"upper_cpu": 30.0, "upper_memory": 91.0}
    return Orchestrator(agents={"resource": Fixed(res), "performance": perf_agent, "cost": Fixed(cost), "security": sec}
                        ).decide(obs(), strategy)


def test_improved_uncorroborated_scale_up_is_not_followed():
    r = run("improved", "SCALE_UP", "SCALE_DOWN", "SCALE_DOWN", fake_sec())
    assert r["final_action"] == "MAINTAIN" and "Conflict" in r["explanation"]


def test_improved_performance_scale_up_wins_over_resource_maintain():
    assert run("improved", "MAINTAIN", "SCALE_UP", "MAINTAIN", fake_sec())["final_action"] == "SCALE_UP"


def test_improved_scale_down_needs_support():
    assert run("improved", "MAINTAIN", "SCALE_DOWN", "MAINTAIN", fake_sec())["final_action"] == "MAINTAIN"
    assert run("improved", "MAINTAIN", "SCALE_DOWN", "SCALE_DOWN", fake_sec())["final_action"] == "SCALE_DOWN"
    assert run("improved", "SCALE_DOWN", "SCALE_DOWN", "MAINTAIN", fake_sec())["final_action"] == "SCALE_DOWN"


def test_improved_never_scales_down_with_missing_agent():
    orch = Orchestrator()
    orch.agents["cost"] = Boom()
    r = orch.decide(obs(cpu=12, pcpu=14, mem=88, pmem=88), "improved")
    assert r["final_action"] == "MAINTAIN" and r["degraded"]


def test_improved_security_has_final_say():
    sec = fake_sec(allowed=("MAINTAIN",))
    assert run("improved", "SCALE_UP", "SCALE_UP", "MAINTAIN", sec)["final_action"] == "MAINTAIN"


def test_improved_accepts_resource_scale_up_if_performance_missing():
    orch = Orchestrator()
    orch.agents["performance"] = Boom()
    r = orch.decide(obs(cpu=82, pcpu=85), "improved")
    assert r["final_action"] == "SCALE_UP" and r["degraded"]


# ---------- Cost-Aware coordinator ----------
def test_cost_aware_equals_improved_when_budget_is_ample():
    cfg = load_agent_config()["cost_agent"]
    cfg["budget_per_machine_hour"] = 100
    from src.agents.cost_agent import CostAgent
    orch = Orchestrator()
    orch.agents["cost"] = CostAgent(config=cfg)
    o = obs(cpu=80, pcpu=84)
    assert orch.decide(o, "cost_aware")["final_action"] == orch.decide(o, "improved")["final_action"] == "SCALE_UP"
    assert "within the budget" in orch.decide(o, "cost_aware")["explanation"]


def test_cost_aware_budget_override_only_when_critical():
    assert Orchestrator().decide(obs(cpu=80, pcpu=84), "cost_aware")["final_action"] == "MAINTAIN"
    r = Orchestrator().decide(obs(cpu=85, pcpu=88), "cost_aware")
    assert r["final_action"] == "SCALE_UP" and r["requires_review"] and "overridden" in r["explanation"]


def test_cost_aware_without_cost_agent_approves_no_spend():
    orch = Orchestrator()
    orch.agents["cost"] = Boom()
    r = orch.decide(obs(cpu=85, pcpu=88), "cost_aware")
    assert r["final_action"] == "MAINTAIN" and "cannot be checked" in r["explanation"]


def test_cost_aware_scales_down_when_current_spend_exceeds_budget():
    cfg = load_agent_config()["cost_agent"]
    cfg["budget_per_machine_hour"] = 4.0   # baseline cost 5.12 is now over budget
    from src.agents.cost_agent import CostAgent
    orch = Orchestrator()
    orch.agents["cost"] = CostAgent(config=cfg)
    # CPU 55%: Performance projects 80% after a scale-down (>= 70), so Improved keeps capacity;
    # Security still allows it (73% < 75%), so Cost-Aware trims it to meet the budget.
    o = obs(cpu=55, pcpu=55, mem=90, pmem=90)
    assert orch.decide(o, "improved")["final_action"] == "MAINTAIN"
    r = orch.decide(o, "cost_aware")
    assert r["final_action"] == "SCALE_DOWN" and "exceeds the budget" in r["explanation"]


def test_cost_aware_respects_security_when_over_budget():
    cfg = load_agent_config()["cost_agent"]
    cfg["budget_per_machine_hour"] = 4.0
    from src.agents.cost_agent import CostAgent
    orch = Orchestrator()
    orch.agents["cost"] = CostAgent(config=cfg)
    # memory 95 -> Security blocks SCALE_DOWN
    assert orch.decide(obs(cpu=45, pcpu=45, mem=95, pmem=95), "cost_aware")["final_action"] != "SCALE_DOWN"


# ---------- comparison ----------
def test_strategies_differ_on_conflict_data():
    observations = [s["obs"] for s in SCENARIOS]
    by = Orchestrator().compare_strategies(observations, STRATEGIES)
    table = comparison_table(by)
    assert list(table.columns) == ["machine_id", *STRATEGIES, "all_agree"]
    assert len(table) == 6 and (~table["all_agree"]).any() and table["all_agree"].any()


def test_every_decision_in_every_strategy_has_explanation_and_valid_action():
    observations = [s["obs"] for s in SCENARIOS]
    for st in STRATEGIES:
        recs, _ = Orchestrator().decide_batch(observations, st)
        assert len({r["machine_id"] for r in recs}) == 6
        for r in recs:
            assert r["final_action"] in ("SCALE_UP", "SCALE_DOWN", "MAINTAIN") and r["explanation"]
            assert r["coordinator_strategy"] == st
