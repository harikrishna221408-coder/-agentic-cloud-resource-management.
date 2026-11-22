import copy
import pytest

from src.agents.base import InvalidObservation, load_agent_config, validate_observation
from src.agents.resource_agent import ResourceAgent
from src.agents.performance_agent import PerformanceAgent
from src.agents.cost_agent import CostAgent
from src.agents.security_agent import SecurityAgent

AGENTS = [ResourceAgent, PerformanceAgent, CostAgent, SecurityAgent]


def obs(cpu=40, pcpu=40, mem=90, pmem=90, **extra):
    d = dict(machine_id="m_1", current_cpu=cpu, predicted_cpu=pcpu, current_memory=mem, predicted_memory=pmem)
    d.update(extra)
    return d


# ---------- shared contract ----------
@pytest.mark.parametrize("Agent", AGENTS)
def test_output_contract(Agent):
    out = Agent().evaluate(obs())
    for key in ("agent", "machine_id", "status", "vote", "risk_level", "reason", "metrics"):
        assert key in out
    assert out["machine_id"] == "m_1" and out["reason"]
    assert out["risk_level"] in ("LOW", "MEDIUM", "HIGH")


@pytest.mark.parametrize("Agent", AGENTS)
@pytest.mark.parametrize("bad", [
    {"current_cpu": None}, {"current_cpu": "abc"}, {"predicted_cpu": 120},
    {"current_memory": -1}, {"predicted_memory": float("nan")},
])
def test_invalid_input_rejected(Agent, bad):
    with pytest.raises(InvalidObservation):
        Agent().evaluate(obs(**bad))


@pytest.mark.parametrize("Agent", AGENTS)
def test_missing_field_rejected(Agent):
    o = obs()
    del o["predicted_cpu"]
    with pytest.raises(InvalidObservation):
        Agent().evaluate(o)
    with pytest.raises(InvalidObservation):
        Agent().evaluate("not a dict")


def test_input_not_mutated():
    o = obs()
    before = copy.deepcopy(o)
    for A in AGENTS:
        A().evaluate(o)
    assert o == before


# ---------- Resource Agent ----------
def test_resource_high_cpu_scales_up():
    out = ResourceAgent().evaluate(obs(cpu=60, pcpu=85))
    assert out["status"] == "CPU_PRESSURE" and out["vote"] == "SCALE_UP"


def test_resource_uses_current_value_too():
    # forecast falls but current is still above the limit -> still pressure
    assert ResourceAgent().evaluate(obs(cpu=88, pcpu=70))["vote"] == "SCALE_UP"


def test_resource_memory_pressure():
    out = ResourceAgent().evaluate(obs(pmem=96))
    assert out["status"] == "MEMORY_PRESSURE" and out["vote"] == "SCALE_UP"


def test_resource_typical_memory_is_not_pressure():
    # ~91% is this dataset's normal memory level; must not trigger pressure
    assert ResourceAgent().evaluate(obs(mem=91, pmem=91.5))["status"] == "NORMAL"


def test_resource_overallocated_scales_down():
    out = ResourceAgent().evaluate(obs(cpu=15, pcpu=18, mem=90, pmem=90))
    assert out["status"] == "OVERALLOCATED" and out["vote"] == "SCALE_DOWN"


def test_resource_low_current_but_rising_forecast_not_overallocated():
    assert ResourceAgent().evaluate(obs(cpu=15, pcpu=40))["vote"] == "MAINTAIN"


def test_resource_both_pressure():
    assert ResourceAgent().evaluate(obs(pcpu=90, pmem=97))["status"] == "BOTH_PRESSURE"


# ---------- Performance Agent ----------
def test_performance_high_risk_scales_up():
    out = PerformanceAgent().evaluate(obs(pcpu=82))  # upper = 82 + 4.32 = 86.32 >= 85
    assert out["risk_level"] == "HIGH" and out["vote"] == "SCALE_UP"


def test_performance_uses_forecast_margin():
    # margin 4.32: HIGH risk starts where pred + 4.32 >= 85, i.e. pred >= 80.68; MEDIUM where pred + 4.32 >= 70
    assert PerformanceAgent().evaluate(obs(pcpu=80.7))["risk_level"] == "HIGH"      # 85.02
    assert PerformanceAgent().evaluate(obs(pcpu=80.6))["risk_level"] == "MEDIUM"    # 84.92
    assert PerformanceAgent().evaluate(obs(pcpu=70))["risk_level"] == "MEDIUM"      # 74.32
    assert PerformanceAgent().evaluate(obs(pcpu=65))["risk_level"] == "LOW"         # 69.32


def test_performance_safe_to_scale_down():
    out = PerformanceAgent().evaluate(obs(pcpu=20, pmem=90))  # upper 24.32 -> projected 32.4
    assert out["vote"] == "SCALE_DOWN" and out["status"] == "SAFE_TO_REDUCE"


def test_performance_no_scale_down_if_projection_too_high():
    # upper 54.32 / 0.75 = 72.4 >= 70 -> must keep capacity even though risk is LOW
    out = PerformanceAgent().evaluate(obs(pcpu=50))
    assert out["risk_level"] == "LOW" and out["vote"] == "MAINTAIN"


def test_performance_memory_risk():
    assert PerformanceAgent().evaluate(obs(pmem=97.5))["memory_pressure"] == "HIGH"   # 97.5 + 0.64 = 98.14 >= 98


# ---------- Cost Agent ----------
def test_cost_numbers_are_traceable():
    cfg = load_agent_config()
    c = cfg["cost_agent"]
    base = 96 * c["vcpu_hourly_price"] + c["simulated_memory_gb"] * c["memory_gb_hourly_price"]
    out = CostAgent().evaluate(obs())
    assert out["current_cost"] == pytest.approx(base)
    assert out["metrics"]["cost_scale_up"] == pytest.approx(base * 1.25)
    assert out["metrics"]["cost_scale_down"] == pytest.approx(base * 0.75)
    assert out["pricing_source"] == "SIMULATED"


def test_cost_uses_machine_metadata_cpu_num():
    a = CostAgent().evaluate(obs())["current_cost"]
    b = CostAgent().evaluate(obs(cpu_num=48))["current_cost"]
    assert b < a


def test_cost_waste_detected_scale_down():
    out = CostAgent().evaluate(obs(pcpu=20))
    assert out["status"] == "WASTE_DETECTED" and out["vote"] == "SCALE_DOWN"
    assert out["cost_difference"] < 0


def test_cost_never_votes_scale_up_but_reports_budget():
    out = CostAgent().evaluate(obs(pcpu=95))
    assert out["vote"] != "SCALE_UP"
    assert out["scale_up_within_budget"] is False  # 6.40 > 6.0 with default simulated budget


def test_cost_budget_configurable():
    cfg = load_agent_config()["cost_agent"]
    cfg["budget_per_machine_hour"] = 100
    out = CostAgent(config=cfg).evaluate(obs(pcpu=95))
    assert out["scale_up_within_budget"] is True


def test_cost_assess_action_over_budget():
    a = CostAgent().assess_action(obs(), "SCALE_UP")
    assert a["budget_status"] == "OVER_BUDGET" and a["cost_difference"] > 0
    with pytest.raises(ValueError):
        CostAgent().assess_action(obs(), "DELETE")


def test_cost_over_budget_current_spend():
    cfg = load_agent_config()["cost_agent"]
    cfg["budget_per_machine_hour"] = 1.0
    assert CostAgent(config=cfg).evaluate(obs())["status"] == "OVER_BUDGET"


# ---------- Security Agent ----------
def test_security_all_clear():
    out = SecurityAgent().evaluate(obs(pcpu=40, pmem=90))
    assert out["verdict"] == "APPROVED" and set(out["allowed_actions"]) == {"SCALE_UP", "SCALE_DOWN", "MAINTAIN"}


def test_security_blocks_scale_down_on_high_memory():
    out = SecurityAgent().evaluate(obs(pcpu=20, pmem=95))
    assert "SCALE_DOWN" in out["blocked_actions"] and out["verdict"] == "RESTRICTED"
    assert "SCALE_UP" in out["allowed_actions"]


def test_security_blocks_scale_down_on_high_projected_cpu():
    assert "SCALE_DOWN" in SecurityAgent().evaluate(obs(pcpu=60))["blocked_actions"]  # 60/0.75 = 80 >= 75


def test_security_status_violation_only_maintain():
    out = SecurityAgent().evaluate(obs(status="IMPORT_INSTALLING"))
    assert out["verdict"] == "VIOLATION" and out["allowed_actions"] == ["MAINTAIN"] and out["requires_review"]


def test_security_unusual_change_needs_review():
    out = SecurityAgent().evaluate(obs(cpu=20, pcpu=50))
    assert out["verdict"] == "REVIEW_REQUIRED" and "UNUSUAL_CPU_CHANGE" in out["alerts"]


def test_security_protected_machine_review():
    cfg = load_agent_config()["security_agent"]
    cfg["protected_machines"] = ["m_1"]
    assert SecurityAgent(config=cfg).evaluate(obs())["verdict"] == "REVIEW_REQUIRED"


def test_security_step_too_large_review():
    out = SecurityAgent(step_fraction=0.8).evaluate(obs())
    assert "STEP_TOO_LARGE" in out["alerts"]


def test_security_policy_checks_listed():
    names = {c["check"] for c in SecurityAgent().evaluate(obs())["policy_checks"]}
    assert {"machine_status", "scale_down_memory", "scale_down_cpu", "unusual_change", "protected_machine"} <= names


# ---------- independence ----------
def votes_for(o):
    return {A.__name__: A().evaluate(o)["vote"] for A in (ResourceAgent, PerformanceAgent, CostAgent)}


def test_agents_can_disagree_low_load():
    # CPU 27%: Resource (needs < 25%) says MAINTAIN; Performance and Cost see room/waste to reduce.
    v = votes_for(obs(cpu=27, pcpu=27))
    assert v == {"ResourceAgent": "MAINTAIN", "PerformanceAgent": "SCALE_DOWN", "CostAgent": "SCALE_DOWN"}


def test_agents_can_disagree_near_limit():
    # CPU 80.2%: Resource (limit 80%) sees pressure; Performance sees only MEDIUM risk (80.2 + 4.32 = 84.52 < 85) and
    # does not ask for a scale-up.
    v = votes_for(obs(cpu=79, pcpu=80.2))
    assert v["ResourceAgent"] == "SCALE_UP" and v["PerformanceAgent"] == "MAINTAIN"
