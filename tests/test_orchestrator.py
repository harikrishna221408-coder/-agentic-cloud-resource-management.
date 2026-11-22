import json
import pandas as pd
import pytest

from src.orchestration.orchestrator import (Orchestrator, observations_from_predictions, save_records,
                                            validate_agent_output)
from src.orchestration.negotiation import detect_conflicts
from src.orchestration.coordinators import get_coordinator


def obs(mid="m_1", cpu=40, pcpu=40, mem=90, pmem=90, **extra):
    d = dict(machine_id=mid, current_cpu=cpu, predicted_cpu=pcpu, current_memory=mem, predicted_memory=pmem)
    d.update(extra)
    return d


class Boom:
    def evaluate(self, o):
        raise RuntimeError("agent crashed")


class Malformed:
    def evaluate(self, o):
        return {"agent": "x", "vote": "EXPLODE"}


class Fixed:
    """Fake agent with a fixed output, for testing coordinator logic in isolation."""
    def __init__(self, vote, **extra):
        self.out = dict(agent="Fake", machine_id="m_1", status="X", vote=vote, risk_level="LOW",
                        reason="fixed", metrics={}, **extra)

    def evaluate(self, o):
        return dict(self.out)


def fake_sec(allowed=("SCALE_UP", "SCALE_DOWN", "MAINTAIN"), review=False):
    return Fixed("APPROVE", allowed_actions=list(allowed), alerts=[], requires_review=review)


# ---------- record contents / traceability ----------
def test_record_contains_all_required_fields():
    r = Orchestrator().decide(obs())
    for k in ("machine_id", "current_cpu", "predicted_cpu", "current_memory", "predicted_memory",
              "agents", "coordinator_strategy", "final_action", "explanation", "timestamp"):
        assert k in r
    assert set(r["agents"]) == {"resource", "performance", "cost", "security"}
    assert all(r["agents"][k] is not None for k in r["agents"])
    assert r["final_action"] in ("SCALE_UP", "SCALE_DOWN", "MAINTAIN")
    assert r["human_approval_required"] is True and r["approval_status"] == "PENDING"


def test_record_traces_to_input():
    o = obs(pcpu=83.3, timestamp=12345)
    r = Orchestrator().decide(o)
    assert r["input"]["predicted_cpu"] == 83.3 and r["input"]["timestamp"] == 12345
    assert r["agents"]["resource"]["predicted_cpu"] == 83.3
    assert r["decision_id"].endswith("m_1:12345")


def test_high_demand_scales_up():
    r = Orchestrator().decide(obs(cpu=82, pcpu=85))
    assert r["final_action"] == "SCALE_UP"


def test_input_dict_not_mutated():
    o = obs()
    snap = dict(o)
    Orchestrator().decide(o)
    assert o == snap


# ---------- invalid input / agent failures ----------
def test_invalid_input_gives_safe_decision():
    r = Orchestrator().decide(obs(cpu=500))
    assert r["decision_status"] == "INVALID_INPUT" and r["final_action"] == "MAINTAIN"
    assert r["requires_review"] is True
    assert "Input rejected" in r["explanation"]


def test_none_and_garbage_input_do_not_crash():
    o = Orchestrator()
    assert o.decide(None)["final_action"] == "MAINTAIN"
    assert o.decide({"machine_id": "m_9"})["decision_status"] == "INVALID_INPUT"


def test_crashing_agent_is_isolated_and_reported():
    orch = Orchestrator()
    orch.agents["cost"] = Boom()
    r = orch.decide(obs(cpu=82, pcpu=85))
    assert r["agents"]["cost"] is None
    assert r["agent_errors"][0]["agent"] == "cost" and "crashed" in r["agent_errors"][0]["error"]
    assert r["decision_status"] == "DEGRADED" and r["final_action"] == "SCALE_UP"
    assert "Cost Agent" in r["explanation"]


def test_malformed_agent_output_treated_as_unavailable():
    orch = Orchestrator()
    orch.agents["performance"] = Malformed()
    r = orch.decide(obs())
    assert r["agents"]["performance"] is None
    assert "invalid output" in r["agent_errors"][0]["error"]


def test_missing_security_agent_forces_maintain_and_review():
    orch = Orchestrator()
    del orch.agents["security"]
    r = orch.decide(obs(cpu=82, pcpu=85))  # would be SCALE_UP
    assert r["final_action"] == "MAINTAIN" and r["requires_review"] and r["degraded"]
    assert r["rejected"][0]["vote"] == "SCALE_UP"


def test_validate_agent_output():
    assert validate_agent_output("resource", {"agent": "a"}) is not None
    assert validate_agent_output("resource", "text") is not None
    good = Fixed("SCALE_UP").evaluate({})
    assert validate_agent_output("resource", good) is None
    assert validate_agent_output("security", good) is not None  # no allowed_actions


# ---------- Original coordinator rules (isolated with fake agents) ----------
def orch_with(res, perf, cost, sec):
    return Orchestrator(agents={"resource": Fixed(res), "performance": Fixed(perf), "cost": Fixed(cost), "security": sec})


def test_original_any_scale_up_wins():
    r = orch_with("SCALE_UP", "MAINTAIN", "SCALE_DOWN", fake_sec()).decide(obs())
    assert r["final_action"] == "SCALE_UP"


def test_original_scale_down_needs_both():
    assert orch_with("SCALE_DOWN", "SCALE_DOWN", "MAINTAIN", fake_sec()).decide(obs())["final_action"] == "SCALE_DOWN"
    assert orch_with("SCALE_DOWN", "MAINTAIN", "SCALE_DOWN", fake_sec()).decide(obs())["final_action"] == "MAINTAIN"


def test_original_security_overrides():
    sec = fake_sec(allowed=("SCALE_UP", "MAINTAIN"))
    r = orch_with("SCALE_DOWN", "SCALE_DOWN", "SCALE_DOWN", sec).decide(obs())
    assert r["final_action"] == "MAINTAIN"
    assert any(x["agent"] == "security" and x["vote"] == "SCALE_DOWN" for x in r["rejected"])


def test_original_ignores_cost_and_says_so():
    r = orch_with("SCALE_UP", "SCALE_UP", "MAINTAIN", fake_sec()).decide(obs())
    assert r["final_action"] == "SCALE_UP"
    assert any(x["agent"] == "cost" and "not used" in x["reason"] for x in r["rejected"])


# ---------- conflicts ----------
def test_conflicts_detected():
    outs = {"resource": Fixed("SCALE_UP").evaluate({}), "performance": Fixed("SCALE_DOWN").evaluate({}),
            "cost": None, "security": fake_sec(allowed=("MAINTAIN",)).evaluate({})}
    types = {c["type"] for c in detect_conflicts(outs)}
    assert "VOTE_DISAGREEMENT" in types and "SECURITY_RESTRICTION" in types


def test_no_conflict_when_agents_agree():
    outs = {k: Fixed("MAINTAIN").evaluate({}) for k in ("resource", "performance", "cost")}
    outs["security"] = fake_sec().evaluate({})
    assert detect_conflicts(outs) == []


def test_unknown_coordinator():
    with pytest.raises(ValueError):
        get_coordinator("magic")


# ---------- batch ----------
def batch_obs():
    return [obs("m_1", 82, 85), obs("m_2", 15, 18), obs("m_3", 40, 40), obs("m_4", 30, 30)]


def test_batch_every_machine_gets_one_decision_with_explanation():
    recs, skipped = Orchestrator().decide_batch(batch_obs())
    assert [r["machine_id"] for r in recs] == ["m_1", "m_2", "m_3", "m_4"] and skipped == []
    assert all(r["explanation"] and r["final_action"] for r in recs)


def test_batch_duplicates_are_skipped_not_decided_twice():
    recs, skipped = Orchestrator().decide_batch(batch_obs() + [obs("m_1", 10, 10)])
    assert len(recs) == 4 and skipped[0]["machine_id"] == "m_1"
    assert len({r["machine_id"] for r in recs}) == 4


def test_batch_with_invalid_machine_still_returns_a_decision():
    recs, _ = Orchestrator().decide_batch(batch_obs() + [obs("m_9", cpu=-5)])
    bad = [r for r in recs if r["machine_id"] == "m_9"][0]
    assert bad["final_action"] == "MAINTAIN" and bad["decision_status"] == "INVALID_INPUT"


# ---------- persistence / observations ----------
def test_save_records_roundtrip(tmp_path):
    recs, _ = Orchestrator().decide_batch(batch_obs())
    jl, csv = save_records(recs, tmp_path, "t")
    with open(jl, encoding="utf-8") as f:
        loaded = [json.loads(line) for line in f]
    assert len(loaded) == 4 and loaded[0]["machine_id"] == "m_1"
    flat = pd.read_csv(csv)
    assert len(flat) == 4 and flat["explanation"].notna().all()


def test_observations_from_predictions():
    pred = pd.DataFrame({"machine_id": ["m_1", "m_1", "m_2"], "time_stamp": [10, 20, 20],
                         "cpu_util_percent": [30, 31, 50], "mem_util_percent": [90, 91, 92],
                         "cpu_pred": [32, 33, 51], "mem_pred": [90.5, 91.5, 92.5]})
    meta = pd.DataFrame({"machine_id": ["m_1", "m_2"], "time_stamp": [0, 0], "cpu_num": [96, 96], "status": ["USING"] * 2})
    out = observations_from_predictions(pred, meta)
    assert len(out) == 2
    m1 = [o for o in out if o["machine_id"] == "m_1"][0]
    assert m1["timestamp"] == 20 and m1["current_cpu"] == 31 and m1["cpu_num"] == 96
