import copy

import pytest

from src.explainability.explanation import build_explanation, check_explanation, explain_text
from src.orchestration.decision_log import append_log, read_log, verify_trace
from src.orchestration.orchestrator import Orchestrator
from src.orchestration.scenarios import SCENARIOS
from tests.test_orchestrator import Boom, obs

STRATEGIES = ("original", "improved", "cost_aware")


def rec_for(name, strategy="improved"):
    sc = [s for s in SCENARIOS if s["name"] == name][0]
    return Orchestrator().decide(sc["obs"], strategy)


# ---------- content ----------
def test_explanation_sections_present():
    d = rec_for("high_resource_demand")["explanation_detail"]
    for k in ("headline", "inputs", "votes", "agent_reasoning", "conflicts", "decision_path", "not_followed",
              "cost_impact", "security", "human_approval", "text"):
        assert k in d
    assert set(d["votes"]) == {"resource", "performance", "cost", "security"}
    assert len(d["agent_reasoning"]) == 4


def test_explanation_uses_actual_numbers():
    r = Orchestrator().decide(obs(cpu=81.2, pcpu=86.7, mem=91, pmem=92.4), "improved")
    t = r["explanation"]
    assert "81.2%" in t and "86.7%" in t and "92.4%" in t
    c = r["agents"]["cost"]
    assert f"{c['metrics']['cost_scale_up']:.2f}" in t  # cost of the SCALE_UP actually chosen


def test_explanation_changes_with_input_not_hardcoded():
    a = Orchestrator().decide(obs(cpu=85, pcpu=88), "improved")["explanation"]
    b = Orchestrator().decide(obs(cpu=12, pcpu=14), "improved")["explanation"]
    assert a != b and a.startswith("SCALE_UP") and b.startswith("SCALE_DOWN")


def test_cost_impact_matches_final_action():
    up = rec_for("high_resource_demand", "improved")["explanation_detail"]["cost_impact"]
    stay = rec_for("security_policy_violation", "improved")["explanation_detail"]["cost_impact"]
    assert up["difference"] > 0 and stay["difference"] == pytest.approx(0)
    down = rec_for("low_resource_demand", "improved")["explanation_detail"]["cost_impact"]
    assert down["difference"] < 0 and down["pricing_source"] == "SIMULATED"


def test_budget_status_in_explanation_is_correct():
    over = rec_for("cost_constraint_violation", "improved")["explanation_detail"]["cost_impact"]
    assert over["budget_status"] == "OVER_BUDGET"            # Improved scaled up past the 6.0 budget
    kept = rec_for("cost_constraint_violation", "cost_aware")["explanation_detail"]["cost_impact"]
    assert kept["budget_status"] == "WITHIN_BUDGET"          # Cost-Aware kept the current cost


def test_security_checks_reported():
    ok = rec_for("low_resource_demand")["explanation_detail"]["security"]
    assert ok["verdict"] == "APPROVED" and ok["checks_passed"] == ok["checks_total"] == 6
    bad = rec_for("security_policy_violation")["explanation_detail"]["security"]
    assert bad["verdict"] == "VIOLATION" and "machine_status" in bad["failed"]
    assert "failed: machine_status" in rec_for("security_policy_violation")["explanation"]


def test_conflicts_and_not_followed_shown():
    r = rec_for("conflicting_agents", "original")
    d = r["explanation_detail"]
    assert d["conflicts"] and any("disagree" in c.lower() or "->" in c for c in d["conflicts"])
    assert "Conflicts:" in r["explanation"]
    r2 = rec_for("conflicting_agents", "improved")
    assert r2["explanation_detail"]["not_followed"]            # Resource's scale-up vote was not followed
    assert any(x["agent"] == "Resource Agent" and x["followed"] is False
               for x in r2["explanation_detail"]["agent_reasoning"])


def test_no_conflicts_stated_when_agents_agree():
    d = rec_for("low_resource_demand")["explanation_detail"]
    assert d["conflicts"] == [] and "none between agents" in d["text"]


def test_human_approval_always_stated():
    for sc in SCENARIOS:
        for st in STRATEGIES:
            assert "Human approval is required" in Orchestrator().decide(sc["obs"], st)["explanation"]


# ---------- degraded / invalid ----------
def test_explanation_for_unavailable_agent():
    orch = Orchestrator()
    orch.agents["cost"] = Boom()
    r = orch.decide(obs(cpu=82, pcpu=85), "improved")
    assert "Cost Agent: unavailable" in r["explanation"] and "Cost impact: not available" in r["explanation"]
    assert "agent crashed" in " ".join(x["reason"] for x in r["explanation_detail"]["agent_reasoning"])


def test_explanation_for_invalid_input():
    r = Orchestrator().decide(obs(cpu=500))
    assert r["explanation"].startswith("MAINTAIN (safe default)") and r["explanation_detail"]["votes"] == {}
    assert "Input rejected" in r["explanation"]


def test_explainer_failure_never_loses_the_decision():
    def bad(_):
        raise RuntimeError("explainer broke")
    r = Orchestrator(explainer=bad).decide(obs(cpu=82, pcpu=85))
    assert r["final_action"] == "SCALE_UP" and "full explanation unavailable" in r["explanation"]


# ---------- consistency checker ----------
@pytest.mark.parametrize("sc", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_generated_explanations_are_consistent(sc, strategy):
    r = Orchestrator().decide(sc["obs"], strategy)
    assert check_explanation(r) == []


def test_checker_detects_wrong_budget_claim():
    r = rec_for("cost_constraint_violation", "improved")
    forged = r["explanation"].replace("OVER_BUDGET", "WITHIN_BUDGET")
    assert check_explanation(r, forged)


def test_checker_detects_wrong_action():
    r = rec_for("high_resource_demand", "improved")
    forged = r["explanation"].replace("SCALE_UP is recommended", "SCALE_DOWN is recommended", 1)
    assert any("final action" in p for p in check_explanation(r, forged))


def test_checker_detects_missing_vote_and_approval():
    r = rec_for("high_resource_demand", "improved")
    probs = check_explanation(r, "SCALE_UP m_x. Human approval is required.")
    assert probs
    assert any("human-approval" in p for p in check_explanation(r, r["explanation"].replace("Human approval", "Some sign-off")))


def test_checker_detects_action_security_forbids():
    r = copy.deepcopy(rec_for("security_policy_violation", "improved"))
    r["final_action"] = "SCALE_UP"
    assert any("not allowed" in p for p in check_explanation(r, r["explanation"].replace("MAINTAIN", "SCALE_UP")))


def test_explain_text_equals_detail_text():
    r = rec_for("high_memory_pressure")
    assert explain_text(r) == r["explanation_detail"]["text"] == r["explanation"]


# ---------- decision log and traceability ----------
def test_log_is_append_only_with_run_ids(tmp_path):
    path = tmp_path / "log.jsonl"
    recs, _ = Orchestrator().decide_batch([s["obs"] for s in SCENARIOS], "improved")
    run1 = append_log(recs, path)
    run2 = append_log(recs[:2], path)
    allrows = read_log(path)
    assert len(allrows) == 8 and run1 != run2
    assert len(read_log(path, run1)) == 6 and len(read_log(path, run2)) == 2
    assert read_log(tmp_path / "missing.jsonl") == []


def test_every_decision_is_traceable_to_its_input():
    for st in STRATEGIES:
        recs, _ = Orchestrator().decide_batch([s["obs"] for s in SCENARIOS], st)
        for r in recs:
            assert verify_trace(r) == [], (st, r["machine_id"])


def test_trace_detects_tampering():
    r = copy.deepcopy(rec_for("high_resource_demand"))
    r["predicted_cpu"] = 10.0
    assert any("predicted_cpu" in p for p in verify_trace(r))
    r2 = copy.deepcopy(rec_for("high_resource_demand"))
    r2["agents"]["cost"]["machine_id"] = "other"
    assert any("another machine" in p for p in verify_trace(r2))


def test_invalid_input_record_is_still_traceable():
    r = Orchestrator().decide(obs(cpu=500))
    assert verify_trace(r) == [] and r["input"]["current_cpu"] == 500


# ---------- regression: "not followed" must be stated relative to the FINAL action ----------
@pytest.mark.parametrize("sc", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_rejected_recommendations_never_equal_final_action(sc, strategy):
    r = Orchestrator().decide(sc["obs"], strategy)
    for x in r["rejected"]:
        assert x["vote"] != r["final_action"], x
        assert "not followed: not followed" not in x["reason"]
    # every agent whose vote differs from the final action is accounted for
    for k, a in r["agents"].items():
        if a and k != "security" and a["vote"] != r["final_action"]:
            assert any(x["agent"] == k and x["vote"] == a["vote"] for x in r["rejected"]), (k, a["vote"])


def test_cost_aware_budget_refusal_is_attributed_to_budget_not_cost_agent():
    r = rec_for("cost_constraint_violation", "cost_aware")
    nf = r["explanation_detail"]["not_followed"]
    assert any(x.startswith("budget (SCALE_UP)") for x in nf)
    assert not any(x.startswith("Cost Agent (MAINTAIN)") for x in nf)  # Cost Agent's MAINTAIN WAS followed
    assert "cost-aware" in " ".join(nf) or "budget" in " ".join(nf)


def test_reject_reason_mentions_security_when_it_also_forbids():
    r = rec_for("conflicting_agents", "improved")
    down = [x for x in r["rejected"] if x["vote"] == "SCALE_DOWN"]
    assert down and all("Security Agent also does not allow SCALE_DOWN" in x["reason"] for x in down)
