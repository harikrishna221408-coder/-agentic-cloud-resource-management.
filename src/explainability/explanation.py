"""Explainable decisions.

Everything here is generated from the decision record (the actual agent outputs and coordinator rationale);
nothing is hard-coded per action, so an explanation cannot describe numbers or votes the record does not contain.

build_explanation(record) -> dict with the sections shown in the dashboard:
    headline, inputs, votes, agent_reasoning, conflicts, decision_path, not_followed,
    cost_impact, security, human_approval, text
explain_text(record) -> the same content as one readable paragraph (stored in record["explanation"]).
check_explanation(record, text) -> list of contradictions (empty = consistent).
"""
from src.orchestration.negotiation import ALL_AGENTS, LABELS


def _final_cost(cost, final_action):
    """Cost of the final action, taken from the Cost Agent output (None if unavailable)."""
    m = cost.get("metrics", {}) if cost else {}
    if not cost or not {"cost_scale_up", "cost_scale_down", "budget"} <= set(m) or "current_cost" not in cost:
        return None
    proposed = {"SCALE_UP": m["cost_scale_up"], "SCALE_DOWN": m["cost_scale_down"],
                "MAINTAIN": cost["current_cost"]}[final_action]
    return {"pricing_source": cost.get("pricing_source", "UNKNOWN"), "current_cost": cost["current_cost"],
            "proposed_cost": proposed, "difference": proposed - cost["current_cost"], "budget": m["budget"],
            "budget_status": "WITHIN_BUDGET" if proposed <= m["budget"] else "OVER_BUDGET"}


def _security_section(sec):
    if not sec:
        return None
    checks = sec.get("policy_checks", [])
    failed = [c for c in checks if not c["passed"]]
    return {"verdict": sec.get("verdict", sec["status"]), "allowed_actions": sec["allowed_actions"],
            "blocked_actions": sec.get("blocked_actions", []), "alerts": sec.get("alerts", []),
            "checks_total": len(checks), "checks_passed": len(checks) - len(failed),
            "checks": checks, "failed": [c["check"] for c in failed]}


def build_explanation(record):
    r = record
    final, mid, strategy = r["final_action"], r["machine_id"], r["coordinator_strategy"]
    agents = r["agents"]
    rationale = r["rationale"]
    invalid = r["decision_status"] == "INVALID_INPUT"

    votes = {k: agents[k]["vote"] for k in ALL_AGENTS if agents.get(k)}
    agent_reasoning = []
    for k in ALL_AGENTS:
        a = agents.get(k)
        if a:
            agent_reasoning.append({"agent": LABELS[k], "vote": a["vote"], "status": a["status"],
                                    "risk_level": a["risk_level"], "reason": a["reason"],
                                    "followed": (final in a["allowed_actions"]) if k == "security" else a["vote"] == final})
        else:
            err = next((e["error"] for e in r["agent_errors"] if e["agent"] == k), "no output")
            agent_reasoning.append({"agent": LABELS[k], "vote": None, "status": "UNAVAILABLE", "risk_level": None,
                                    "reason": err if not invalid else "not run (input rejected)", "followed": None})

    cost_impact = _final_cost(agents.get("cost"), final)
    security = _security_section(agents.get("security"))
    conflicts = [c["detail"] for c in r["conflicts"]]
    not_followed = [f"{LABELS.get(x['agent'], x['agent'])} ({x['vote']}): {x['reason']}" for x in r["rejected"]]

    if invalid:
        headline = f"{final} (safe default) for {mid}: the input could not be used."
        inputs = "Input rejected."
    else:
        headline = f"{final} is recommended for {mid} under the {strategy} strategy."
        inputs = (f"CPU {r['current_cpu']:.1f}% now, {r['predicted_cpu']:.1f}% predicted; "
                  f"memory {r['current_memory']:.1f}% now, {r['predicted_memory']:.1f}% predicted.")

    approval = ("This is a recommendation only. Human approval is required before any change"
                + (", and a reviewer should look at it first." if r["requires_review"] else "."))

    parts = [headline, inputs]
    if not invalid:
        parts.append("Agent assessments: " + " ".join(
            f"{x['agent']}: {x['vote'] or 'unavailable'}" + (f" ({x['status']}, {x['risk_level']} risk)" if x["vote"] else "") + "."
            for x in agent_reasoning))
        parts.append("Conflicts: " + ("; ".join(conflicts) + "." if conflicts else "none between agents."))
    parts.append("How the decision was reached: " + " ".join(rationale))
    if not_followed:
        parts.append("Recommendations not followed: " + "; ".join(not_followed) + ".")
    if cost_impact:
        parts.append(f"Cost impact ({cost_impact['pricing_source']} pricing): {cost_impact['current_cost']:.2f}/h now, "
                     f"{cost_impact['proposed_cost']:.2f}/h after {final} ({cost_impact['difference']:+.2f}/h), budget "
                     f"{cost_impact['budget']:.2f}/h, {cost_impact['budget_status']}.")
    elif not invalid:
        parts.append("Cost impact: not available (Cost Agent unavailable or incomplete).")
    if security:
        s = security
        parts.append(f"Security: verdict {s['verdict']}; {s['checks_passed']} of {s['checks_total']} policy checks passed"
                     + (f" (failed: {', '.join(s['failed'])})" if s["failed"] else "")
                     + f"; allowed actions {s['allowed_actions']}.")
    elif not invalid:
        parts.append("Security: not available (Security Agent unavailable), so only MAINTAIN is considered safe.")
    parts.append(approval)

    return {"headline": headline, "inputs": inputs, "votes": votes, "agent_reasoning": agent_reasoning,
            "conflicts": conflicts, "decision_path": rationale, "not_followed": not_followed,
            "cost_impact": cost_impact, "security": security, "human_approval": approval, "text": " ".join(parts)}


def explain_text(record):
    return build_explanation(record)["text"]


def check_explanation(record, text=None):
    """Return a list of problems where the explanation contradicts the record. Empty list = consistent."""
    text = text if text is not None else record["explanation"]
    problems = []
    final, agents = record["final_action"], record["agents"]
    if not text.startswith(final) and f"{final} " not in text[:80]:
        problems.append("explanation does not state the final action first")
    if str(record["machine_id"]) not in text:
        problems.append("machine id missing")
    for k in ALL_AGENTS:
        a = agents.get(k)
        if a and a["vote"] not in text:
            problems.append(f"{LABELS[k]} vote {a['vote']} not shown")
        if not a and LABELS[k] not in text and record["decision_status"] != "INVALID_INPUT":
            problems.append(f"unavailable {LABELS[k]} not mentioned")
    cost = agents.get("cost")
    c = _final_cost(cost, final) if cost else None
    if c:
        if f"{c['proposed_cost']:.2f}" not in text or c["budget_status"] not in text:
            problems.append("cost impact numbers or budget status missing or wrong")
        other = "OVER_BUDGET" if c["budget_status"] == "WITHIN_BUDGET" else "WITHIN_BUDGET"
        if f"{c['budget']:.2f}/h, {other}" in text:
            problems.append("budget status contradicts the cost calculation")
    sec = agents.get("security")
    if sec:
        pc = sec.get("policy_checks", [])
        failed = [c for c in pc if not c["passed"]]
        if f"{len(pc) - len(failed)} of {len(pc)} policy checks passed" not in text:
            problems.append("security check count wrong")
        if final not in sec["allowed_actions"]:
            problems.append("final action is not allowed by the Security Agent")
    if "human approval" not in text.lower():
        problems.append("human-approval statement missing")
    return problems
