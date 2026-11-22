"""Coordinator strategies. A coordinator turns the four agent outputs into ONE final action.

Interface:  coordinator.decide(outputs, obs) -> dict with
    final_action   SCALE_UP | SCALE_DOWN | MAINTAIN
    rationale      list of plain-text sentences built from the actual agent outputs (the audit trail)
    rejected       list of {agent, vote, reason} for recommendations that were not followed
    requires_review bool
    degraded       True if an agent was unavailable

Strategies: Original (notebook rules), Improved (conflict-aware, priority based), Cost-Aware (Improved + budget).
"""
from src.agents.base import SCALE_UP, SCALE_DOWN, MAINTAIN, load_agent_config
from src.orchestration.negotiation import (collect_votes, allowed_actions, unavailable_agents, LABELS)


class OriginalCoordinator:
    """Ported from the original notebook's coordinator (agents' votes combined by simple rules).

    Rules, in order:
      1. Any SCALE_UP vote from the Resource or Performance Agent -> propose SCALE_UP.
      2. Else, if every responding Resource/Performance vote is SCALE_DOWN -> propose SCALE_DOWN.
      3. Else MAINTAIN.
      4. The Security Agent's allowed actions are then enforced: a proposal that is not allowed becomes MAINTAIN.
    The Cost Agent is NOT used by this strategy (that is why the Cost-Aware strategy exists).
    """
    name = "original"

    def decide(self, outputs, obs):
        rationale, rejected = [], []
        votes = collect_votes(outputs)
        missing = unavailable_agents(outputs)
        core = {k: v for k, v in votes.items() if k in ("resource", "performance")}

        if core and SCALE_UP in core.values():
            proposed = SCALE_UP
            who = [LABELS[a] for a, v in core.items() if v == SCALE_UP]
            rationale.append(f"{' and '.join(who)} voted SCALE_UP, and the Original strategy follows any scale-up vote.")
        elif core and all(v == SCALE_DOWN for v in core.values()):
            proposed = SCALE_DOWN
            rationale.append(f"{' and '.join(LABELS[a] for a in core)} voted SCALE_DOWN.")
        else:
            proposed = MAINTAIN
            rationale.append("No scale-up vote and no unanimous scale-down vote from the Resource and Performance Agents.")

        final = proposed
        allowed = allowed_actions(outputs)
        if proposed not in allowed:
            sec = outputs.get("security")
            why = (f"Security Agent allows only {allowed} (alerts: {', '.join(sec['alerts']) or 'none'})"
                   if sec else "Security Agent unavailable, so policy cannot be verified")
            rejected.append({"agent": "security" if sec else "orchestrator", "vote": proposed, "reason": why})
            rationale.append(f"{proposed} was rejected: {why}. Final action is MAINTAIN.")
            final = MAINTAIN
        for agent, vote in votes.items():
            if vote != final and not any(r["agent"] == agent for r in rejected):
                rejected.append({"agent": agent, "vote": vote,
                                 "reason": "Cost Agent is not used by the Original strategy" if agent == "cost"
                                 else f"outweighed by the strategy's rules (final action {final})"})
        if missing:
            rationale.append(f"Unavailable agents: {', '.join(LABELS[m] for m in missing)}.")

        review = bool(outputs.get("security") and outputs["security"]["requires_review"]) or "security" in missing
        return {"final_action": final, "rationale": rationale, "rejected": rejected,
                "requires_review": review, "degraded": bool(missing)}


class ImprovedCoordinator:
    """Conflict-aware coordinator. Priority order: 1 Security constraints, 2 Performance/Resource protection, 3 cost.

    SCALE_UP is chosen when
      a) the Performance Agent votes SCALE_UP (high risk, even with the forecast error margin), or
      b) the Resource Agent votes SCALE_UP and the Performance Agent confirms at least MEDIUM risk, or
      c) the Resource Agent votes SCALE_UP and the Performance Agent is unavailable (protect performance, flagged).
    If the Resource Agent sees pressure but the Performance Agent sees LOW risk, that conflict is NOT resolved
    by scaling either way: the result is MAINTAIN (scaling down under pressure is unsafe; scaling up on one
    uncorroborated signal wastes money).
    SCALE_DOWN is chosen only when the Performance Agent says it is safe, at least one other agent (Resource or
    Cost) supports it, no agent shows pressure, and every agent responded.
    Finally the Security Agent's allowed actions are enforced (a forbidden action becomes MAINTAIN).
    """
    name = "improved"

    def decide(self, outputs, obs):
        res, perf, cost, sec = (outputs.get(k) for k in ("resource", "performance", "cost", "security"))
        votes = collect_votes(outputs)
        missing = unavailable_agents(outputs)
        rationale, rejected = [], []
        res_vote, perf_vote, cost_vote = votes.get("resource"), votes.get("performance"), votes.get("cost")

        if res_vote == SCALE_UP and perf and perf["risk_level"] == "LOW" and perf_vote != SCALE_UP:
            proposed = MAINTAIN
            rationale.append(
                f"Conflict: the Resource Agent reports {res['status']} ({res['reason']}) but the Performance Agent sees "
                f"LOW risk (upper-bound CPU {perf['metrics']['upper_cpu']:.1f}%, memory {perf['metrics']['upper_memory']:.1f}%). "
                f"The scale-up signal is not corroborated and scaling down under pressure is unsafe, so capacity is kept.")
        elif perf_vote == SCALE_UP:
            proposed = SCALE_UP
            rationale.append(f"The Performance Agent sees {perf['risk_level']} risk: {perf['reason']}")
        elif res_vote == SCALE_UP and perf:
            proposed = SCALE_UP
            rationale.append(f"The Resource Agent reports {res['status']} and the Performance Agent confirms "
                             f"{perf['risk_level']} risk (CPU {perf['cpu_pressure']}, memory {perf['memory_pressure']}).")
        elif res_vote == SCALE_UP:
            proposed = SCALE_UP
            rationale.append(f"The Resource Agent reports {res['status']}; the Performance Agent is unavailable, "
                             f"so the scale-up is accepted to protect performance.")
        elif (perf_vote == SCALE_DOWN and not missing and SCALE_UP not in votes.values()
              and (res_vote == SCALE_DOWN or cost_vote == SCALE_DOWN)):
            proposed = SCALE_DOWN
            support = [LABELS[a] for a, v in votes.items() if v == SCALE_DOWN and a != "performance"]
            rationale.append(f"The Performance Agent says capacity can be reduced safely ({perf['reason']}) "
                             f"and it is supported by the {' and '.join(support)}.")
        else:
            proposed = MAINTAIN
            if perf_vote == SCALE_DOWN:
                why = ("an agent is unavailable" if missing else
                       "no other agent supports it" if not (res_vote == SCALE_DOWN or cost_vote == SCALE_DOWN)
                       else "another agent reports pressure")
                rationale.append(f"The Performance Agent allows a scale-down but {why}, so capacity is kept.")
            else:
                rationale.append("No agent shows enough pressure or spare capacity to justify a change.")

        final = proposed
        allowed = allowed_actions(outputs)
        if proposed not in allowed:
            why = (f"Security Agent allows only {allowed} (alerts: {', '.join(sec['alerts']) or 'none'})"
                   if sec else "Security Agent unavailable, so policy cannot be verified")
            rejected.append({"agent": "security" if sec else "orchestrator", "vote": proposed, "reason": why})
            rationale.append(f"{proposed} was rejected: {why}. Final action is MAINTAIN.")
            final = MAINTAIN
        for agent, vote in votes.items():
            if vote != final and not any(r["agent"] == agent and r["vote"] == vote for r in rejected):
                rejected.append({"agent": agent, "vote": vote,
                                 "reason": _reject_reason(agent, vote, final, allowed, "conflict-aware")})
        if missing:
            rationale.append(f"Unavailable agents: {', '.join(LABELS[m] for m in missing)}.")
        review = bool(sec and sec["requires_review"]) or "security" in missing
        return {"final_action": final, "rationale": rationale, "rejected": rejected,
                "requires_review": review, "degraded": bool(missing)}


def _reject_reason(agent, vote, final, allowed, label):
    """Why an agent's vote was not followed, stated relative to the FINAL action of this strategy."""
    if agent == "cost" and vote == SCALE_DOWN:
        text = f"the saving did not outweigh the other agents' constraints (final action {final})"
    else:
        text = f"overruled by the {label} rules, which produced {final}"
    if vote not in allowed:
        text += f"; the Security Agent also does not allow {vote}"
    return text


class CostAwareCoordinator:
    """Improved decision + budget enforcement.

      * A SCALE_UP whose cost is over the budget becomes MAINTAIN, unless the forecast is critical
        (Performance Agent upper-bound CPU or memory above the critical limits in config).
      * If the budget cannot be checked (Cost Agent unavailable), no new spend is approved.
      * If current spend already exceeds the budget and a scale-down is safe (Security allows it, no pressure,
        Performance risk not HIGH), the action becomes SCALE_DOWN.
    """
    name = "cost_aware"

    def __init__(self, config=None):
        self.cfg = config or load_agent_config()["coordinators"]["cost_aware"]
        self.base = ImprovedCoordinator()

    def decide(self, outputs, obs):
        result = self.base.decide(outputs, obs)
        final, cost, perf = result["final_action"], outputs.get("cost"), outputs.get("performance")
        rationale = list(result["rationale"])
        # keep only the Security/orchestrator rejections of the base step; agent-vote rejections are rebuilt below
        rejected = [x for x in result["rejected"] if x["agent"] in ("security", "orchestrator")]
        review, degraded = result["requires_review"], result["degraded"]

        if final == SCALE_UP:
            if not cost:
                rationale.append("Cost-Aware: the Cost Agent is unavailable, so the budget cannot be checked; "
                                 "no new spend is approved. Final action is MAINTAIN.")
                rejected.append({"agent": "orchestrator", "vote": SCALE_UP, "reason": "the budget could not be verified"})
                final, review = MAINTAIN, True
            elif not cost["scale_up_within_budget"]:
                m = cost["metrics"]
                critical = bool(perf and (perf["metrics"]["upper_cpu"] >= self.cfg["critical_upper_cpu"]
                                          or perf["metrics"]["upper_memory"] >= self.cfg["critical_upper_memory"]))
                over = (f"SCALE_UP would cost {m['cost_scale_up']:.2f}/h, above the budget of {m['budget']:.2f}/h "
                        f"({cost['pricing_source']} pricing)")
                if critical:
                    rationale.append(f"Cost-Aware: {over}, but the forecast is critical (upper-bound CPU "
                                     f"{perf['metrics']['upper_cpu']:.1f}%, memory {perf['metrics']['upper_memory']:.1f}%), "
                                     f"so the budget is overridden and the action is flagged for review.")
                    review = True
                else:
                    rationale.append(f"Cost-Aware: {over}, and the forecast is not critical. "
                                     f"Final action is MAINTAIN.")
                    rejected.append({"agent": "budget", "vote": SCALE_UP,
                                     "reason": f"SCALE_UP costs {m['cost_scale_up']:.2f}/h, over the {m['budget']:.2f}/h budget, "
                                               f"and the forecast is not critical"})
                    final, review = MAINTAIN, True
            else:
                rationale.append(f"Cost-Aware: SCALE_UP costs {cost['metrics']['cost_scale_up']:.2f}/h, "
                                 f"within the budget of {cost['metrics']['budget']:.2f}/h.")
        elif final == MAINTAIN and cost and cost["status"] == "OVER_BUDGET":
            sec, res = outputs.get("security"), outputs.get("resource")
            safe = (sec is not None and SCALE_DOWN in allowed_actions(outputs) and perf is not None
                    and perf["risk_level"] != "HIGH" and not (res and res["vote"] == SCALE_UP))
            if safe:
                final = SCALE_DOWN
                rationale.append(f"Cost-Aware: current spend {cost['current_cost']:.2f}/h exceeds the budget "
                                 f"{cost['metrics']['budget']:.2f}/h and a scale-down is allowed and safe.")
        allowed = allowed_actions(outputs)
        for agent, vote in collect_votes(outputs).items():
            if vote != final and not any(r["agent"] == agent and r["vote"] == vote for r in rejected):
                rejected.append({"agent": agent, "vote": vote,
                                 "reason": _reject_reason(agent, vote, final, allowed, "cost-aware")})
        return {"final_action": final, "rationale": rationale, "rejected": rejected,
                "requires_review": review, "degraded": degraded}


COORDINATORS = {"original": OriginalCoordinator, "improved": ImprovedCoordinator,
                "cost_aware": CostAwareCoordinator}


def get_coordinator(name):
    try:
        return COORDINATORS[name]()
    except KeyError:
        raise ValueError(f"unknown coordinator {name!r}; available: {sorted(COORDINATORS)}")
