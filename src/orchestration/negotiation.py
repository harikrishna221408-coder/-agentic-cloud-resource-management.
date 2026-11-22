"""Building blocks for negotiation: collecting votes, finding conflicts, applying security limits.

Agent outputs are passed around as a dict {agent_key: output_dict_or_None}; None means the agent was
unavailable (error, malformed output). Keys: resource, performance, cost, security.
"""
from src.agents.base import SCALE_UP, SCALE_DOWN, MAINTAIN, ACTIONS

ACTION_AGENTS = ("resource", "performance", "cost")   # agents that vote for an action
ALL_AGENTS = ACTION_AGENTS + ("security",)
LABELS = {"resource": "Resource Agent", "performance": "Performance Agent",
          "cost": "Cost Agent", "security": "Security Agent"}


def collect_votes(outputs):
    """{agent_key: action} for the action-voting agents that responded."""
    return {k: outputs[k]["vote"] for k in ACTION_AGENTS if outputs.get(k)}


def unavailable_agents(outputs):
    return [k for k in ALL_AGENTS if not outputs.get(k)]


def allowed_actions(outputs):
    """Actions the Security Agent permits. If it is unavailable, only MAINTAIN is safe."""
    sec = outputs.get("security")
    if not sec:
        return [MAINTAIN]
    return [a for a in ACTIONS if a in sec["allowed_actions"]]


def detect_conflicts(outputs):
    """List every disagreement between agents as a plain dict (type, agents, detail)."""
    conflicts = []
    votes = collect_votes(outputs)
    by_action = {}
    for agent, action in votes.items():
        by_action.setdefault(action, []).append(agent)
    if len(by_action) > 1:
        detail = "; ".join(f"{', '.join(LABELS[a] for a in ags)} -> {act}" for act, ags in by_action.items())
        conflicts.append({"type": "VOTE_DISAGREEMENT", "agents": sorted(votes), "detail": detail})
    allowed = allowed_actions(outputs)
    for agent, action in votes.items():
        if action not in allowed:
            conflicts.append({"type": "SECURITY_RESTRICTION", "agents": [agent, "security"],
                              "detail": f"{LABELS[agent]} voted {action} but Security Agent allows only {allowed}"})
    cost = outputs.get("cost")
    if cost and not cost.get("scale_up_within_budget", True) and SCALE_UP in votes.values():
        ups = [LABELS[a] for a, v in votes.items() if v == SCALE_UP]
        conflicts.append({"type": "BUDGET_CONFLICT", "agents": ["cost"],
                          "detail": f"{', '.join(ups)} voted SCALE_UP but a SCALE_UP exceeds the budget "
                                    f"({cost['metrics']['cost_scale_up']:.2f} > {cost['metrics']['budget']:.2f} per hour)"})
    return conflicts
