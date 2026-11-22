"""Cost Agent: what does each possible action cost, and is spend being wasted?

All prices come from config/agents.yaml and are SIMULATED (pricing_source says so); no real cloud
price is used. Cost = vCPUs * vcpu_price + memory_GB * gb_price, per hour. SCALE_UP / SCALE_DOWN change the
allocation by action_step_fraction.

  SCALE_DOWN vote   if current spend exceeds the budget, or predicted CPU < low_utilisation_cpu and the hourly
                    saving >= min_saving_per_hour
  MAINTAIN vote     otherwise. The Cost Agent never asks for more spend; it reports whether a SCALE_UP fits the
                    budget (scale_up_within_budget) so the coordinators can respect it.
"""
from src.agents.base import (SCALE_UP, SCALE_DOWN, MAINTAIN, ACTIONS, load_agent_config, validate_observation)


class CostAgent:
    name = "Cost Agent"

    def __init__(self, config=None, step_fraction=None):
        full = load_agent_config()
        self.cfg = config or full["cost_agent"]
        self.step = step_fraction if step_fraction is not None else full["action_step_fraction"]

    def _allocation(self, o):
        vcpus = float(o.get("cpu_num") or self.cfg["default_vcpus"])
        return vcpus, float(self.cfg["simulated_memory_gb"])

    def _cost(self, vcpus, mem_gb):
        return vcpus * self.cfg["vcpu_hourly_price"] + mem_gb * self.cfg["memory_gb_hourly_price"]

    def cost_of_action(self, o, action):
        """Hourly cost if `action` were applied."""
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        vcpus, mem_gb = self._allocation(o)
        factor = {SCALE_UP: 1 + self.step, SCALE_DOWN: 1 - self.step, MAINTAIN: 1.0}[action]
        return self._cost(vcpus * factor, mem_gb * factor)

    def assess_action(self, obs, action):
        """Cost assessment of one specific proposed action (used by the coordinators)."""
        o = validate_observation(obs)
        current = self.cost_of_action(o, MAINTAIN)
        proposed = self.cost_of_action(o, action)
        budget = self.cfg["budget_per_machine_hour"]
        return {"action": action, "current_cost": current, "proposed_cost": proposed,
                "cost_difference": proposed - current, "budget": budget,
                "within_budget": proposed <= budget,
                "budget_status": "WITHIN_BUDGET" if proposed <= budget else "OVER_BUDGET"}

    def evaluate(self, obs):
        o = validate_observation(obs)
        c = self.cfg
        current = self.cost_of_action(o, MAINTAIN)
        up = self.assess_action(o, SCALE_UP)
        down = self.assess_action(o, SCALE_DOWN)
        saving = current - down["proposed_cost"]
        idle_cost = current * (1 - o["predicted_cpu"] / 100)

        if current > c["budget_per_machine_hour"]:
            status, vote, risk = "OVER_BUDGET", SCALE_DOWN, "MEDIUM"
            reason = f"Current cost {current:.2f}/h exceeds the budget {c['budget_per_machine_hour']:.2f}/h."
        elif o["predicted_cpu"] < c["low_utilisation_cpu"] and saving >= c["min_saving_per_hour"]:
            status, vote, risk = "WASTE_DETECTED", SCALE_DOWN, "LOW"
            reason = (f"Predicted CPU {o['predicted_cpu']:.1f}% < {c['low_utilisation_cpu']}%; scaling down would save "
                      f"{saving:.2f}/h (>= {c['min_saving_per_hour']}).")
        else:
            status, vote, risk = "COST_ACCEPTABLE", MAINTAIN, "LOW"
            reason = (f"No worthwhile saving (predicted CPU {o['predicted_cpu']:.1f}%, possible saving {saving:.2f}/h). "
                      f"A SCALE_UP would cost {up['proposed_cost']:.2f}/h ({up['budget_status']}).")

        return {
            "agent": self.name, "machine_id": o["machine_id"], "status": status, "vote": vote,
            "recommended_action": vote, "risk_level": risk, "reason": reason,
            "pricing_source": c["pricing_source"],
            "current_cost": current, "proposed_cost": self.cost_of_action(o, vote),
            "cost_difference": self.cost_of_action(o, vote) - current,
            "budget_status": "WITHIN_BUDGET" if self.cost_of_action(o, vote) <= c["budget_per_machine_hour"] else "OVER_BUDGET",
            "scale_up_within_budget": up["within_budget"],
            "metrics": {"cost_scale_up": up["proposed_cost"], "cost_scale_down": down["proposed_cost"],
                        "potential_saving_per_hour": saving, "estimated_idle_cost": idle_cost,
                        "budget": c["budget_per_machine_hour"]},
        }
