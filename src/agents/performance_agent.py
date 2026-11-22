"""Performance Agent: what could go wrong for application performance?

Different logic from the Resource Agent: it looks at the UPPER BOUND of the forecast
(prediction + forecast error margin, from the Random Forest RMSE) and checks what would
happen to utilisation if capacity were removed or added.

  upper_cpu    = predicted_cpu    + cpu_margin       upper_memory = predicted_memory + memory_margin
  risk         = HIGH / MEDIUM / LOW from the upper bounds
  SCALE_UP     if overall risk is HIGH
  SCALE_DOWN   only if risk is LOW AND projected CPU after removing capacity (upper_cpu / (1 - step))
               stays below max_post_scale_down_cpu
  MAINTAIN     otherwise
"""
from src.agents.base import (SCALE_UP, SCALE_DOWN, MAINTAIN, load_agent_config, validate_observation, risk_max)


class PerformanceAgent:
    name = "Performance Agent"

    def __init__(self, config=None, step_fraction=None):
        full = load_agent_config()
        self.cfg = config or full["performance_agent"]
        self.step = step_fraction if step_fraction is not None else full["action_step_fraction"]

    def evaluate(self, obs):
        o = validate_observation(obs)
        c = self.cfg
        upper_cpu = min(100.0, o["predicted_cpu"] + c["cpu_margin"])
        upper_mem = min(100.0, o["predicted_memory"] + c["memory_margin"])

        cpu_risk = "HIGH" if upper_cpu >= c["cpu_high_risk"] else "MEDIUM" if upper_cpu >= c["cpu_medium_risk"] else "LOW"
        mem_risk = "HIGH" if upper_mem >= c["memory_high_risk"] else "MEDIUM" if upper_mem >= c["memory_medium_risk"] else "LOW"
        risk = risk_max(cpu_risk, mem_risk)

        projected_cpu_down = min(100.0, upper_cpu / (1 - self.step))
        projected_cpu_up = upper_cpu / (1 + self.step)

        if risk == "HIGH":
            status, vote = "PERFORMANCE_RISK", SCALE_UP
            parts = []
            if cpu_risk == "HIGH":
                parts.append(f"upper-bound CPU {upper_cpu:.1f}% >= {c['cpu_high_risk']}%")
            if mem_risk == "HIGH":
                parts.append(f"upper-bound memory {upper_mem:.1f}% >= {c['memory_high_risk']}%")
            reason = "; ".join(parts) + f". Adding {self.step:.0%} capacity would bring CPU to about {projected_cpu_up:.1f}%."
        elif risk == "LOW" and projected_cpu_down < c["max_post_scale_down_cpu"]:
            status, vote = "SAFE_TO_REDUCE", SCALE_DOWN
            reason = (f"Risk is low (upper-bound CPU {upper_cpu:.1f}%, memory {upper_mem:.1f}%). After removing "
                      f"{self.step:.0%} capacity CPU would be about {projected_cpu_down:.1f}% (< {c['max_post_scale_down_cpu']}%).")
        elif risk == "MEDIUM":
            status, vote = "WATCH", MAINTAIN
            reason = f"Medium risk (CPU {cpu_risk}, memory {mem_risk}); keep current capacity."
        else:
            status, vote = "STABLE", MAINTAIN
            reason = (f"Risk is low but removing {self.step:.0%} capacity would push CPU to about "
                      f"{projected_cpu_down:.1f}% (>= {c['max_post_scale_down_cpu']}%).")

        return {
            "agent": self.name, "machine_id": o["machine_id"], "status": status, "vote": vote,
            "recommended_action": vote, "risk_level": risk, "reason": reason,
            "cpu_pressure": cpu_risk, "memory_pressure": mem_risk,
            "metrics": {"upper_cpu": upper_cpu, "upper_memory": upper_mem,
                        "projected_cpu_after_scale_down": projected_cpu_down,
                        "projected_cpu_after_scale_up": projected_cpu_up},
        }
