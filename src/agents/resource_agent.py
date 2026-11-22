"""Resource Agent: is the machine short of capacity, or holding capacity it does not need?

Logic (uses BOTH current and predicted values):
  pressure     if max(current, predicted) CPU >= cpu_high  OR  max(current, predicted) memory >= memory_high
  over-allocated if current AND predicted CPU < cpu_low  and memory is not under pressure
  otherwise    normal
"""
from src.agents.base import (SCALE_UP, SCALE_DOWN, MAINTAIN, load_agent_config, validate_observation)


class ResourceAgent:
    name = "Resource Agent"

    def __init__(self, config=None):
        self.cfg = config or load_agent_config()["resource_agent"]

    def evaluate(self, obs):
        o = validate_observation(obs)
        c = self.cfg
        cpu_now, cpu_pred = o["current_cpu"], o["predicted_cpu"]
        mem_now, mem_pred = o["current_memory"], o["predicted_memory"]
        cpu_demand, mem_demand = max(cpu_now, cpu_pred), max(mem_now, mem_pred)

        cpu_pressure = cpu_demand >= c["cpu_high"]
        mem_pressure = mem_demand >= c["memory_high"]
        cpu_idle = cpu_now < c["cpu_low"] and cpu_pred < c["cpu_low"]

        if cpu_pressure and mem_pressure:
            status, vote, risk = "BOTH_PRESSURE", SCALE_UP, "HIGH"
            reason = (f"CPU demand {cpu_demand:.1f}% >= {c['cpu_high']}% and memory demand "
                      f"{mem_demand:.1f}% >= {c['memory_high']}%.")
        elif cpu_pressure:
            status, vote, risk = "CPU_PRESSURE", SCALE_UP, "HIGH"
            reason = f"CPU demand {cpu_demand:.1f}% (current {cpu_now:.1f}%, predicted {cpu_pred:.1f}%) >= {c['cpu_high']}%."
        elif mem_pressure:
            status, vote, risk = "MEMORY_PRESSURE", SCALE_UP, "MEDIUM"
            reason = f"Memory demand {mem_demand:.1f}% >= {c['memory_high']}%."
        elif cpu_idle:
            status, vote, risk = "OVERALLOCATED", SCALE_DOWN, "LOW"
            reason = (f"CPU is low now ({cpu_now:.1f}%) and predicted ({cpu_pred:.1f}%), both < {c['cpu_low']}%, "
                      f"and memory ({mem_demand:.1f}%) is below the pressure limit {c['memory_high']}%.")
        else:
            status, vote, risk = "NORMAL", MAINTAIN, "LOW"
            reason = (f"CPU {cpu_demand:.1f}% and memory {mem_demand:.1f}% are within "
                      f"[{c['cpu_low']}%, {c['cpu_high']}%) / below {c['memory_high']}%.")

        return {
            "agent": self.name, "machine_id": o["machine_id"], "status": status, "vote": vote,
            "recommended_action": vote, "risk_level": risk, "reason": reason,
            "current_cpu": cpu_now, "predicted_cpu": cpu_pred,
            "current_memory": mem_now, "predicted_memory": mem_pred,
            "metrics": {"cpu_demand": cpu_demand, "memory_demand": mem_demand,
                        "cpu_headroom": 100 - cpu_demand, "memory_headroom": 100 - mem_demand},
        }
