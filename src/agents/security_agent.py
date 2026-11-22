"""Security Agent: resource-management POLICY checks (not cybersecurity threat detection).

Checks, each reported as pass/fail in `policy_checks`:
  machine_status      machine status must be in allowed_statuses       -> else only MAINTAIN, review required
  step_size           configured scaling step <= max_step_fraction      -> else changes need review
  scale_down_memory   predicted memory < scale_down_max_memory          -> else SCALE_DOWN blocked
  scale_down_cpu      projected CPU after scale-down < limit            -> else SCALE_DOWN blocked
  unusual_change      |predicted - current| CPU < spike_cpu_delta       -> else review (not an attack detector)
  protected_machine   machine not in protected_machines                 -> else review

Result: allowed_actions (what the coordinators may choose) and a verdict:
  VIOLATION       machine status is not allowed (only MAINTAIN is permitted)
  REVIEW_REQUIRED a review flag was raised (large step, unusual change, protected machine)
  RESTRICTED      a guard blocks SCALE_DOWN (e.g. memory too high); normal operation, not a violation
  APPROVED        all checks passed, no action restricted
"""
from src.agents.base import (SCALE_UP, SCALE_DOWN, MAINTAIN, load_agent_config, validate_observation)


class SecurityAgent:
    name = "Security Agent"

    def __init__(self, config=None, step_fraction=None):
        full = load_agent_config()
        self.cfg = config or full["security_agent"]
        self.step = step_fraction if step_fraction is not None else full["action_step_fraction"]

    def evaluate(self, obs):
        o = validate_observation(obs)
        c = self.cfg
        allowed = {SCALE_UP, SCALE_DOWN, MAINTAIN}
        checks, alerts = [], []
        review = False

        def check(name, passed, detail):
            checks.append({"check": name, "passed": bool(passed), "detail": detail})

        status = o.get("status", "USING")
        ok = status_ok = status in c["allowed_statuses"]
        check("machine_status", ok, f"status={status}, allowed={c['allowed_statuses']}")
        if not ok:
            allowed = {MAINTAIN}
            review = True
            alerts.append(f"MACHINE_STATUS_{status}")

        ok = self.step <= c["max_step_fraction"]
        check("step_size", ok, f"step={self.step:.0%}, max={c['max_step_fraction']:.0%}")
        if not ok:
            review = True
            alerts.append("STEP_TOO_LARGE")

        ok = o["predicted_memory"] < c["scale_down_max_memory"]
        check("scale_down_memory", ok, f"predicted memory {o['predicted_memory']:.1f}% vs limit {c['scale_down_max_memory']}%")
        if not ok:
            allowed.discard(SCALE_DOWN)
            alerts.append("SCALE_DOWN_BLOCKED_HIGH_MEMORY")

        projected = min(100.0, o["predicted_cpu"] / (1 - self.step))
        ok = projected < c["max_post_scale_down_cpu"]
        check("scale_down_cpu", ok, f"projected CPU after scale-down {projected:.1f}% vs limit {c['max_post_scale_down_cpu']}%")
        if not ok:
            allowed.discard(SCALE_DOWN)
            alerts.append("SCALE_DOWN_BLOCKED_HIGH_CPU")

        delta = abs(o["predicted_cpu"] - o["current_cpu"])
        ok = delta < c["spike_cpu_delta"]
        check("unusual_change", ok, f"|predicted - current| CPU = {delta:.1f} vs limit {c['spike_cpu_delta']}")
        if not ok:
            review = True
            alerts.append("UNUSUAL_CPU_CHANGE")

        ok = o["machine_id"] not in c["protected_machines"]
        check("protected_machine", ok, "machine is protected" if not ok else "not protected")
        if not ok:
            review = True
            alerts.append("PROTECTED_MACHINE")

        blocked = {SCALE_UP, SCALE_DOWN, MAINTAIN} - allowed
        if not status_ok:
            verdict, risk = "VIOLATION", "HIGH"
        elif review:
            verdict, risk = "REVIEW_REQUIRED", "MEDIUM"
        elif blocked:
            verdict, risk = "RESTRICTED", "LOW"
        else:
            verdict, risk = "APPROVED", "LOW"

        failed = [x["check"] for x in checks if not x["passed"]]
        reason = ("All policy checks passed." if not failed else
                  f"Failed checks: {', '.join(failed)}. Allowed actions: {sorted(allowed)}.")
        return {
            "agent": self.name, "machine_id": o["machine_id"], "status": verdict,
            "vote": {"APPROVED": "APPROVE", "RESTRICTED": "RESTRICT", "REVIEW_REQUIRED": "REVIEW", "VIOLATION": "BLOCK"}[verdict],
            "recommended_action": MAINTAIN if allowed == {MAINTAIN} else "NO_OBJECTION",
            "risk_level": risk, "reason": reason, "verdict": verdict,
            "allowed_actions": sorted(allowed), "blocked_actions": sorted(blocked),
            "requires_review": review, "alerts": alerts, "policy_checks": checks,
            "metrics": {"projected_cpu_after_scale_down": projected, "cpu_change": delta},
        }
