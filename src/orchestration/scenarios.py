"""The six required test scenarios, with expected outcomes worked out by hand from the agent rules
(default config/agents.yaml: CPU high 80, memory high 95, step 25%, simulated budget 6.0/h, scale-up costs 6.40/h).

Each scenario: an observation plus the expected final action for each coordinator and why.
"""

SCENARIOS = [
    {
        "name": "high_resource_demand",
        "obs": dict(machine_id="s1", current_cpu=85, predicted_cpu=88, current_memory=91, predicted_memory=92),
        "expected": {"original": "SCALE_UP", "improved": "SCALE_UP", "cost_aware": "SCALE_UP"},
        "why": "CPU is above 80% everywhere. SCALE_UP exceeds the simulated budget, but the forecast upper bound "
               "(88 + 4.32 = 92.32%) is critical, so the Cost-Aware strategy overrides the budget.",
    },
    {
        "name": "low_resource_demand",
        "obs": dict(machine_id="s2", current_cpu=12, predicted_cpu=14, current_memory=88, predicted_memory=88),
        "expected": {"original": "SCALE_DOWN", "improved": "SCALE_DOWN", "cost_aware": "SCALE_DOWN"},
        "why": "Resource, Performance and Cost all vote SCALE_DOWN and Security allows it.",
    },
    {
        "name": "high_memory_pressure",
        "obs": dict(machine_id="s3", current_cpu=30, predicted_cpu=30, current_memory=96, predicted_memory=97),
        "expected": {"original": "SCALE_UP", "improved": "SCALE_UP", "cost_aware": "MAINTAIN"},
        "why": "Resource sees memory pressure and Performance confirms MEDIUM risk (upper bound 97.64%), so Original "
               "and Improved scale up. The forecast is not critical (< 98%) and SCALE_UP is over budget, so Cost-Aware keeps capacity.",
    },
    {
        "name": "cost_constraint_violation",
        "obs": dict(machine_id="s4", current_cpu=80, predicted_cpu=84, current_memory=91, predicted_memory=91),
        "expected": {"original": "SCALE_UP", "improved": "SCALE_UP", "cost_aware": "MAINTAIN"},
        "why": "CPU pressure is real, but SCALE_UP (6.40/h) exceeds the 6.0/h budget and the upper bound "
               "(84 + 4.32 = 88.32%) is below the critical 90%, so only Cost-Aware refuses.",
    },
    {
        "name": "security_policy_violation",
        "obs": dict(machine_id="s5", current_cpu=85, predicted_cpu=88, current_memory=91, predicted_memory=92,
                    status="IMPORT_INSTALLING"),
        "expected": {"original": "MAINTAIN", "improved": "MAINTAIN", "cost_aware": "MAINTAIN"},
        "why": "The machine is not in an allowed status, so the Security Agent permits only MAINTAIN for every strategy.",
    },
    {
        "name": "conflicting_agents",
        "obs": dict(machine_id="s6", current_cpu=14, predicted_cpu=16, current_memory=95, predicted_memory=95),
        "expected": {"original": "SCALE_UP", "improved": "MAINTAIN", "cost_aware": "MAINTAIN"},
        "why": "Resource sees memory pressure (95%) while Performance and Cost see room to scale down (CPU 16%). "
               "Original follows the scale-up vote; Improved refuses to act on an uncorroborated signal and Security "
               "blocks the scale-down (memory too high), so the result is MAINTAIN.",
    },
]
