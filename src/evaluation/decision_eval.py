"""Evaluation of the DECISIONS (forecast evaluation lives in src/forecasting/evaluation.py).

All functions take decision records (from the orchestrator) or observations and return DataFrames, so the dashboard,
the CLI and the tests use the same code.
"""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.base import ACTIONS, load_agent_config
from src.agents.resource_agent import ResourceAgent
from src.agents.performance_agent import PerformanceAgent
from src.agents.cost_agent import CostAgent
from src.agents.security_agent import SecurityAgent
from src.orchestration.orchestrator import Orchestrator, observations_from_predictions
from src.utils.logging_setup import get_logger

log = get_logger(__name__)
STRATEGIES = ("original", "improved", "cost_aware")


def sample_observations(pred_df, meta_df=None, n=2000, seed=42):
    """A seeded random sample of observations spread over time (not just each machine's last row).

    Neighbouring rows of one machine are correlated, so this describes behaviour across the test period; it is not
    2000 independent samples.
    """
    obs = observations_from_predictions(pred_df, meta_df, latest_only=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(obs), size=min(n, len(obs)), replace=False)
    sampled = [obs[i] for i in sorted(idx)]
    # decide_batch keeps one decision per machine_id, so make ids unique per (machine, time)
    for o in sampled:
        o["machine_id"] = f"{o['machine_id']}@{o['timestamp']}"
    return sampled


def decision_distribution(by_strategy):
    """Counts and percentages of each action per strategy (rows = action)."""
    counts = pd.DataFrame({s: pd.Series([r["final_action"] for r in recs]).value_counts()
                           for s, recs in by_strategy.items()}).reindex(list(ACTIONS)).fillna(0).astype(int)
    return counts


def strategy_agreement(by_strategy):
    """Share of machines on which each pair of strategies chose the same action."""
    acts = {s: pd.Series({r["machine_id"]: r["final_action"] for r in recs}) for s, recs in by_strategy.items()}
    rows = []
    for a, b in itertools.combinations(acts, 2):
        rows.append({"strategy_a": a, "strategy_b": b, "agreement": float((acts[a] == acts[b]).mean()),
                     "n": len(acts[a])})
    return pd.DataFrame(rows)


def agent_vote_summary(records):
    """How often each agent voted for each action (Security shown by verdict)."""
    rows = []
    for k in ("resource", "performance", "cost"):
        votes = pd.Series([r["agents"][k]["vote"] for r in records if r["agents"].get(k)]).value_counts()
        rows.append({"agent": k, **{a: int(votes.get(a, 0)) for a in ACTIONS}})
    return pd.DataFrame(rows)


def _orchestrator_with(cfg_changes, explain=False):
    """Orchestrator whose agents use a modified copy of the default config. cfg_changes: {section: {key: value}}."""
    cfg = load_agent_config()
    for section, changes in cfg_changes.items():
        cfg[section].update(changes)
    step = cfg["action_step_fraction"]
    agents = {"resource": ResourceAgent(cfg["resource_agent"]),
              "performance": PerformanceAgent(cfg["performance_agent"], step),
              "cost": CostAgent(cfg["cost_agent"], step),
              "security": SecurityAgent(cfg["security_agent"], step)}
    return Orchestrator(agents=agents, explain=explain)


def threshold_sensitivity(observations, cpu_high_values=(70, 75, 80, 85, 90), memory_high_values=(93, 95, 97),
                          strategies=STRATEGIES):
    """How decisions change when the Resource Agent's pressure thresholds change.

    One row per (cpu_high, memory_high, strategy) with action counts and shares. The Performance, Cost and Security
    agents keep their default settings, so this isolates the effect of the Resource Agent's thresholds.
    """
    rows = []
    for cpu_high, mem_high in itertools.product(cpu_high_values, memory_high_values):
        orch = _orchestrator_with({"resource_agent": {"cpu_high": cpu_high, "memory_high": mem_high}})
        for s in strategies:
            recs, _ = orch.decide_batch(observations, s)
            c = pd.Series([r["final_action"] for r in recs]).value_counts()
            n = len(recs)
            rows.append({"cpu_high": cpu_high, "memory_high": mem_high, "strategy": s, "n": n,
                         **{a: int(c.get(a, 0)) for a in ACTIONS},
                         **{f"{a}_pct": round(100 * c.get(a, 0) / n, 2) for a in ACTIONS}})
        log.info("sensitivity cpu_high=%s memory_high=%s done", cpu_high, mem_high)
    return pd.DataFrame(rows)


def budget_sensitivity(observations, budgets=(5.5, 6.0, 6.4, 7.0), strategy="cost_aware"):
    """Effect of the (simulated) budget on the Cost-Aware strategy: a scale-up costs 6.40/h at the defaults."""
    rows = []
    for b in budgets:
        orch = _orchestrator_with({"cost_agent": {"budget_per_machine_hour": b}})
        recs, _ = orch.decide_batch(observations, strategy)
        c = pd.Series([r["final_action"] for r in recs]).value_counts()
        rows.append({"budget_per_hour": b, "n": len(recs), **{a: int(c.get(a, 0)) for a in ACTIONS}})
    return pd.DataFrame(rows)


# ---------- graphs ----------
def plot_distribution(dist, path, title="Final actions by coordinator strategy"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dist.T.plot(kind="bar", ax=ax, color={"SCALE_UP": "#d95f02", "SCALE_DOWN": "#1b9e77", "MAINTAIN": "#7570b3"})
    ax.set(ylabel="Number of decisions", xlabel="Coordinator strategy", title=title)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(sens, path, strategy="improved", memory_high=95):
    d = sens[(sens["strategy"] == strategy) & (sens["memory_high"] == memory_high)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for a, color in [("SCALE_UP", "#d95f02"), ("SCALE_DOWN", "#1b9e77"), ("MAINTAIN", "#7570b3")]:
        ax.plot(d["cpu_high"], d[f"{a}_pct"], marker="o", label=a, color=color)
    ax.set(xlabel="Resource Agent CPU-high threshold (%)", ylabel="Share of decisions (%)",
           title=f"Threshold sensitivity ({strategy} strategy, memory threshold {memory_high}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
