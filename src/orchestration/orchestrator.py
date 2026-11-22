"""Central orchestrator: runs the four agents, validates their outputs, runs a coordinator, saves decision records.

Workflow per machine (spec section 6):
  1 validate the input observation          5 run the chosen coordinator (negotiation)
  2 call the four agents (failures caught)  6 build the explanation
  3 validate every agent output             7 return / save a decision record
  4 detect conflicts between agents
This is decision SUPPORT: every record has human_approval_required = True and nothing touches a real cloud.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.agents.base import ACTIONS, MAINTAIN, InvalidObservation, validate_observation
from src.agents.cost_agent import CostAgent
from src.agents.performance_agent import PerformanceAgent
from src.agents.resource_agent import ResourceAgent
from src.agents.security_agent import SecurityAgent
from src.explainability.explanation import build_explanation, explain_text
from src.orchestration.coordinators import get_coordinator
from src.orchestration.negotiation import ALL_AGENTS, LABELS, detect_conflicts
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

SECURITY_VOTES = ("APPROVE", "RESTRICT", "REVIEW", "BLOCK")


def validate_agent_output(key, out):
    """Return None if the output is well-formed, otherwise a short description of the problem."""
    if not isinstance(out, dict):
        return "output is not a dict"
    for field in ("agent", "machine_id", "status", "vote", "risk_level", "reason"):
        if field not in out:
            return f"missing field {field!r}"
    if key == "security":
        allowed = out.get("allowed_actions")
        if not isinstance(allowed, list) or not set(allowed) <= set(ACTIONS):
            return "allowed_actions missing or invalid"
        if out["vote"] not in SECURITY_VOTES:
            return f"invalid security vote {out['vote']!r}"
    elif out["vote"] not in ACTIONS:
        return f"invalid vote {out['vote']!r}"
    if out["risk_level"] not in ("LOW", "MEDIUM", "HIGH"):
        return f"invalid risk_level {out['risk_level']!r}"
    return None


class Orchestrator:
    def __init__(self, agents=None, explainer=None, explain=True):
        self.agents = agents or {"resource": ResourceAgent(), "performance": PerformanceAgent(),
                                 "cost": CostAgent(), "security": SecurityAgent()}
        self.explainer = explainer or explain_text
        self.explain = explain   # False only for bulk evaluation runs (sensitivity analysis); never for real decisions

    def _explain(self, rec):
        """Add the explanation. A failure here must never lose the decision, so fall back to the plain rationale."""
        if not self.explain:
            rec["explanation"], rec["explanation_detail"] = "", None
            return
        try:
            rec["explanation"] = self.explainer(rec)
            rec["explanation_detail"] = build_explanation(rec)
        except Exception as e:
            log.error("explanation failed for %s: %s", rec.get("machine_id"), e)
            rec["explanation"] = f"{rec['final_action']}: " + " ".join(rec["rationale"]) + " (full explanation unavailable)"
            rec["explanation_detail"] = None

    # ----- agents -----
    def run_agents(self, obs):
        """Call each agent separately. Returns (outputs, errors); a failed agent gives None plus an error entry."""
        outputs, errors = {}, []
        for key in ALL_AGENTS:
            agent = self.agents.get(key)
            if agent is None:
                outputs[key] = None
                errors.append({"agent": key, "error": "agent not configured"})
                continue
            try:
                out = agent.evaluate(obs)
            except Exception as e:  # any agent failure is isolated and recorded
                log.warning("%s failed for %s: %s", key, obs.get("machine_id"), e)
                outputs[key] = None
                errors.append({"agent": key, "error": f"{type(e).__name__}: {e}"})
                continue
            problem = validate_agent_output(key, out)
            if problem:
                log.warning("%s gave invalid output for %s: %s", key, obs.get("machine_id"), problem)
                outputs[key] = None
                errors.append({"agent": key, "error": f"invalid output: {problem}"})
            else:
                outputs[key] = out
        return outputs, errors

    # ----- one machine -----
    def decide(self, obs, strategy="original"):
        coordinator = get_coordinator(strategy)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        base = {"machine_id": obs.get("machine_id") if isinstance(obs, dict) else None,
                "coordinator_strategy": strategy, "timestamp": now, "input": obs,
                "human_approval_required": True, "approval_status": "PENDING"}
        try:
            clean = validate_observation(obs)
        except InvalidObservation as e:
            rec = {**base, "decision_status": "INVALID_INPUT", "final_action": MAINTAIN,
                   "current_cpu": None, "predicted_cpu": None, "current_memory": None, "predicted_memory": None,
                   "agents": {k: None for k in ALL_AGENTS}, "agent_errors": [], "conflicts": [],
                   "rejected": [], "rationale": [f"Input rejected: {e}. MAINTAIN is the safe default; no change proposed."],
                   "requires_review": True, "degraded": True}
            self._explain(rec)
            return rec

        outputs, errors = self.run_agents(clean)
        conflicts = detect_conflicts(outputs)
        result = coordinator.decide(outputs, clean)
        rec = {**base, "input": clean,
               "decision_status": "DEGRADED" if result["degraded"] else "OK",
               "current_cpu": clean["current_cpu"], "predicted_cpu": clean["predicted_cpu"],
               "current_memory": clean["current_memory"], "predicted_memory": clean["predicted_memory"],
               "agents": outputs, "agent_errors": errors, "conflicts": conflicts,
               "final_action": result["final_action"], "rationale": result["rationale"],
               "rejected": result["rejected"], "requires_review": result["requires_review"],
               "degraded": result["degraded"]}
        rec["decision_id"] = f"{strategy}:{clean['machine_id']}:{clean.get('timestamp', 'na')}"
        self._explain(rec)
        return rec

    # ----- many machines -----
    def decide_batch(self, observations, strategy="original"):
        """One decision per machine. A repeated machine_id is skipped (and reported), never decided twice."""
        records, skipped, seen = [], [], set()
        for obs in observations:
            mid = obs.get("machine_id") if isinstance(obs, dict) else None
            if mid is not None and mid in seen:
                skipped.append({"machine_id": mid, "reason": "duplicate machine_id in batch"})
                continue
            seen.add(mid)
            records.append(self.decide(obs, strategy))
        if skipped:
            log.warning("skipped %d duplicate machine(s)", len(skipped))
        return records, skipped

    def compare_strategies(self, observations, strategies):
        """Run several coordinators on the same observations. Returns {strategy: records}."""
        return {s: self.decide_batch(observations, s)[0] for s in strategies}


# ----- persistence and helpers -----
def save_records(records, out_dir, name):
    """Write full records as JSON lines and a flat CSV summary. Returns the two paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / f"{name}.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    csv = out_dir / f"{name}.csv"
    flat_rows = [flatten(r) for r in records]
    pd.DataFrame(flat_rows).to_csv(csv, index=False)
    return jl, csv


def flatten(r):
    a = r["agents"]

    def vote(k):
        return a[k]["vote"] if a.get(k) else "UNAVAILABLE"
    return {"machine_id": r["machine_id"], "coordinator": r["coordinator_strategy"], "final_action": r["final_action"],
            "current_cpu": r["current_cpu"], "predicted_cpu": r["predicted_cpu"],
            "current_memory": r["current_memory"], "predicted_memory": r["predicted_memory"],
            "resource_vote": vote("resource"), "performance_vote": vote("performance"),
            "cost_vote": vote("cost"), "security_vote": vote("security"),
            "n_conflicts": len(r["conflicts"]), "requires_review": r["requires_review"],
            "decision_status": r["decision_status"], "explanation": r["explanation"], "timestamp": r["timestamp"]}


def comparison_table(by_strategy):
    """One row per machine, one column per strategy, plus whether the strategies agree."""
    frames = []
    for strategy, records in by_strategy.items():
        frames.append(pd.Series({r["machine_id"]: r["final_action"] for r in records}, name=strategy))
    table = pd.concat(frames, axis=1)
    table["all_agree"] = table.nunique(axis=1) == 1
    table.index.name = "machine_id"
    return table.reset_index()


def observations_from_predictions(pred_df, meta_df=None, latest_only=True):
    """Build agent observations from the Phase 3 prediction table (and optional machine metadata).

    One observation per machine (its latest test row). `source` records which data row it came from, so every
    decision can be traced back to the original input.
    """
    df = pred_df.sort_values("time_stamp")
    if latest_only:
        df = df.groupby("machine_id").tail(1)
    meta = {}
    if meta_df is not None:
        last = meta_df.sort_values("time_stamp").groupby("machine_id").tail(1)
        meta = last.set_index("machine_id")[["cpu_num", "status"]].to_dict("index")
    obs = []
    for _, r in df.iterrows():
        m = meta.get(r["machine_id"], {})
        o = {"machine_id": r["machine_id"], "timestamp": int(r["time_stamp"]),
             "current_cpu": float(r["cpu_util_percent"]), "predicted_cpu": float(r["cpu_pred"]),
             "current_memory": float(r["mem_util_percent"]), "predicted_memory": float(r["mem_pred"]),
             "source": "results/metrics/test_predictions.csv"}
        if "cpu_num" in m:
            o["cpu_num"], o["status"] = int(m["cpu_num"]), m["status"]
        obs.append(o)
    return obs
