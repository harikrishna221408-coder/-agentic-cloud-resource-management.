"""Main application: runs the whole system end to end.

    python -m src.app run                 data -> forecasting models -> agents -> 3 coordinators -> evaluation
    python -m src.app run --skip-data     reuse the processed dataset
    python -m src.app run --skip-train    reuse the saved models and test predictions too
    python -m src.app analyze m_103       one machine: forecast, four agents, three coordinators
    python -m src.app machines            list machines

Decision SUPPORT only: nothing here changes a real cloud resource.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.agents.base import load_agent_config
from src.data import pipeline
from src.data.loader import META_COLUMNS
from src.evaluation import decision_eval as ev
from src.explainability.explanation import check_explanation
from src.forecasting import train as train_mod
from src.orchestration.decision_log import append_log, new_run_id, verify_trace
from src.orchestration.orchestrator import (Orchestrator, comparison_table, observations_from_predictions,
                                            save_records)
from src.service import MachineStore, analyze_machine
from src.utils.config import ROOT, load_config
from src.utils.logging_setup import get_logger

log = get_logger("app")
STRATEGIES = ("original", "improved", "cost_aware")


def _meta(root, cfg):
    p = root / cfg["paths"]["raw_meta"]
    return pd.read_csv(p, header=None, names=META_COLUMNS) if p.exists() else None


def decide_all(root, cfg):
    """Run the three strategies on every machine's latest observation; save, log and verify the records."""
    pred = pd.read_csv(root / "results" / "metrics" / "test_predictions.csv")
    obs = observations_from_predictions(pred, _meta(root, cfg))
    orch = Orchestrator()
    by = orch.compare_strategies(obs, STRATEGIES)
    out = root / "results" / "decisions"
    run_id, problems = new_run_id(), []
    for s, recs in by.items():
        save_records(recs, out, f"decisions_{s}")
        for r in recs:
            issues = check_explanation(r) + verify_trace(r)
            if issues:
                problems.append((s, r["machine_id"], issues))
        # the log is append-only: every run adds its records under its own run_id
        append_log(recs, out / "decision_log.jsonl", run_id)
    table = comparison_table(by)
    cur = pd.DataFrame(obs).set_index("machine_id")[["current_cpu", "predicted_cpu", "current_memory", "predicted_memory"]].round(1)
    table.merge(cur, left_on="machine_id", right_index=True).to_csv(out / "strategy_comparison.csv", index=False)
    if problems:
        log.error("%d explanation/trace problems: %s", len(problems), problems[:3])
    return by, obs, run_id, problems


def evaluate_all(root, cfg, by, n_sample=2000):
    """Decision evaluation tables and graphs (distribution, agreement, votes, sensitivity)."""
    pred = pd.read_csv(root / "results" / "metrics" / "test_predictions.csv")
    metrics, graphs = root / "results" / "metrics", root / "results" / "graphs"
    metrics.mkdir(parents=True, exist_ok=True)
    graphs.mkdir(parents=True, exist_ok=True)

    dist = ev.decision_distribution(by)
    dist.to_csv(metrics / "decision_distribution.csv")
    ev.plot_distribution(dist, graphs / "decision_distribution.png")
    ev.strategy_agreement(by).to_csv(metrics / "strategy_agreement.csv", index=False)
    ev.agent_vote_summary(by["improved"]).to_csv(metrics / "agent_votes.csv", index=False)

    sample = ev.sample_observations(pred, _meta(root, cfg), n=n_sample)
    sens = ev.threshold_sensitivity(sample)
    sens.to_csv(metrics / "threshold_sensitivity.csv", index=False)
    ev.plot_sensitivity(sens, graphs / "threshold_sensitivity.png")
    ev.budget_sensitivity(sample).to_csv(metrics / "budget_sensitivity.csv", index=False)
    sample_by = Orchestrator(explain=False).compare_strategies(sample, STRATEGIES)
    ev.decision_distribution(sample_by).to_csv(metrics / "decision_distribution_sample.csv")
    return dist, len(sample)


def run_all(root=None, cfg=None, skip_data=False, skip_train=False, n_sample=2000):
    root = Path(root) if root else ROOT
    cfg = cfg or load_config()
    t0 = datetime.now(timezone.utc)
    if not skip_data:
        log.info("STAGE 1/4 data")
        pipeline.run(root, cfg)
    if not skip_train:
        log.info("STAGE 2/4 forecasting models")
        train_mod.train(root=root, cfg=cfg)
    log.info("STAGE 3/4 agents + coordinators")
    by, obs, run_id, problems = decide_all(root, cfg)
    log.info("STAGE 4/4 evaluation")
    dist, n_sampled = evaluate_all(root, cfg, by, n_sample)
    summary = {"run_id": run_id, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "machines": len(obs), "strategies": list(STRATEGIES), "decisions_per_strategy": len(obs),
               "explanation_or_trace_problems": len(problems), "sample_size_for_sensitivity": n_sampled,
               "action_counts": {s: dist[s].to_dict() for s in dist.columns},
               "pricing_source": load_agent_config()["cost_agent"]["pricing_source"],
               "seconds": round((datetime.now(timezone.utc) - t0).total_seconds(), 1)}
    (root / "results" / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("done in %.1fs; summary written", summary["seconds"])
    return summary


def main(argv=None, root=None):
    ap = argparse.ArgumentParser(description="Agentic cloud resource management (decision support).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--skip-data", action="store_true")
    r.add_argument("--skip-train", action="store_true")
    r.add_argument("--sample", type=int, default=2000, help="observations for the sensitivity analysis")
    a = sub.add_parser("analyze", help="analyse one machine")
    a.add_argument("machine")
    a.add_argument("--index", type=int, default=None, help="row index in the machine's history (default: latest)")
    sub.add_parser("machines", help="list machines")
    args = ap.parse_args(argv)
    try:
        return _dispatch(args, root)
    except FileNotFoundError as e:       # a missing input is a normal first-run situation: explain it, no traceback
        msg = str(e)
        if "Run" not in msg:
            msg += " (Run `python -m src.app run` first; run scripts/download_data.py before that if the raw data is missing.)"
        print(f"error: {msg}", file=sys.stderr)
        return 1


def _dispatch(args, root):
    if args.cmd == "run":
        print(json.dumps(run_all(root, skip_data=args.skip_data, skip_train=args.skip_train, n_sample=args.sample), indent=2))
        return 0
    store = MachineStore.from_project(root)
    if args.cmd == "machines":
        print("\n".join(store.machines()))
        return 0
    result = analyze_machine(store, args.machine, args.index)
    o = result["observation"]
    print(f"{o['machine_id']} at t={o['timestamp']}: CPU {o['current_cpu']:.1f}% -> {o['predicted_cpu']:.1f}% (predicted), "
          f"memory {o['current_memory']:.1f}% -> {o['predicted_memory']:.1f}% (predicted)")
    for s, rec in result["decisions"].items():
        print(f"\n[{s}] {rec['final_action']}\n{rec['explanation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
