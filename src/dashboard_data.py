"""Data access for the dashboard (and anything else that reads saved results).

Every loader takes `root` (default: the project folder) and raises ResultsNotFound with a plain-language fix if a
file has not been produced yet, so the dashboard can show a helpful message instead of crashing.
"""
import json
from pathlib import Path

import pandas as pd

from src.orchestration.decision_log import read_log
from src.utils.config import ROOT

STRATEGIES = ("original", "improved", "cost_aware")


class ResultsNotFound(FileNotFoundError):
    pass


def _path(root, *parts):
    return (Path(root) if root else ROOT).joinpath(*parts)


def _require(path, fix):
    if not path.exists():
        raise ResultsNotFound(f"{path.name} not found. {fix}")
    return path


RUN_HINT = "Run `python -m src.app run` first."


def load_forecast_metrics(root=None):
    return pd.read_csv(_require(_path(root, "results", "metrics", "forecasting_comparison.csv"), RUN_HINT))


def load_per_machine_metrics(root=None):
    return pd.read_csv(_require(_path(root, "results", "metrics", "per_machine_forecasting.csv"), RUN_HINT))


def load_test_predictions(root=None):
    return pd.read_csv(_require(_path(root, "results", "metrics", "test_predictions.csv"), RUN_HINT))


def load_decisions(strategy, root=None):
    """List of full decision records for one strategy (latest run in the saved JSONL)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    p = _require(_path(root, "results", "decisions", f"decisions_{strategy}.jsonl"), RUN_HINT)
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_decision_table(strategy, root=None):
    return pd.read_csv(_require(_path(root, "results", "decisions", f"decisions_{strategy}.csv"), RUN_HINT))


def load_strategy_comparison(root=None):
    return pd.read_csv(_require(_path(root, "results", "decisions", "strategy_comparison.csv"), RUN_HINT))


def load_decision_log(root=None, run_id=None):
    return read_log(_path(root, "results", "decisions", "decision_log.jsonl"), run_id)


def load_eval_table(name, root=None):
    """name in: decision_distribution, strategy_agreement, agent_votes, threshold_sensitivity, budget_sensitivity"""
    return pd.read_csv(_require(_path(root, "results", "metrics", f"{name}.csv"), RUN_HINT))


def load_run_summary(root=None):
    p = _require(_path(root, "results", "metrics", "run_summary.json"), RUN_HINT)
    return json.loads(p.read_text(encoding="utf-8"))


def load_usage(root=None, machines=None):
    """Usage history of the selected machines (large: ~1.3M rows). Optionally restrict to some machines."""
    p = _require(_path(root, "data", "processed", "selected_machine_usage.csv"),
                 "Run `python -m src.app run` (data stage) first.")
    df = pd.read_csv(p, usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"])
    return df if machines is None else df[df["machine_id"].isin(machines)]


def list_graphs(root=None):
    d = _path(root, "results", "graphs")
    return sorted(d.glob("*.png")) if d.exists() else []
