"""Train the CPU and memory Random Forest models (both feature variants) and evaluate them.

Usage: python -m src.forecasting.train
Trains on the chronological first 80% of each machine's data and evaluates on the last 20%. Two variants are trained
with identical settings and differ only in their input features:
  notebook       7 features (lags 1-3 + input gap), as in the original notebook
  current_value  the same 7 + the current CPU and memory
Saves models (joblib, models/<variant>/), metrics (CSV/JSON) and graphs. The variant named in
config/config.yaml (forecast.active_variant) provides cpu_pred / mem_pred for the agents and dashboard.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data.preprocessing import FEATURE_SETS, chronological_split
from src.forecasting import evaluation
from src.forecasting.predict import active_variant
from src.utils.config import load_config, ROOT
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

# Same settings as the original notebook.
RF_PARAMS = dict(n_estimators=100, max_depth=15, min_samples_leaf=2, max_features=1.0,
                 n_jobs=-1, random_state=42)


def train(df=None, root=None, cfg=None):
    """Train both variants and write models, metrics and graphs under `root` (default: the project folder)."""
    root = Path(root) if root else ROOT
    cfg = cfg or load_config()
    active = active_variant(cfg)
    if df is None:
        df = pd.read_csv(root / cfg["paths"]["forecast_dataset"])
    train_df, test_df = chronological_split(df, cfg["data"]["train_fraction"])
    log.info("train=%d test=%d active variant=%s", len(train_df), len(test_df), active)

    models, predictions = {}, {}
    for variant, features in FEATURE_SETS.items():
        models[variant] = {}
        for name, target in [("cpu", "cpu_target"), ("memory", "mem_target")]:
            t0 = time.time()
            models[variant][name] = RandomForestRegressor(**RF_PARAMS).fit(train_df[features], train_df[target])
            log.info("%s / %s model trained in %.1fs (%d features)", variant, name, time.time() - t0, len(features))
        vdir = root / "models" / variant
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models[variant]["cpu"], vdir / "cpu_random_forest.joblib")
        joblib.dump(models[variant]["memory"], vdir / "memory_random_forest.joblib")
        predictions[variant] = (models[variant]["cpu"].predict(test_df[features]),
                                models[variant]["memory"].predict(test_df[features]))

    metrics_dir = root / "results" / "metrics"
    graph_dir = root / "results" / "graphs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    comp = evaluation.compare(test_df, predictions)
    comp.to_csv(metrics_dir / "forecasting_comparison.csv", index=False)
    evaluation.per_machine(test_df, predictions).to_csv(metrics_dir / "per_machine_forecasting.csv", index=False)

    # Test predictions: cpu_pred / mem_pred = the ACTIVE variant (what the agents use); every variant is also kept.
    out = test_df[["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
                   "cpu_target", "mem_target", "input_gap", "target_gap"]].copy()
    out["cpu_pred"], out["mem_pred"] = predictions[active]
    for v, (cpu, mem) in predictions.items():
        out[f"cpu_pred_{v}"], out[f"mem_pred_{v}"] = cpu, mem
    out["cpu_naive"], out["mem_naive"] = test_df["cpu_lag_1"].values, test_df["mem_lag_1"].values
    out.to_csv(metrics_dir / "test_predictions.csv", index=False)

    first = sorted(test_df["machine_id"].unique())[0]
    evaluation.plot_actual_vs_pred(test_df, predictions, first, graph_dir)

    info = {"train_rows": len(train_df), "test_rows": len(test_df), "active_variant": active,
            "features": {v: f for v, f in FEATURE_SETS.items()}, "rf_params": RF_PARAMS,
            "split": "chronological per machine"}
    (metrics_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    log.info("\n%s", comp.round(4).to_string(index=False))
    return models, comp


if __name__ == "__main__":
    train()
