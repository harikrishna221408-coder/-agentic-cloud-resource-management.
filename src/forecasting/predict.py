"""Reusable prediction function for the agent system and dashboard."""
from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocessing import FEATURE_SETS
from src.utils.config import ROOT, load_config

_cache = {}


def active_variant(cfg=None):
    """The forecast variant the system uses (config/config.yaml: forecast.active_variant)."""
    variant = (cfg or load_config()).get("forecast", {}).get("active_variant", "current_value")
    if variant not in FEATURE_SETS:
        raise ValueError(f"unknown forecast variant {variant!r}; choose from {sorted(FEATURE_SETS)}")
    return variant


def load_models(model_dir=None, variant=None):
    """(cpu_model, memory_model) for a variant. Models live in <model_dir>/<variant>/."""
    variant = variant or active_variant()
    if variant not in FEATURE_SETS:
        raise ValueError(f"unknown forecast variant {variant!r}; choose from {sorted(FEATURE_SETS)}")
    d = Path(model_dir or ROOT / "models") / variant
    key = str(d)
    if key not in _cache:
        _cache[key] = (joblib.load(d / "cpu_random_forest.joblib"), joblib.load(d / "memory_random_forest.joblib"))
    return _cache[key]


def predict_next(cpu_history, mem_history, input_gap=10.0, model_dir=None, variant=None):
    """Predict the NEXT CPU and memory utilisation (%), i.e. the observation after the current one.

    cpu_history / mem_history: observations oldest first, and the LAST value is the CURRENT observation x(t).
    At least 4 values are needed: x(t-3), x(t-2), x(t-1), x(t). Lags are x(t-1), x(t-2), x(t-3).
    The 'current_value' variant also uses x(t) itself; the 'notebook' variant ignores it (as the original notebook did).
    input_gap = seconds between x(t-1) and x(t).
    Returns (predicted_cpu, predicted_memory), clipped to 0-100.
    """
    if len(cpu_history) < 4 or len(mem_history) < 4:
        raise ValueError("need at least 4 CPU and 4 memory observations (3 lags + the current value)")
    vals = list(cpu_history[-4:]) + list(mem_history[-4:]) + [input_gap]
    if any(pd.isna(v) for v in vals):
        raise ValueError("history contains missing values")
    variant = variant or active_variant()
    if variant not in FEATURE_SETS:
        raise ValueError(f"unknown forecast variant {variant!r}; choose from {sorted(FEATURE_SETS)}")
    c3, c2, c1, c0 = cpu_history[-4:]
    m3, m2, m1, m0 = mem_history[-4:]
    row = {"cpu_lag_1": c1, "cpu_lag_2": c2, "cpu_lag_3": c3, "mem_lag_1": m1, "mem_lag_2": m2, "mem_lag_3": m3,
           "input_gap": float(input_gap), "cpu_util_percent": c0, "mem_util_percent": m0}
    X = pd.DataFrame([row])[FEATURE_SETS[variant]]
    cpu_model, mem_model = load_models(model_dir, variant)
    cpu = float(cpu_model.predict(X)[0])
    mem = float(mem_model.predict(X)[0])
    return min(max(cpu, 0.0), 100.0), min(max(mem, 0.0), 100.0)
