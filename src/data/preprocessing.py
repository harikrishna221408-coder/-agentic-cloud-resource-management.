"""Cleaning, feature engineering and chronological splitting.

Method follows the original notebook:
  features = CPU lag 1-3, memory lag 1-3, input_gap (seconds since previous observation)
  targets  = next recorded CPU / memory value
Changes vs. the notebook (documented): invalid/duplicate rows are removed explicitly, and the
notebook's duplicated 'input_gap' column is replaced by one input_gap and one target_gap.
"""
import pandas as pd
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

NOTEBOOK_FEATURES = ["cpu_lag_1", "cpu_lag_2", "cpu_lag_3", "mem_lag_1", "mem_lag_2", "mem_lag_3", "input_gap"]
# Adds the CURRENT observation, which the original notebook's features leave out (lag 1 is the one BEFORE it).
CURRENT_VALUE_FEATURES = NOTEBOOK_FEATURES + ["cpu_util_percent", "mem_util_percent"]
FEATURE_SETS = {"notebook": NOTEBOOK_FEATURES, "current_value": CURRENT_VALUE_FEATURES}
FEATURES = NOTEBOOK_FEATURES   # columns that must exist for a row to be usable
TARGETS = ["cpu_target", "mem_target"]


def clean(df):
    """Drop duplicates and invalid utilisation values (missing or outside 0-100); sort by time."""
    n0 = len(df)
    df = df.drop_duplicates(["machine_id", "time_stamp"])
    df = df.dropna(subset=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"])
    ok = df["cpu_util_percent"].between(0, 100) & df["mem_util_percent"].between(0, 100)
    df = df[ok]
    log.info("clean: %d -> %d rows", n0, len(df))
    return df.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True)


def build_features(df, max_gap=60):
    """Create lag features, gaps and next-step targets per machine, then filter by time gap."""
    df = df.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True)
    g = df.groupby("machine_id")
    df["input_gap"] = g["time_stamp"].diff()
    df["target_gap"] = -g["time_stamp"].diff(-1)
    for lag in (1, 2, 3):
        df[f"cpu_lag_{lag}"] = g["cpu_util_percent"].shift(lag)
        df[f"mem_lag_{lag}"] = g["mem_util_percent"].shift(lag)
    df["cpu_target"] = g["cpu_util_percent"].shift(-1)
    df["mem_target"] = g["mem_util_percent"].shift(-1)
    df = df.dropna(subset=FEATURES + TARGETS + ["target_gap"])
    df = df[(df["input_gap"] <= max_gap) & (df["target_gap"] <= max_gap)]
    return df.reset_index(drop=True)


def chronological_split(df, train_fraction=0.8):
    """Per machine: first 80% of time = train, last 20% = test (no future leakage)."""
    train, test = [], []
    for _, grp in df.sort_values("time_stamp").groupby("machine_id"):
        k = int(len(grp) * train_fraction)
        train.append(grp.iloc[:k])
        test.append(grp.iloc[k:])
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)
