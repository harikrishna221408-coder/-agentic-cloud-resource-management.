"""Shared test helpers."""
import copy

import numpy as np
import pandas as pd

from src.utils.config import load_config


def make_project(tmp_path, machines=("m_1", "m_2", "m_3"), n=700):
    """A tiny project folder with a synthetic raw trace in the real file format (no header, 9 / 7 columns)."""
    rng = np.random.default_rng(1)
    rows, meta = [], []
    for k, m in enumerate(machines):
        cpu = np.clip(np.round(40 + 20 * k - 30 + np.cumsum(rng.normal(0, 4, n))), 3, 98)
        mem = np.clip(np.round(91 + np.cumsum(rng.normal(0, 0.4, n))), 60, 98)
        for i in range(n):
            rows.append([m, i * 10, cpu[i], mem[i], np.nan, np.nan, 30.0, 20.0, 1])
        meta.append([m, 0, 1, 1, 96, 100, "USING"])
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame(rows).sample(frac=1, random_state=0).to_csv(raw / "machine_usage.csv", header=False, index=False)
    pd.DataFrame(meta).to_csv(raw / "machine_meta.csv", header=False, index=False)
    cfg = copy.deepcopy(load_config())
    cfg["data"]["top_n_machines"] = len(machines)
    cfg["data"]["chunk_size"] = 500
    return cfg
