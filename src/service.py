"""Single-machine workflow used by the dashboard and the CLI:

    pick a machine -> see its current usage -> forecast the next CPU/memory -> four agents -> three coordinators

The forecast follows the training setup exactly: the models use the three observations before the current one
(and, for the 'current_value' variant, the current one too; see src/forecasting/predict.py) and predict the
observation AFTER the current one.
"""
from pathlib import Path

import pandas as pd

from src.agents.base import load_agent_config
from src.forecasting.predict import predict_next
from src.orchestration.orchestrator import Orchestrator
from src.utils.config import ROOT, load_config

STRATEGIES = ("original", "improved", "cost_aware")


class MachineStore:
    """Holds the usage history of the selected machines and the machine metadata."""

    def __init__(self, usage_df, meta_df=None, model_dir=None, max_gap=60, variant=None):
        need = {"machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"}
        if not need <= set(usage_df.columns):
            raise ValueError(f"usage data needs columns {sorted(need)}")
        self.usage = usage_df.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True)
        self._groups = {m: g.reset_index(drop=True) for m, g in self.usage.groupby("machine_id")}
        self.meta = {}
        if meta_df is not None:
            last = meta_df.sort_values("time_stamp").groupby("machine_id").tail(1)
            self.meta = last.set_index("machine_id")[["cpu_num", "status"]].to_dict("index")
        self.model_dir = model_dir
        self.variant = variant          # None = the active variant from config/config.yaml
        self.max_gap = max_gap

    @classmethod
    def from_project(cls, root=None):
        root = Path(root) if root else ROOT
        cfg = load_config()
        usage_path = root / cfg["paths"]["selected_usage"]
        if not usage_path.exists():
            raise FileNotFoundError(f"processed data not found: {usage_path}. Run `python -m src.app run` first "
                                    f"(after scripts/download_data.py if the raw data is missing).")
        usage = pd.read_csv(usage_path, usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"])
        meta_path = root / cfg["paths"]["raw_meta"]
        meta = None
        if meta_path.exists():
            from src.data.loader import META_COLUMNS
            meta = pd.read_csv(meta_path, header=None, names=META_COLUMNS)
        return cls(usage, meta, model_dir=root / "models", max_gap=cfg["data"]["max_time_gap_seconds"])

    def machines(self):
        return sorted(self._groups)

    def series(self, machine_id):
        if machine_id not in self._groups:
            raise KeyError(f"unknown machine {machine_id!r}")
        return self._groups[machine_id]

    def n_observations(self, machine_id):
        return len(self.series(machine_id))

    def observation(self, machine_id, index=None):
        """Observation for one machine at row `index` (default: the latest row; negative indexes allowed).

        Returns a dict ready for the orchestrator, plus `actual_next_*` when the following row exists (to check
        the forecast) and `data_warning` if the time gap makes the forecast unreliable.
        """
        g = self.series(machine_id)
        i = len(g) - 1 if index is None else (index if index >= 0 else len(g) + index)
        if not 3 <= i < len(g):
            raise IndexError(f"index must be between 3 and {len(g) - 1} for {machine_id}")
        cur, prev = g.iloc[i], g.iloc[i - 1]
        gap = float(cur["time_stamp"] - prev["time_stamp"])
        hist = g.iloc[i - 3:i + 1]                  # x(t-3), x(t-2), x(t-1), x(t) = the current observation
        pcpu, pmem = predict_next(hist["cpu_util_percent"].tolist(), hist["mem_util_percent"].tolist(),
                                  gap, model_dir=self.model_dir, variant=self.variant)
        obs = {"machine_id": machine_id, "timestamp": int(cur["time_stamp"]),
               "current_cpu": float(cur["cpu_util_percent"]), "predicted_cpu": pcpu,
               "current_memory": float(cur["mem_util_percent"]), "predicted_memory": pmem,
               "input_gap": gap, "source": "usage history + Random Forest forecast"}
        m = self.meta.get(machine_id)
        if m:
            obs["cpu_num"], obs["status"] = int(m["cpu_num"]), m["status"]
        if i + 1 < len(g):
            nxt = g.iloc[i + 1]
            obs["actual_next_cpu"], obs["actual_next_memory"] = float(nxt["cpu_util_percent"]), float(nxt["mem_util_percent"])
        if gap > self.max_gap:
            obs["data_warning"] = f"time gap {gap:.0f}s exceeds {self.max_gap}s used in training; forecast may be unreliable"
        return obs


def analyze_machine(store, machine_id, index=None, strategies=STRATEGIES, orchestrator=None):
    """Full pipeline for one machine. Returns {'observation': ..., 'decisions': {strategy: record}}."""
    obs = store.observation(machine_id, index)
    orch = orchestrator or Orchestrator()
    return {"observation": obs, "decisions": {s: orch.decide(obs, s) for s in strategies}}
