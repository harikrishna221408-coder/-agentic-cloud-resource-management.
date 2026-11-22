"""Shared helpers for the four agents: actions, config loading and input validation.

Every agent receives the same observation dictionary:
  machine_id, current_cpu, predicted_cpu, current_memory, predicted_memory   (required, 0-100)
  cpu_num, status, timestamp                                                  (optional machine metadata)
and returns a dictionary that always contains: agent, machine_id, status, vote, risk_level, reason, metrics.
"""
import math
import numbers
from pathlib import Path

import yaml

from src.utils.config import ROOT

SCALE_UP, SCALE_DOWN, MAINTAIN = "SCALE_UP", "SCALE_DOWN", "MAINTAIN"
ACTIONS = (SCALE_UP, SCALE_DOWN, MAINTAIN)
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
REQUIRED = ("machine_id", "current_cpu", "predicted_cpu", "current_memory", "predicted_memory")


class InvalidObservation(ValueError):
    """Raised when an agent gets missing, non-numeric or out-of-range input."""


def load_agent_config(path=None):
    path = Path(path) if path else ROOT / "config" / "agents.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _number(name, value):
    """A real, finite number. Booleans and strings are rejected on purpose: they usually mean broken data."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidObservation(f"{name} is not numeric: {value!r}")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise InvalidObservation(f"{name} must be a finite number, got {v}")
    return v


def validate_observation(obs):
    """Return a cleaned copy of obs (floats), or raise InvalidObservation with a clear message."""
    if not isinstance(obs, dict):
        raise InvalidObservation("observation must be a dict")
    missing = [k for k in REQUIRED if k not in obs or obs[k] is None]
    if missing:
        raise InvalidObservation(f"missing fields: {missing}")
    mid = obs["machine_id"]
    if isinstance(mid, bool) or not isinstance(mid, (str, int)) or (isinstance(mid, str) and not mid.strip()):
        raise InvalidObservation(f"machine_id must be a non-empty string or integer, got {mid!r}")
    clean = dict(obs)
    for k in REQUIRED[1:]:
        v = _number(k, obs[k])
        if not 0 <= v <= 100:
            raise InvalidObservation(f"{k} must be between 0 and 100, got {v}")
        clean[k] = v
    if obs.get("cpu_num") is not None:                       # optional machine metadata
        n = _number("cpu_num", obs["cpu_num"])
        if n <= 0:
            raise InvalidObservation(f"cpu_num must be positive, got {n}")
        clean["cpu_num"] = n
    if obs.get("status") is not None and not isinstance(obs["status"], str):
        raise InvalidObservation(f"status must be a string, got {obs['status']!r}")
    return clean


def risk_max(*levels):
    return max(levels, key=lambda r: RISK_ORDER[r])
