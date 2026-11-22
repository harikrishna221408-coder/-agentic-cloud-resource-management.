"""Append-only decision log and traceability checks.

The log is a JSON-lines file: one line per decision, never rewritten. Each line carries a run_id so several runs
can be told apart. verify_trace() proves a record can be traced back to the input it was made from.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.orchestration.negotiation import ALL_AGENTS

TRACE_FIELDS = (("current_cpu", "current_cpu"), ("predicted_cpu", "predicted_cpu"),
                ("current_memory", "current_memory"), ("predicted_memory", "predicted_memory"))


def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]


def append_log(records, path, run_id=None):
    """Append records to the log. Returns the run_id used."""
    run_id = run_id or new_run_id()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"run_id": run_id, **r}, default=str) + "\n")
    return run_id


def read_log(path, run_id=None):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if run_id is None or r["run_id"] == run_id]


def verify_trace(record):
    """List the ways a record fails to trace back to its input (empty list = fully traceable)."""
    problems = []
    inp = record.get("input")
    if not isinstance(inp, dict):
        return ["record has no input observation"] if record.get("decision_status") != "INVALID_INPUT" else []
    if inp.get("machine_id") != record["machine_id"]:
        problems.append("machine_id differs from the input")
    for rec_key, in_key in TRACE_FIELDS:
        if record.get(rec_key) is not None and record[rec_key] != float(inp[in_key]):
            problems.append(f"{rec_key} differs from the input")
    for k in ALL_AGENTS:
        a = record["agents"].get(k)
        if a and a["machine_id"] != record["machine_id"]:
            problems.append(f"{k} agent output belongs to another machine")
        if a and k == "resource":
            if a["predicted_cpu"] != inp["predicted_cpu"] or a["current_memory"] != inp["current_memory"]:
                problems.append("resource agent used different numbers than the input")
    for need in ("timestamp", "decision_status", "explanation", "coordinator_strategy"):
        if need not in record:
            problems.append(f"missing {need}")
    return problems
