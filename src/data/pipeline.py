"""Run the full data pipeline: raw usage -> top-20 machines -> clean -> features -> CSV.

Usage: python -m src.data.pipeline
"""
from pathlib import Path

import pandas as pd
from src.data import loader, preprocessing
from src.utils.config import ROOT, load_config
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def run(root=None, cfg=None):
    """Build the forecasting dataset under `root` (default: the project folder)."""
    root = Path(root) if root else ROOT
    cfg = cfg or load_config()
    raw = root / cfg["paths"]["raw_usage"]
    if not raw.exists():
        raise FileNotFoundError(f"raw usage file not found: {raw}. Run scripts/download_data.py first.")
    sel_path = root / cfg["paths"]["selected_usage"]
    out_path = root / cfg["paths"]["forecast_dataset"]
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = cfg["data"]["chunk_size"]

    if sel_path.exists():
        selected = pd.read_csv(sel_path)
        log.info("loaded cached selection: %s", sel_path)
    else:
        counts = loader.count_observations(raw, chunk)
        log.info("total rows=%d machines=%d", counts.sum(), len(counts))
        machines = loader.select_top_machines(counts, cfg["data"]["top_n_machines"])
        selected = loader.extract_machines(raw, machines, chunk)
        selected.to_csv(sel_path, index=False)
    log.info("inspection: %s", loader.inspect(selected))

    cleaned = preprocessing.clean(selected)
    feats = preprocessing.build_features(cleaned, cfg["data"]["max_time_gap_seconds"])
    feats.to_csv(out_path, index=False)
    log.info("forecast dataset: %d rows, %d machines -> %s", len(feats), feats["machine_id"].nunique(), out_path)
    return feats


if __name__ == "__main__":
    run()
