"""Loading Alibaba Cluster Trace 2018 files.

The raw CSVs have NO header row, so column names are assigned here
(names taken from the official trace documentation and the original notebook).
"""
import pandas as pd
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

USAGE_COLUMNS = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
                 "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
META_COLUMNS = ["machine_id", "time_stamp", "failure_domain_1", "failure_domain_2",
                "cpu_num", "mem_size", "status"]


def read_usage_chunks(path, chunksize=500_000, usecols=None):
    """Yield the (huge) usage file in chunks so memory stays small."""
    return pd.read_csv(path, header=None, names=USAGE_COLUMNS, chunksize=chunksize, usecols=usecols)


def load_meta(path):
    return pd.read_csv(path, header=None, names=META_COLUMNS)


def count_observations(path, chunksize=500_000):
    """Number of usage rows per machine (one pass over the file)."""
    counts = pd.Series(dtype="int64")
    for chunk in read_usage_chunks(path, chunksize, usecols=["machine_id"]):
        counts = counts.add(chunk["machine_id"].value_counts(), fill_value=0)
    return counts.astype("int64").sort_values(ascending=False)


def extract_machines(path, machine_ids, chunksize=500_000):
    """Read only the rows of the selected machines, sorted by machine and time."""
    ids = set(machine_ids)
    parts = [c[c["machine_id"].isin(ids)] for c in read_usage_chunks(path, chunksize)]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return df.sort_values(["machine_id", "time_stamp"]).reset_index(drop=True)


def select_top_machines(counts, n=20):
    return counts.sort_values(ascending=False).head(n).index.tolist()


def inspect(df):
    """Short data-quality summary (used by notebooks, README and the dashboard)."""
    return {
        "rows": len(df),
        "machines": df["machine_id"].nunique(),
        "missing": df.isnull().sum().to_dict(),
        "duplicate_machine_time": int(df.duplicated(["machine_id", "time_stamp"]).sum()),
        "cpu_range": (df["cpu_util_percent"].min(), df["cpu_util_percent"].max()),
        "mem_range": (df["mem_util_percent"].min(), df["mem_util_percent"].max()),
    }
