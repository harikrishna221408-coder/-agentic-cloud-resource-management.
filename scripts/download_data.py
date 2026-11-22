"""Download Alibaba Cluster Trace 2018 (machine_usage, machine_meta) with parallel, resumable range requests.

The server is slow for a single connection, so the file is split into blocks fetched by several threads.
Completed blocks are kept in data/raw/.parts/, so re-running the script resumes where it stopped.

Usage:  python scripts/download_data.py [--workers 24]
"""
import argparse
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/"
FILES = ["machine_meta.tar.gz", "machine_usage.tar.gz"]
BLOCK = 16 * 1024 * 1024
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_block(url, start, end, dest):
    if dest.exists() and dest.stat().st_size == end - start + 1:
        return
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError("short read")
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception as e:  # retry
            last = e
    raise last


def download(name, workers):
    url, out = BASE + name, RAW / name
    total = size_of(url)
    if out.exists() and out.stat().st_size == total:
        print(name, "already downloaded")
        return out
    parts_dir = RAW / ".parts" / name
    parts_dir.mkdir(parents=True, exist_ok=True)
    blocks = [(i, s, min(s + BLOCK, total) - 1) for i, s in enumerate(range(0, total, BLOCK))]
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_block, url, s, e, parts_dir / f"{i:05d}") for i, s, e in blocks]
        for f in futs:
            f.result()
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f"{name}: {done}/{len(futs)} blocks", flush=True)
    with open(out, "wb") as w:
        for i, _, _ in blocks:
            w.write((parts_dir / f"{i:05d}").read_bytes())
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for n in FILES:
        archive = download(n, args.workers)
        csv = RAW / n.replace(".tar.gz", ".csv")
        if not csv.exists():
            with tarfile.open(archive) as t:
                t.extractall(RAW)
            print("extracted", csv.name)
    (RAW / "download.done").write_text("ok")
    print("ALL DONE")
