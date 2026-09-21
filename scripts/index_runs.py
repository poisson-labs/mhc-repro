#!/usr/bin/env python3
"""Index the Part 2 run directories into a CSV manifest.

Reads runs_c4/*/ (the 18 main runs) and runs_c4_stress/*/ (the stress runs). Each run
directory holds config.json and, where it was kept, history.json. Both files are written
when a run finishes, so the modification time of config.json is the finish time.

One row per run: its settings, the commit it recorded, when it started and finished,
the SHA-256 of its history.json, and the numbers the figure scripts derive from it
(loss at the last step, the maximum of composite_amax over all steps).

The output is deterministic: rows are sorted, times are UTC, and no path is absolute.
The start time is the `timestamp` field of config.json, taken as UTC (the training node's
clock). The finish time is a file modification time, so it is only as good as the copy
of the run directory it was read from.

Usage:
    python scripts/index_runs.py [--runs-root .] [--out manifests/part2-runs.csv]
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RUN_SETS = (("main", "runs_c4"), ("stress", "runs_c4_stress"))
COLUMNS = [
    "set", "run", "method", "depth", "seed", "batch_size", "learning_rate", "max_steps",
    "recorded_git_hash", "started_utc", "finished_utc", "duration_h",
    "history_bytes", "history_sha256", "history_records", "history_trailing_bytes",
    "final_train_loss", "max_composite_amax",
]


def utc_iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_history(path):
    """Return (records, bytes_after_first_json_value). Tolerates trailing bytes."""
    text = path.read_text()
    records, end = json.JSONDecoder().raw_decode(text)
    return records, len(text) - end


def index_run(set_name, run_dir):
    cfg = json.loads((run_dir / "config.json").read_text())
    started = datetime.fromisoformat(cfg["timestamp"]).replace(tzinfo=timezone.utc)
    finished = datetime.fromtimestamp((run_dir / "config.json").stat().st_mtime, timezone.utc)
    row = {
        "set": set_name,
        "run": run_dir.name,
        "method": cfg["model"]["connection_type"],
        "depth": cfg["model"]["n_layers"],
        "seed": cfg["train"]["seed"],
        "batch_size": cfg["train"]["batch_size"],
        "learning_rate": cfg["train"]["learning_rate"],
        "max_steps": cfg["train"]["max_steps"],
        "recorded_git_hash": cfg["git_hash"],
        "started_utc": utc_iso(started),
        "finished_utc": utc_iso(finished),
        "duration_h": f"{(finished - started).total_seconds() / 3600:.2f}",
    }
    hist = run_dir / "history.json"
    if hist.exists():
        records, trailing = read_history(hist)
        amax = [r["composite_amax"] for r in records if "composite_amax" in r]
        row.update({
            "history_bytes": hist.stat().st_size,
            "history_sha256": hashlib.sha256(hist.read_bytes()).hexdigest(),
            "history_records": len(records),
            "history_trailing_bytes": trailing,
            "final_train_loss": f"{records[-1]['loss']:.6f}",
            "max_composite_amax": f"{max(amax):.6g}" if amax else "",
        })
    return row, started, finished


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", default=".", help="directory holding runs_c4/ and runs_c4_stress/")
    parser.add_argument("--out", default="manifests/part2-runs.csv")
    args = parser.parse_args()

    root = Path(args.runs_root)
    rows, spans = [], {}
    for set_name, dirname in RUN_SETS:
        for cfg_path in sorted((root / dirname).glob("*/config.json")):
            row, started, finished = index_run(set_name, cfg_path.parent)
            rows.append(row)
            first, last = spans.get(set_name, (started, finished))
            spans[set_name] = (min(first, started), max(last, finished))
    rows.sort(key=lambda r: (r["set"], r["method"], int(r["depth"]), int(r["seed"])))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out}: {len(rows)} runs, {sum(1 for r in rows if 'history_sha256' in r)} with a history.json")
    for name, (first, last) in spans.items():
        print(f"{name:7s} first start {utc_iso(first)}  last finish {utc_iso(last)}  "
              f"span {(last - first).total_seconds() / 3600:.2f} h")
    if len(spans) > 1:
        first = min(s[0] for s in spans.values())
        last = max(s[1] for s in spans.values())
        print(f"{'all':7s} first start {utc_iso(first)}  last finish {utc_iso(last)}  "
              f"span {(last - first).total_seconds() / 3600:.2f} h")


if __name__ == "__main__":
    main()
