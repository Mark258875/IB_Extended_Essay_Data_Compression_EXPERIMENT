#!/usr/bin/env python3
"""
Compute empirical (zero‑order) Shannon entropy in bits per byte (H1) for all
text files matched by paths.input_globs in conf/config.yaml, and write a CSV
with two columns: file, entropy_bpb.

Output file: <paths.output_dir>/entropy.csv

Usage:
  python entropy_scan.py
"""
from __future__ import annotations

import csv
import glob
import math
from pathlib import Path

from config import Config


def expand_files(patterns):
    """Expand a list of glob patterns into a sorted, de-duplicated file list."""
    files = sorted(set(sum((glob.glob(pat) for pat in patterns), [])))
    return [f for f in files if Path(f).is_file()]


def entropy_bpb(data: bytes) -> float:
    """Zero-order Shannon entropy estimate in bits per byte.
    Returns 0.0 for empty input.
    """
    n = len(data)
    if n == 0:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    H = 0.0
    inv_n = 1.0 / n
    for c in counts:
        if c:
            p = c * inv_n
            H -= p * math.log2(p)
    return H


def main() -> None:
    cfg = Config.load("conf/config.yaml")
    files = expand_files(cfg.paths.input_globs)
    if not files:
        print("No input files matched. Edit conf/config.yaml -> paths.input_globs")
        return

    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "entropy.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "entropy_bpb"])  # header
        for fp in files:
            data = Path(fp).read_bytes()
            H = entropy_bpb(data)
            w.writerow([Path(fp).name, f"{H:.6f}"])
            print(f"{Path(fp).name:30s} H1={H:.6f} bits/byte")

    print(f"\nWrote entropy for {len(files)} file(s) -> {out_csv}")


if __name__ == "__main__":
    main()
