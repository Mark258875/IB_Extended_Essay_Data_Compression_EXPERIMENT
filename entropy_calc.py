#!/usr/bin/env python3
"""
Compute empirical zero‑order entropy for all text files matched by
paths.input_globs in conf/config.yaml.

Outputs ./results/entropy.csv with columns:
  file, size_bytes, unique_bytes, entropy_bpb, max_bits_bytes, eff_vs_8bit,
  decoded_utf8, unique_chars, entropy_bpc, max_bits_chars, eff_vs_max_chars

Definitions
- entropy_bpb: H over the *byte* distribution (0..255), in bits per byte.
- entropy_bpc: H over the *character* distribution (after UTF‑8 decode), in bits per character.
  If a file is not valid UTF‑8, entropy_bpc columns are left blank.

Run:
  python entropy_scan.py
"""
from __future__ import annotations

import csv
import glob
import math
from pathlib import Path
from typing import List
from collections import Counter

from config import Config


def expand_files(patterns: List[str]) -> List[str]:
    files = sorted(set(sum((glob.glob(pat) for pat in patterns), [])))
    return [f for f in files if Path(f).is_file()]

'''
OPTION 1: 

def entropy_bpb(data: bytes) -> float:
    """Zero-order Shannon entropy estimate in bits per byte (byte alphabet 0..255).
    Symbols with p=0 contribute 0, so unused byte values do not affect the sum.
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
'''
def entropy_bpb(data: bytes) -> float:
    """
    Zero-order Shannon entropy in bits per byte, computed only over
    the symbols that actually appear (no fixed 256-bucket table).
    """
    n = len(data)
    if n == 0:
        return 0.0

    inv_n = 1.0 / n
    H = 0.0
    for c in Counter(data).values():  # counts of observed byte values
        p = c * inv_n                 
        H -= p * math.log2(p)
    return H


def entropy_bpc_text(s: str) -> float:
    """Zero-order Shannon entropy in bits per *character* over Unicode code points."""
    n = len(s)
    if n == 0:
        return 0.0
    # Count only observed symbols (no need for fixed-size table here)
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    inv_n = 1.0 / n
    H = 0.0
    for c in freq.values():
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
        w.writerow([
            "file",
            "size_bytes",
            "unique_bytes",
            "entropy_bpb",
            "max_bits_bytes",
            "eff_vs_8bit",
            "decoded_utf8",
            "unique_chars",
            "entropy_bpc",
            "max_bits_chars",
            "eff_vs_max_chars",
        ])

        for fp in files:
            p = Path(fp)
            data = p.read_bytes()
            n_bytes = len(data)

            # Byte-level stats
            H_bpb = entropy_bpb(data)
            # number of distinct bytes actually present
            unique_bytes = len({b for b in data}) if n_bytes else 0
            max_bits_bytes = math.log2(unique_bytes) if unique_bytes > 0 else 0.0
            eff_vs_8bit = (H_bpb / 8.0) if n_bytes else 0.0

            # Character-level stats (UTF-8 decode best-effort)
            decoded_utf8 = False
            unique_chars = ""
            H_bpc = ""
            max_bits_chars = ""
            eff_vs_max_chars = ""
            try:
                text = data.decode("utf-8")
                decoded_utf8 = True
                H_bpc_val = entropy_bpc_text(text)
                uniq_chars_cnt = len(set(text)) if text else 0
                max_bits_chars_val = math.log2(uniq_chars_cnt) if uniq_chars_cnt > 0 else 0.0
                eff_vs_max_chars_val = (H_bpc_val / max_bits_chars_val) if max_bits_chars_val > 0 else 0.0

                unique_chars = str(uniq_chars_cnt)
                H_bpc = f"{H_bpc_val:.6f}"
                max_bits_chars = f"{max_bits_chars_val:.6f}"
                eff_vs_max_chars = f"{eff_vs_max_chars_val:.6f}"
            except UnicodeDecodeError:
                decoded_utf8 = False

            w.writerow([
                p.name,
                n_bytes,
                unique_bytes,
                f"{H_bpb:.6f}",
                f"{max_bits_bytes:.6f}",
                f"{eff_vs_8bit:.6f}",
                str(decoded_utf8),
                unique_chars,
                H_bpc,
                max_bits_chars,
                eff_vs_max_chars,
            ])
            # Console preview
            if decoded_utf8:
                print(f"{p.name:30s} H_bpb={H_bpb:.6f} bits/byte | H_bpc={H_bpc} bits/char (UTF-8)")
            else:
                print(f"{p.name:30s} H_bpb={H_bpb:.6f} bits/byte | (not valid UTF-8)")

    print(f"Wrote entropy for {len(files)} file(s) -> {out_csv}")

if __name__ == "__main__":
    main()
