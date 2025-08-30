#!/usr/bin/env python3
# plot_binary_entropy_vs_results.py
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

# ---------- config ----------
RESULTS_DIR = Path("results")
BROTLI_CSV = RESULTS_DIR / "results_brotli.csv"
LZ4_CSV    = RESULTS_DIR / "results_lz4.csv"
PLOT_DIR   = RESULTS_DIR / "plots"

# We compare to *binary* entropy, so we only use datasets produced by the 0/1 generator.
# Theoretical comparison H(p) is valid for i.i.d. Bernoulli sources.
# We'll use **random** order only (not alternating/blocks), because alternating/blocks are not i.i.d.
USE_ONLY_RANDOM = True

# ---------- helpers & types ----------

@dataclass
class Row:
    alg: str                 # 'brotli' or 'lz4' (as in CSV)
    dataset: str             # dataset path string from CSV
    p: float                 # Bernoulli p extracted from path
    quality: Optional[int]   # brotli quality or lz4 level
    order: str               # 'random' | 'alternating' | 'blocks' | 'other'
    kind: str                # 'ascii01' | 'bitpack' | 'other'
    s_bits_per_byte: int     # 1 for ascii01; 8 for bitpack; otherwise inferred
    original_bytes: int
    compressed_bytes: int
    ratio: float
    bpb: float               # bits per *byte* (8 * ratio)
    bpbit: float             # bits per *source bit* = bpb / s_bits_per_byte

def binary_entropy(p: float) -> float:
    """H(p) in bits per bit."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))

_re_p = re.compile(r"[\\/]+p([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def extract_p(dataset_path: str) -> Optional[float]:
    m = _re_p.search(dataset_path)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def detect_kind_and_order(dataset_path: str) -> Tuple[str, str, int]:
    lower = dataset_path.lower()
    kind = "bitpack" if "bitpack" in lower or lower.endswith(".bin") else \
           ("ascii01" if "ascii01" in lower or lower.endswith(".txt") else "other")
    if "_random_" in lower:
        order = "random"
    elif "_alternating_" in lower:
        order = "alternating"
    elif "_blocks_" in lower:
        order = "blocks"
    else:
        order = "other"
    s = 8 if kind == "bitpack" else 1
    return kind, order, s

def parse_quality(params_field: str, alg: str) -> Optional[int]:
    """Params is JSON in the CSV (with proper CSV quoting)."""
    try:
        params = json.loads(params_field)
    except Exception:
        return None
    # brotli: "quality"
    if alg.lower() == "brotli":
        q = params.get("quality")
        return int(q) if q is not None else None
    # lz4: use "lz4_level" or "level" if present
    q = params.get("lz4_level", params.get("level"))
    return int(q) if q is not None else None

def read_results(csv_path: Path, alg_expect: str) -> List[Row]:
    rows: List[Row] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line in r:
            alg = line.get("alg", "").strip().lower()
            if alg != alg_expect:
                continue
            dataset = line.get("dataset", "")
            # only consider our 0/1 experiment inputs
            if "data" not in dataset or "binary" not in dataset:
                continue
            p = extract_p(dataset)
            if p is None:
                continue

            kind, order, s = detect_kind_and_order(dataset)
            if USE_ONLY_RANDOM and order != "random":
                continue

            try:
                orig = int(line["original_bytes"])
                comp = int(line["compressed_bytes"])
            except Exception:
                # compute from ratio if needed
                try:
                    ratio = float(line.get("ratio", "nan"))
                    # If original_bytes missing, skip (we need bpbit)
                    continue
                except Exception:
                    continue

            # prefer explicit ratio; compute if missing
            try:
                ratio = float(line.get("ratio", "")) if line.get("ratio") else comp / orig
            except Exception:
                ratio = comp / orig

            bpb = 8.0 * ratio
            bpbit = bpb / s

            quality = parse_quality(line.get("params", "{}"), alg)

            rows.append(Row(
                alg=alg_expect,
                dataset=dataset,
                p=p,
                quality=quality,
                order=order,
                kind=kind,
                s_bits_per_byte=s,
                original_bytes=orig,
                compressed_bytes=comp,
                ratio=ratio,
                bpb=bpb,
                bpbit=bpbit
            ))
    return rows

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def p_grid(n: int = 1000) -> List[float]:
    return [i / n for i in range(n + 1)]

# ---------- plotting ----------

def palette_for_qualities(qualities: List[int]) -> Dict[int, str]:
    # deterministic mapping using tab20
    cmap = plt.get_cmap("tab20")
    colors = {}
    for i, q in enumerate(sorted(qualities)):
        colors[q] = cmap(i % 20)
    return colors

def markers_for_kind(kind: str) -> str:
    return "o" if kind == "bitpack" else "^" if kind == "ascii01" else "s"

def plot_all_qualities(rows: List[Row], alg: str, save_dir: Path) -> None:
    if not rows:
        return
    # collect qualities present
    qualities = sorted({r.quality for r in rows if r.quality is not None})
    colors = palette_for_qualities(qualities)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    # theoretical H(p) curve
    xs = p_grid(800)
    ys = [binary_entropy(p) for p in xs]
    ax.plot(xs, ys, linewidth=2.0, label="Binary entropy H(p)", zorder=1)

    # points grouped by quality
    for q in qualities:
        sub = [r for r in rows if r.quality == q]
        for r in sub:
            ax.scatter(r.p, r.bpbit,
                       s=50, alpha=0.9, color=colors[q],
                       marker=markers_for_kind(r.kind),
                       label=None, zorder=3)
        # add one legend entry per quality (proxy)
        ax.scatter([], [], s=60, color=colors[q], label=f"q={q}", marker="o")

    # marker legend for kind
    from matplotlib.lines import Line2D
    kind_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black", label="bitpack", markersize=7),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="black", label="ascii01", markersize=7),
    ]

    ax.set_title(f"{alg.upper()} — bits per source bit vs p (randomized binary data)")
    ax.set_xlabel("p (Pr[1])")
    ax.set_ylabel("bits per source bit (bpbit)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)  # H(p) ≤ 1
    leg1 = ax.legend(loc="upper center", ncol=min(6, max(2, len(qualities))), frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=kind_handles, loc="upper right", frameon=False, title="Dataset kind")

    ax.grid(True, alpha=0.3)
    ensure_dir(save_dir)
    out = save_dir / f"{alg.lower()}_all_qualities_bpbit_vs_p.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.show()

def write_gap_csv(rows: List[Row], alg: str, out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "alg","quality","p","dataset","kind","order",
            "bpbit","H_p","gap_abs","gap_pct"
        ])
        for r in sorted(rows, key=lambda x: (x.quality if x.quality is not None else -1, x.p)):
            H = binary_entropy(r.p)
            gap_abs = r.bpbit - H
            gap_pct = (r.bpbit / H - 1.0) * 100.0 if H > 0 else float("inf") if r.bpbit > 0 else 0.0
            w.writerow([
                r.alg, r.quality if r.quality is not None else "",
                f"{r.p:.6f}", r.dataset, r.kind, r.order,
                f"{r.bpbit:.6f}", f"{H:.6f}",
                f"{gap_abs:.6f}", f"{gap_pct:.2f}"
            ])

def plot_per_quality(rows: List[Row], alg: str, save_dir: Path) -> None:
    # group by quality
    by_q: Dict[int, List[Row]] = defaultdict(list)
    for r in rows:
        if r.quality is not None:
            by_q[r.quality].append(r)

    for q, sub in sorted(by_q.items()):
        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        xs = p_grid(800)
        ys = [binary_entropy(p) for p in xs]
        ax.plot(xs, ys, linewidth=2.0, label="Binary entropy H(p)", zorder=1)

        # points & arrows
        # alternate label offsets to reduce clutter
        offsets = [(-20, -10), (-20, 10), (20, -10), (20, 10)]
        k = 0

        # show bitpack vs ascii markers separately
        colors = {"bitpack": "#1f77b4", "ascii01": "#ff7f0e", "other": "#2ca02c"}

        kinds_present = sorted({r.kind for r in sub})
        for r in sorted(sub, key=lambda x: x.p):
            y_meas = r.bpbit
            H = binary_entropy(r.p)
            marker = markers_for_kind(r.kind)
            ax.scatter(r.p, y_meas, s=55, color=colors.get(r.kind, "black"),
                       marker=marker, zorder=3)

            # arrow from (p, H) to (p, y_meas)
            ax.annotate(
                "",
                xy=(r.p, y_meas),
                xytext=(r.p, H),
                arrowprops=dict(arrowstyle="->", linewidth=1.2, alpha=0.8, color=colors.get(r.kind, "black")),
                zorder=2,
            )
            # label
            gap_abs = y_meas - H
            gap_pct = (y_meas / H - 1.0) * 100.0 if H > 0 else float("inf") if y_meas > 0 else 0.0
            off = offsets[k % len(offsets)]; k += 1
            ax.annotate(
                f"Δ={gap_abs:.3f}  ({gap_pct:+.1f}%)",
                xy=(r.p, (y_meas + H) / 2),
                xytext=off,
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

        # legends
        from matplotlib.lines import Line2D
        kind_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["bitpack"], label="bitpack", markersize=7),
            Line2D([0], [0], marker="^", color="none", markerfacecolor=colors["ascii01"], label="ascii01", markersize=7),
        ]
        ax.legend(handles=kind_handles, title="Dataset kind", loc="upper right", frameon=False)

        ax.set_title(f"{alg.upper()} q={q} — bpbit vs p with entropy gap")
        ax.set_xlabel("p (Pr[1])")
        ax.set_ylabel("bits per source bit (bpbit)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ensure_dir(save_dir)
        out = save_dir / f"{alg.lower()}_q{q}_bpbit_vs_p.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.show()

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Plot binary entropy vs measured compression (bits per source bit) for 0/1 randomized data."
    )
    parser.add_argument("-b", "--brotli", action="store_true", help="Plot only Brotli figures")
    parser.add_argument("-l", "--lz4",    action="store_true", help="Plot only LZ4 figures")
    args = parser.parse_args()

    do_brotli = args.brotli or (not args.brotli and not args.lz4)
    do_lz4    = args.lz4    or (not args.brotli and not args.lz4)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if do_brotli:
        bro_rows = read_results(BROTLI_CSV, "brotli")
        # All-qualities chart
        plot_all_qualities(bro_rows, "brotli", PLOT_DIR)
        # Per-quality charts with arrows & gap CSV
        plot_per_quality(bro_rows, "brotli", PLOT_DIR)
        write_gap_csv(bro_rows, "brotli", RESULTS_DIR / "entropy_gap_brotli.csv")

    if do_lz4:
        lz4_rows = read_results(LZ4_CSV, "lz4")
        plot_all_qualities(lz4_rows, "lz4", PLOT_DIR)
        plot_per_quality(lz4_rows, "lz4", PLOT_DIR)
        write_gap_csv(lz4_rows, "lz4", RESULTS_DIR / "entropy_gap_lz4.csv")

if __name__ == "__main__":
    main()
