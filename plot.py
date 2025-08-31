#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# ---------- theory ----------

def binary_entropy(p: float) -> float:
    """H(p) in bits per source bit (Bernoulli i.i.d.)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


def p_grid(n: int = 512) -> List[float]:
    return [i / (n - 1) for i in range(n)]


# ---------- style ----------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def palette_for_qualities(qualities: List[int]) -> Dict[int, str]:
    cmap = plt.cm.get_cmap("tab20")
    return {q: cmap(i % 20) for i, q in enumerate(qualities)}


def markers_for_kind(kind: str) -> str:
    return "o" if kind == "bitpack" else "^" if kind == "ascii01" else "s"


# ---------- parsing ----------

_P_TAG = re.compile(r"(?:^|[\\/])p(?P<p>(?:0(?:\.\d+)?)|(?:1(?:\.0+)?))(?:[\\/]|_)", re.IGNORECASE)

def parse_p_from_dataset_path(dataset_field: str) -> Optional[float]:
    m = _P_TAG.search(dataset_field)
    if not m:
        return None
    try:
        return float(m.group("p"))
    except Exception:
        return None

def infer_kind(dataset_field: str) -> str:
    s = dataset_field.lower()
    if "bitpack" in s:
        return "bitpack"
    if "ascii01" in s:
        return "ascii01"
    return "other"

def infer_order(dataset_field: str) -> str:
    return "random" if "random" in dataset_field.lower() else "sequential"

def parse_quality(params_json: str) -> Optional[int]:
    try:
        d = json.loads(params_json)
        if isinstance(d, dict):
            if "quality" in d and isinstance(d["quality"], (int, float)):
                return int(d["quality"])
            if "compression_level" in d and isinstance(d["compression_level"], (int, float)):
                return int(d["compression_level"])
    except Exception:
        pass
    return None


# ---------- data model ----------

@dataclass
class Row:
    dataset: str
    alg: str
    quality: Optional[int]
    p: float
    kind: str           # ascii01 | bitpack | other
    order: str          # random | sequential
    bpbit: float        # compressed bits per source bit

    @staticmethod
    def _compute_bpbit(kind: str, comp_bytes: float, orig_bytes: float) -> float:
        if orig_bytes <= 0:
            return float("nan")
        if kind == "ascii01":
            # Each source bit stored as one ASCII byte '0'/'1':
            # bpbit = (compressed bits) / (source bits) = (comp_bytes*8) / (orig_bytes*1)
            return (comp_bytes * 8.0) / orig_bytes
        elif kind == "bitpack":
            # Each original byte holds 8 source bits:
            # bpbit = (comp_bytes*8)/(orig_bytes*8) = comp_bytes/orig_bytes
            return comp_bytes / orig_bytes
        else:
            # Not expected here; fall back to bits-per-byte (won’t be compared to H(p) anyway)
            return (comp_bytes * 8.0) / orig_bytes

    @staticmethod
    def from_csv_row(d: Dict[str, str]) -> Optional["Row"]:
        dataset = d.get("dataset", "")
        alg = d.get("alg", "").lower()

        p = parse_p_from_dataset_path(dataset)
        if p is None:
            return None

        kind = infer_kind(dataset)
        order = infer_order(dataset)

        try:
            og = float(d.get("original_bytes", "0"))
            comp = float(d.get("compressed_bytes", "0"))
        except Exception:
            return None

        bpbit = Row._compute_bpbit(kind, comp, og)
        quality = parse_quality(d.get("params", ""))

        return Row(
            dataset=dataset,
            alg=alg,
            quality=quality,
            p=p,
            kind=kind,
            order=order,
            bpbit=bpbit,
        )


def read_rows(csv_path: Path) -> List[Row]:
    if not csv_path.exists():
        return []
    rows: List[Row] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            row = Row.from_csv_row(d)
            if row is not None:
                rows.append(row)
    return rows


# ---------- plotting ----------

def plot_all_qualities(rows: List[Row], alg: str, save_dir: Path) -> None:
    if not rows:
        return
    qualities = sorted({r.quality for r in rows if r.quality is not None})
    colors = palette_for_qualities(qualities)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    xs = p_grid(800)
    ys = [binary_entropy(p) for p in xs]
    ax.plot(xs, ys, linewidth=2.0, label="Binary entropy H(p) (i.i.d. bound)", zorder=1)

    for q in qualities:
        sub = [r for r in rows if r.quality == q]
        for r in sub:
            ax.scatter(
                r.p, r.bpbit,
                s=50, alpha=0.9, color=colors[q],
                marker=markers_for_kind(r.kind),
                label=None, zorder=3
            )

    from matplotlib.lines import Line2D
    qual_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=colors[q], markeredgecolor="none",
               label=f"q={q}", markersize=7)
        for q in qualities
    ]
    qual_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=colors[q], markeredgecolor="none",
               label=f"q={q}", markersize=7)
        for q in qualities
    ]
    kind_handles = [
        Line2D([0], [0], marker=markers_for_kind("bitpack"), linestyle="none",
               markerfacecolor="black", markeredgecolor="none", label="bitpack", markersize=7),
        Line2D([0], [0], marker=markers_for_kind("ascii01"), linestyle="none",
               markerfacecolor="black", markeredgecolor="none", label="ascii01", markersize=7),
    ]
    handles = qual_handles + kind_handles
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(6, max(2, len(handles))),
        frameon=False
    )

    ax.set_title(f"{alg.upper()} — bits per source bit vs p ({rows[0].order})")
    ax.set_xlabel("p (Pr[1])")
    ax.set_ylabel("bits per source bit (bpbit)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)


    ensure_dir(save_dir)
    out = save_dir / f"{alg.lower()}_{rows[0].order}_all_qualities_bpbit_vs_p.png"
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out)
    plt.show()


def plot_per_quality(rows: List[Row], alg: str, save_dir: Path) -> None:
    by_q: Dict[int, List[Row]] = defaultdict(list)
    for r in rows:
        if r.quality is not None:
            by_q[r.quality].append(r)

    for q, sub in sorted(by_q.items()):
        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        xs = p_grid(800)
        ys = [binary_entropy(p) for p in xs]
        ax.plot(xs, ys, linewidth=2.0, label="Binary entropy H(p) (i.i.d. bound)", zorder=1)

        offsets = [(-20, -10), (-20, 10), (20, -10), (20, 10)]
        k = 0
        colors = {"bitpack": "#1f77b4", "ascii01": "#ff7f0e", "other": "#2ca02c"}

        for r in sorted(sub, key=lambda x: x.p):
            y_meas = r.bpbit
            H = binary_entropy(r.p)
            marker = markers_for_kind(r.kind)
            ax.scatter(r.p, y_meas, s=55, color=colors.get(r.kind, "black"),
                       marker=marker, zorder=3)

            # arrow showing gap to H(p)
            ax.annotate("", xy=(r.p, y_meas), xytext=(r.p, H),
                        arrowprops=dict(arrowstyle="->", linewidth=1.2, alpha=0.8,
                                        color=colors.get(r.kind, "black")),
                        zorder=2)
            gap_abs = y_meas - H
            gap_pct = (y_meas / H - 1.0) * 100.0 if H > 0 else float("inf") if y_meas > 0 else 0.0
            off = offsets[k % len(offsets)]; k += 1
            ax.annotate(f"Δ={gap_abs:.3f}  ({gap_pct:+.1f}%)",
                        xy=(r.p, (y_meas + H) / 2), xytext=off,
                        textcoords="offset points", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            ax.annotate(f"Δ={gap_abs:.3f}  ({gap_pct:+.1f}%)",
                        xy=(r.p, (y_meas + H) / 2), xytext=off,
                        textcoords="offset points", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        from matplotlib.lines import Line2D
        kind_handles = [
            Line2D([0], [0], marker=markers_for_kind("bitpack"), linestyle="none",
                   markerfacecolor=colors["bitpack"], markeredgecolor="none", label="bitpack", markersize=7),
            Line2D([0], [0], marker=markers_for_kind("ascii01"), linestyle="none",
                   markerfacecolor=colors["ascii01"], markeredgecolor="none", label="ascii01", markersize=7),
        ]
        ax.legend(
            handles=kind_handles,
            title="Dataset kind",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=len(kind_handles),
            frameon=False
        )

        ax.set_title(f"{alg.upper()} q={q} — bpbit vs p with entropy gap ({sub[0].order})")
        ax.set_xlabel("p (Pr[1])")
        ax.set_ylabel("bits per source bit (bpbit)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)


        ensure_dir(save_dir)
        out = save_dir / f"{alg.lower()}_{sub[0].order}_q{q}_bpbit_vs_p.png"
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(out)
        plt.show()


# ---------- CLI / main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Plot bpbit vs p for binary datasets against the theoretical i.i.d. binary entropy."
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("-r", "--random", action="store_true", help="Use randomized-order datasets.")
    grp.add_argument("-s", "--sequential", action="store_true", help="Use sequentialized datasets (anything not containing 'random').")

    ap.add_argument("-b", "--brotli", action="store_true", help="Plot BROTLI only.")
    ap.add_argument("-l", "--lz4", action="store_true", help="Plot LZ4 only.")
    ap.add_argument("--brotli-csv", type=Path, default=Path("results") / "results_brotli.csv")
    ap.add_argument("--lz4-csv", type=Path, default=Path("results") / "results_lz4.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("plots"))

    args = ap.parse_args()

    want_random = bool(args.random)
    alg_filters = []
    if args.brotli:
        alg_filters.append(("brotli", args.brotli_csv))
    if args.lz4:
        alg_filters.append(("lz4", args.lz4_csv))
    if not alg_filters:
        alg_filters = [("brotli", args.brotli_csv), ("lz4", args.lz4_csv)]

    for alg, csv_path in alg_filters:
        rows = read_rows(csv_path)
        rows = [
            r for r in rows
            if r.alg == alg and r.order == ("random" if want_random else "sequential")
               and r.kind in {"ascii01", "bitpack"}
        ]
        if not rows:
            print(f"[{alg}] no rows matched ({'random' if want_random else 'sequential'}) in {csv_path}")
            continue

        out_dir = args.out_dir / alg
        plot_all_qualities(rows, alg, out_dir)
        plot_per_quality(rows, alg, out_dir)


if __name__ == "__main__":
    main()
