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


import matplotlib.colors as mcolors
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


def quality_cmap_norm(qualities: List[int], cmap_name: str = "cividis"):
    """
    Return (cmap, norm) so you can color by:  cmap(norm(q)).
    Uses a single monotonic tonal palette across quality/level values.
    """
    cmap = plt.cm.get_cmap(cmap_name)
    if not qualities:
        return cmap, mcolors.Normalize(vmin=0, vmax=1)
    qmin, qmax = min(qualities), max(qualities)
    if qmin == qmax:
        # avoid zero span when only one quality exists
        qmin -= 0.5
        qmax += 0.5
    norm = mcolors.Normalize(vmin=qmin, vmax=qmax)
    return cmap, norm


def palette_all_qualities(qualities):
    """Map each quality to a color on a monotonic colormap."""
    cmap, norm = quality_cmap_norm(qualities)
    return {q: cmap(norm(q)) for q in qualities}

def palette_for_qualities(qualities):
    # Same behavior; kept separate because both names are used.
    return palette_all_qualities(qualities)


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
            for k in ("quality", "compression_level", "level", "preset", "q"):
                if k in d:
                    v = d[k]
                    if isinstance(v, (int, float)):
                        return int(v)
                    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                        return int(v.strip())
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
    bpbit: float        # compressed bits per source bit (for H(p) plots)
    bpb_measured: Optional[float] = None      # CSV 'bpb' (bits per *input byte*)
    entropy_bpb: Optional[float] = None       # CSV 'entropy_bpb' (zero-order, bits/byte)

    @staticmethod
    def _compute_bpbit(kind: str, comp_bytes: float, orig_bytes: float) -> float:
        if orig_bytes <= 0:
            return float("nan")
        if kind == "ascii01":
            # bp/bit = (comp_bytes*8)/(orig_bytes*1)
            return (comp_bytes*8) / orig_bytes                    #*8 
        elif kind == "bitpack":
            # bp/bit = (comp_bytes*8)/(orig_bytes*8) = comp_bytes/orig_bytes
            return comp_bytes / orig_bytes
        else:
            return (comp_bytes*8) / orig_bytes                    #*8

    @staticmethod
    def _parse_float_safe(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

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

        # Optional: measured bpb and zero-order entropy_bpb from CSV
        bpb_measured = Row._parse_float_safe(d.get("bpb", ""))
        entropy_bpb = Row._parse_float_safe(d.get("entropy_bpb", ""))

        return Row(
            dataset=dataset,
            alg=alg,
            quality=quality,
            p=p,
            kind=kind,
            order=order,
            bpbit=bpbit,
            bpb_measured=bpb_measured,
            entropy_bpb=entropy_bpb,
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


# ---------- plotting: bpbit vs p (existing) ----------

def plot_all_qualities(rows: List[Row], alg: str, save_dir: Path, kind_filter: Optional[str] = None) -> None:
    if not rows:
        return

    # Optional kind filtering just for this 'overall' plot
    if kind_filter in {"ascii01", "bitpack"}:
        rows = [r for r in rows if r.kind == kind_filter]
        if not rows:
            print(f"[{alg}] no rows matched first-graph kind filter '{kind_filter}'.")
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

    present_kinds = sorted({r.kind for r in rows})
    kind_labels = {
        "bitpack": "bitpack",
        "ascii01": "ascii01",
    }
    kind_handles = [
        Line2D([0], [0], marker=markers_for_kind(k), linestyle="none",
               markerfacecolor="black", markeredgecolor="none",
               label=kind_labels.get(k, k), markersize=7)
        for k in present_kinds
    ]

    handles = qual_handles + kind_handles
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(6, max(2, len(handles))),
        frameon=False
    )

    extra = f", {kind_filter}" if kind_filter in {"ascii01", "bitpack"} else ""
    ax.set_title(f"{alg.upper()} — bits per source bit vs p ({rows[0].order}{extra})")
    ax.set_xlabel("p (Pr[1])")
    ax.set_ylabel("bits per source bit (bpbit)")
    ax.set_xlim(0.0, 1.0)
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


# ---------- NEW: entropy_bpb (x) vs actual bpb (y) ----------

def plot_entropy_vs_bpb(rows: List[Row], alg: str, save_dir: Path, order_label: str) -> None:
    """Scatter: x = zero-order entropy (bits/byte), y = measured compression (bits/byte).
    Adds y=x reference line and 1:1 axes."""
    # Keep rows that actually have both numbers
    use = [r for r in rows if r.bpb_measured is not None and r.entropy_bpb is not None]
    if not use:
        print(f"[{alg}] no rows with both bpb and entropy_bpb present.")
        return

    qualities = sorted({r.quality for r in use if r.quality is not None})
    colors = palette_for_qualities(qualities)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    # Scatter: color by quality, marker by kind
    for r in use:
        c = colors.get(r.quality, "black") if r.quality is not None else "black"
        ax.scatter(r.entropy_bpb, r.bpb_measured, s=50, alpha=0.9,
                   color=c, marker=markers_for_kind(r.kind), zorder=3)

    # y=x reference line
    xy_max = max(
        max((r.entropy_bpb for r in use), default=1.0),
        max((r.bpb_measured for r in use), default=1.0)
    )
    m = xy_max * 1.05
    ax.plot([0, m], [0, m], linestyle="--", linewidth=1.5, color="gray", alpha=0.8, label="y = x")

    # Legends (quality & kind)
    from matplotlib.lines import Line2D
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
    ax.legend(
        handles=qual_handles + kind_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(6, max(2, len(qual_handles) + len(kind_handles))),
        frameon=False
    )

    ax.set_title(f"{alg.upper()} — actual bpb vs zero-order entropy bpb ({order_label})")
    ax.set_xlabel("zero-order entropy (bits per input byte)")
    ax.set_ylabel("actual compression (bits per input byte)")
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    ensure_dir(save_dir)
    out = save_dir / f"{alg.lower()}_{order_label}_entropy_bpb_vs_bpb.png"
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out)
    plt.show()


# ---------- CLI / main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Plot bpbit vs p for binary datasets, and entropy_bpb vs actual bpb."
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("-r", "--random", action="store_true",
                     help="Use randomized-order datasets.")
    grp.add_argument("-s", "--sequential", action="store_true",
                     help="Use sequentialized datasets (anything not containing 'random').")
    grp.add_argument("-m", "--mixed", action="store_true",
                     help="Use BOTH random and sequential datasets; ONLY plot entropy_bpb vs actual bpb.")

    ap.add_argument("-b", "--brotli", action="store_true", help="Plot BROTLI only.")
    ap.add_argument("-l", "--lz4", action="store_true", help="Plot LZ4 only.")
    ap.add_argument("--brotli-csv", type=Path, default=Path("results") / "results_brotli.csv")
    ap.add_argument("--lz4-csv", type=Path, default=Path("results") / "results_lz4.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("plots"))
    ap.add_argument(
    "--first-kind",
    choices=["all", "ascii01", "bitpack"],
    default="all",
    help="Limit ONLY the 'all-qualities' plot to ASCII .txt datasets ('ascii01') "
         "or bitpacked .bin datasets ('bitpack')."
)

    args = ap.parse_args()

    if args.mixed:
        order_filter = None
        order_label = "mixed"
    elif args.random:
        order_filter = "random"
        order_label = "random"
    else:
        order_filter = "sequential"
        order_label = "sequential"

    alg_filters = []
    if args.brotli:
        alg_filters.append(("brotli", args.brotli_csv))
    if args.lz4:
        alg_filters.append(("lz4", args.lz4_csv))
    if not alg_filters:
        alg_filters = [("brotli", args.brotli_csv), ("lz4", args.lz4_csv)]

    for alg, csv_path in alg_filters:
        rows = read_rows(csv_path)
        # keep only our binary datasets
        rows = [r for r in rows if r.alg == alg and r.kind in {"ascii01", "bitpack"}]
        if order_filter is not None:
            rows = [r for r in rows if r.order == order_filter]

        if not rows:
            print(f"[{alg}] no rows matched ({order_label}) in {csv_path}")
            continue

        out_dir = args.out_dir / alg

        if args.mixed:
            # MIXED MODE: show ONLY the entropy vs actual bpb plot
            plot_entropy_vs_bpb(rows, alg, out_dir, order_label)
            continue

        # RANDOM/SEQUENTIAL: show the original charts plus the new one
        plot_all_qualities(rows, alg, out_dir, kind_filter=None if args.first_kind == "all" else args.first_kind)
        plot_per_quality(rows, alg, out_dir)
        plot_entropy_vs_bpb(rows, alg, out_dir, order_label)


if __name__ == "__main__":
    main()
