#!/usr/bin/env python3
"""
Mocked demo plot (3 points): actual compression (bits/byte) vs. zero-order entropy (bits/byte)
- Exactly 3 points (one per file) with distinct markers
- Color encodes *quality level* using a fixed cividis scale (1 → 11)
- y = x reference line (annotated as COMPRESSION LIMIT, placed below the line)
- One point annotated with a vertical "entropy gap (bpb)" arrow,
  with the label placed to the LEFT of the arrow, showing absolute and % gap.

Usage:
    python mock_entropy_vs_actual_3pts.py --alg BROTLI --seed 7 \
        --out plots/mock_entropy_vs_actual_3pts.png
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import random

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ----------------- config -----------------
COLORMAP = "cividis"
MARKERS = ["o", "s", "D"]  # exactly 3 distinct markers
QUALITY_MIN, QUALITY_MAX = 1, 11  # fixed level scale for colorbar (preserved)

# ----------------- data structures -----------------
@dataclass
class Point:
    file_name: str
    quality: Optional[int]   # quality "level"
    entropy_bpb: float       # x
    actual_bpb: float        # y

# ----------------- helpers -----------------
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def fixed_quality_norm() -> mcolors.Normalize:
    """Fixed normalization so the colorbar scale is preserved."""
    return mcolors.Normalize(vmin=QUALITY_MIN, vmax=QUALITY_MAX)

def synth_three_points(seed: int) -> List[Point]:
    """
    Create exactly three synthetic points:
    - entropy_bpb ~ U(0.8, 6.5)  (bits per byte)
    - actual_bpb  = entropy_bpb * (1 + gap_frac), gap_frac ~ U(0.06, 0.22)
    - quality levels chosen from {1, 5, 9} for visual spread across the scale
    """
    rng = random.Random(seed)
    files = ["file_1.bin", "file_2.bin", "file_3.bin"]
    qualities = [1, 5, 9]  # spread across the fixed [1,11] scale

    pts: List[Point] = []
    for fn, q in zip(files, qualities):
        x = rng.uniform(0.8, 6.5)
        gap_frac = rng.uniform(0.06, 0.22)
        y = x * (1.0 + gap_frac)
        pts.append(Point(file_name=fn, quality=q, entropy_bpb=x, actual_bpb=y))
    return pts

def make_plot(points: List[Point], alg: str, out_path: Path, annotate_one: bool = True) -> None:
    if len(points) != 3:
        raise ValueError("This mock is designed to plot exactly 3 points.")

    # file -> marker
    marker_map: Dict[str, str] = {p.file_name: MARKERS[i] for i, p in enumerate(points)}
    cmap = plt.cm.get_cmap(COLORMAP)
    norm = fixed_quality_norm()  # fixed scale

    fig, ax = plt.subplots(figsize=(9, 7), dpi=120, constrained_layout=True)

    # scatter
    for p in points:
        color = cmap(norm(p.quality)) if p.quality is not None else (0.5, 0.5, 0.5, 0.9)
        ax.scatter(
            p.entropy_bpb, p.actual_bpb,
            s=70, marker=marker_map[p.file_name],
            color=color, edgecolor="none", alpha=0.95, zorder=3
        )

    # y = x reference
    x_max = max(p.entropy_bpb for p in points)
    y_max = max(p.actual_bpb for p in points)
    lim = max(x_max, y_max) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.6, color="black", alpha=0.65,
            label="y = x (theoretical limit)", zorder=1)


    # Label the y=x line as "COMPRESSION LIMIT" BELOW the line, larger font
    ax.annotate(
        "COMPRESSION LIMIT",
        xy=(0.45 * lim, 0.62 * lim),  # a point on the line
        xytext=(0, 0),             # shift down/away so it's not on the line
        textcoords="offset points",
        rotation=45,                  # align with slope = 1
        ha="left", va="top",          # 'top' since label is below the anchor
        fontsize=12, fontweight="bold", color="black", alpha=0.9,
        zorder=2,
    )

    # annotate the point with the largest vertical gap
    if annotate_one:
        target = max(points, key=lambda p: p.actual_bpb - p.entropy_bpb)
        x, y = target.entropy_bpb, target.actual_bpb
        gap_bpb = y - x
        pct = (gap_bpb / x) * 100.0 if x > 0 else float("inf")

        # arrow from (x, x) to (x, y)
        ax.annotate("", xy=(x, y), xytext=(x, x),
                    arrowprops=dict(arrowstyle="->", linewidth=1.4, alpha=0.9, color="black"),
                    zorder=2)
        # label placed just LEFT of the arrow line, showing absolute + percentage
        ax.annotate(
            f"entropy gap (bpb): {gap_bpb:.3f}  ({pct:+.1f}%)",
            xy=(x, (y + x) / 2), xytext=(-10, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
        )

    # axes & layout
    ax.set_title(f"BROTLI/LZ4 | actual_compression (bits/byte) VS entropy(bits/byte)")
    ax.set_xlabel("entropy_bpb (bits per byte)")
    ax.set_ylabel("actual bpb = 8 * compressed_bytes / original_bytes")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # file legend (markers only)
    file_handles = [
        Line2D([0], [0], marker=marker_map[p.file_name], linestyle="none",
               markerfacecolor="white", markeredgecolor="black",
               label=p.file_name, markersize=8)
        for p in points
    ]
    leg1 = ax.legend(handles=file_handles, title="File (marker)",
                     loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.add_artist(leg1)

    # colorbar for levels (fixed scale preserved)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("quality level")
    cbar.set_ticks([1, 3, 5, 7, 9, 11])

    ensure_parent_dir(out_path)
    fig.savefig(out_path)
    print(f"saved: {out_path}")
    plt.show()

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(
        description="BROTLI/LZ4 | actual_compression (bits/byte) VS entropy(bits/byte)"
    )
    ap.add_argument("--alg", type=str, default="brotli", help="Algorithm name for the title.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--no-annotate", action="store_true", help="Disable the sample entropy-gap annotation.")
    ap.add_argument("--out", type=Path, default=Path("plots") / "mock_entropy_vs_actual_3pts.png",
                    help="Output PNG path.")
    args = ap.parse_args()

    pts = synth_three_points(args.seed)
    make_plot(pts, args.alg, args.out, annotate_one=not args.no_annotate)

if __name__ == "__main__":
    main()
