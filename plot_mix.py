#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ----------------- config -----------------
# Single tonal gradient for quality (light -> dark)
COLORMAP = "cividis"

# Up to 5 distinct files => 5 marker shapes
MARKERS = ["o", "s", "D", "^", "v"]


# ----------------- data model / parsing -----------------

@dataclass
class Point:
    file_name: str      # basename of dataset path
    quality: Optional[int]
    entropy_bpb: float  # x-axis (from CSV column)
    actual_bpb: float   # y-axis (computed = 8 * compressed_bytes / original_bytes)

def parse_quality(params_json: str) -> Optional[int]:
    def to_int(x) -> Optional[int]:
        try:
            # handle int, float, or numeric string
            return int(float(x))
        except Exception:
            return None

    try:
        d = json.loads(params_json)
        if isinstance(d, dict):
            for key in ("quality", "compression_level", "level", "preset", "acceleration"):
                if key in d:
                    q = to_int(d[key])
                    if q is not None:
                        return q
    except Exception:
        pass
    return None

def read_points(csv_path: Path, alg_name: str) -> List[Point]:
    pts: List[Point] = []
    if not csv_path.exists():
        return pts
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("alg", "").lower() != alg_name:
                continue

            # dataset -> file name (normalize slashes)
            ds = (row.get("dataset") or "").replace("\\", "/")
            file_name = Path(ds).name or ds

            try:
                og = float(row.get("original_bytes", "0"))
                comp = float(row.get("compressed_bytes", "0"))
                ent_bpb = float(row.get("entropy_bpb", "nan"))
            except Exception:
                continue
            if og <= 0:
                continue

            quality = parse_quality(row.get("params", ""))
            actual_bpb = (comp * 8.0) / og

            pts.append(Point(file_name=file_name,
                             quality=quality,
                             entropy_bpb=ent_bpb,
                             actual_bpb=actual_bpb))
    return pts


# ----------------- plotting -----------------

def palette_for_quality(qualities: List[int]) -> Tuple[mcolors.Colormap, mcolors.Normalize]:
    """
    Map numeric quality to a single tonal gradient.
    Returns (cmap, norm) to compute colors via cmap(norm(q)).
    """
    cmap = plt.cm.get_cmap(COLORMAP)
    if not qualities:
        # dummy normalization (unused)
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        qmin, qmax = min(qualities), max(qualities)
        if qmin == qmax:
            # avoid zero-range; make a tiny span around that single value
            norm = mcolors.Normalize(vmin=qmin - 0.5, vmax=qmax + 0.5)
        else:
            norm = mcolors.Normalize(vmin=qmin, vmax=qmax)
    return cmap, norm

def make_plot(points: List[Point], alg: str, out_dir: Path) -> None:
    if not points:
        print(f"[{alg}] no data to plot")
        return

    # map file name -> marker
    file_names = sorted({p.file_name for p in points})
    if len(file_names) > len(MARKERS):
        print(f"[warn] more than {len(MARKERS)} files detected; extra files will reuse marker shapes.")
    marker_map: Dict[str, str] = {fn: MARKERS[i % len(MARKERS)] for i, fn in enumerate(file_names)}

    # build quality -> color mapping via tonal gradient
    qualities = sorted({p.quality for p in points if p.quality is not None})
    cmap, norm = palette_for_quality(qualities)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)

    # scatter points (color by quality, marker by file)
    for p in points:
        color = cmap(norm(p.quality)) if p.quality is not None else (0.5, 0.5, 0.5, 0.8)
        marker = marker_map.get(p.file_name, "o")
        ax.scatter(p.entropy_bpb, p.actual_bpb,
                   s=55, marker=marker, color=color, alpha=0.95, edgecolor="none")

    # y = x (theoretical lower bound)
    x_max = max((p.entropy_bpb for p in points), default=1.0)
    y_max = max((p.actual_bpb for p in points), default=1.0)
    lim = max(x_max, y_max) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.5, color="black", alpha=0.6, label="y = x (theoretical limit)")

    ax.set_title(f"{alg.upper()}: actual compression (bits/byte) vs entropy (bits/byte)")
    ax.set_xlabel("entropy_bpb (bits per byte)")
    ax.set_ylabel("actual bpb = 8 * compressed_bytes / original_bytes")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # --- legends ---
    from matplotlib.lines import Line2D

    # legend for file names (markers only)
    file_handles: List[Line2D] = []
    for fn in file_names:
        file_handles.append(Line2D([0], [0], marker=marker_map[fn], linestyle="none",
                                   markerfacecolor="white", markeredgecolor="black",
                                   label=fn, markersize=8))
    ax.legend(handles=file_handles, title="File (marker)",
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False)

    # colorbar for qualities (tonal gradient shows order clearly)
    if qualities:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("quality")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{alg.lower()}_entropy_vs_actual_by_file.png"
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # room for file legend on right
    fig.savefig(out_path)
    print(f"saved: {out_path}")
    plt.show()


# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot entropy_bpb (x) vs actual bits/byte (y) with y=x; marker by file name, color by quality (tonal gradient)."
    )
    ap.add_argument("-b", "--brotli", action="store_true", help="Use BROTLI results only.")
    ap.add_argument("-l", "--lz4", action="store_true", help="Use LZ4 results only.")
    ap.add_argument("--brotli-csv", type=Path, default=Path("results") / "results_brotli.csv")
    ap.add_argument("--lz4-csv", type=Path, default=Path("results") / "results_lz4.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("plots") / "by_file")
    args = ap.parse_args()

    # decide which algs
    algs: List[Tuple[str, Path]] = []
    if args.brotli:
        algs.append(("brotli", args.brotli_csv))
    if args.lz4:
        algs.append(("lz4", args.lz4_csv))
    if not algs:
        algs = [("brotli", args.brotli_csv), ("lz4", args.lz4_csv)]

    for alg, csv_path in algs:
        pts = read_points(csv_path, alg)
        make_plot(pts, alg, args.out_dir)

if __name__ == "__main__":
    main()
