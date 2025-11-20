#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import yaml


# ---- style config ----
COLORMAP = "cividis"  # single tonal map for quality

# ---------- helpers ----------
def norm(p: str) -> str:
    return str(p).replace("\\", "/")

def load_manifest(manifest_path: Path) -> Dict[str, dict]:
    by_path = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            by_path[norm(row["path"])] = row
    return by_path


def parse_quality(params_json: str) -> Optional[int]:
    def to_int(x) -> Optional[int]:
        try:
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



@dataclass
class Point:
    path: str
    alg: str
    quality: Optional[int]
    K: int
    model: str
    n_symbols: int
    comp_bytes: int
    comp_MB_s: float
    bp_per_symbol: float           # measured
    H0_per_symbol: Optional[float] # from manifest
    Hrate_per_symbol: Optional[float] # from manifest

    @property
    def H_baseline(self) -> Optional[float]:
        # prefer entropy rate if present, else H0
        if self.Hrate_per_symbol not in (None, ""):
            return float(self.Hrate_per_symbol)
        if self.H0_per_symbol not in (None, ""):
            return float(self.H0_per_symbol)
        return None


def palette_for_quality(qualities: List[int], q_min: int = 0, q_max: int = 16):
    """
    Return (cmap, norm) with a FIXED quality range [q_min, q_max].
    This way Brotli (0–11) and LZ4 (0–16) share the same scale.
    """
    cmap = plt.cm.get_cmap(COLORMAP)
    norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
    return cmap, norm


def load_results(results_csv: Path, manifest: Dict[str, dict]) -> List[Point]:
    pts: List[Point] = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            path = norm(row.get("dataset",""))
            if path not in manifest:
                # allow relative/absolute mismatches by trying basename match
                # fall back: skip
                continue
            mf = manifest[path]
            try:
                n_symbols = int(mf["n_symbols"])
                comp_bytes = int(row["compressed_bytes"])
                alg = row["alg"].lower()
                quality = parse_quality(row.get("params",""))
                comp_MB_s = float(row.get("comp_MB_s","nan"))
                # measured bits per symbol
                bp_sym = (comp_bytes * 8.0) / n_symbols if n_symbols > 0 else float("nan")
                K = int(mf["K"])
                model = str(mf["model"])
                H0 = float(mf["measured_H0_bits_per_symbol"]) if mf.get("measured_H0_bits_per_symbol","") else None
                Hrate = float(mf["measured_Hrate_bits_per_symbol"]) if mf.get("measured_Hrate_bits_per_symbol","") else None
                pts.append(Point(
                    path=path, alg=alg, quality=quality, K=K, model=model,
                    n_symbols=n_symbols, comp_bytes=comp_bytes, comp_MB_s=comp_MB_s,
                    bp_per_symbol=bp_sym, H0_per_symbol=H0, Hrate_per_symbol=Hrate
                ))
            except Exception:
                continue
    return pts


def quality_cmap_norm(qualities: List[int], cmap_name: str = "cividis"):
    """
    Return (cmap, norm) so you can color by: cmap(norm(q)).
    Uses a single monotonic tonal palette across quality values.
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


def quality_palette(points: List[Point]):
    """
    Build a mapping {quality -> color} and return (colors, sorted_qualities).
    Only considers points with a non-None quality.
    """
    qualities = sorted({p.quality for p in points if p.quality is not None})
    cmap, norm = quality_cmap_norm(qualities)
    colors = {q: cmap(norm(q)) for q in qualities}
    return colors, qualities


def marker_for_alg(alg: str) -> str:
    """Different marker shapes for algorithms."""
    a = alg.lower()
    if a == "brotli":
        return "o"  # circle
    if a == "lz4":
        return "^"  # triangle
    return "s"      # square for anything else

# ---------- plotting ----------

def palette():
    # simple distinct colors for algs
    return {"brotli": "navy", "lz4": "red"}

def marker_for_model(m: str) -> str:
    return {"iid_peaked": "o", "zipf": "^", "markov_persistent": "s", "histperm": "D"}.get(m, "o")


def plot_universal_yx(
    points: List[Point],
    outdir: Path,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> None:
    # keep only points with a baseline
    pts = [p for p in points if p.H_baseline is not None]
    if not pts:
        print("[plot] no points with entropy baseline available.")
        return

    # collect qualities for colormap scaling
    qualities = sorted({p.quality for p in pts if p.quality is not None})
    cmap, norm = palette_for_quality(qualities, q_min=0, q_max=16)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    # scatter: color by quality, marker by algorithm
    for p in pts:
        color = cmap(norm(p.quality)) if p.quality is not None else (0.5, 0.5, 0.5, 0.8)
        marker = marker_for_alg(p.alg)
        ax.scatter(
            p.H_baseline,
            p.bp_per_symbol,
            s=55,
            alpha=0.95,
            color=color,
            marker=marker,
            edgecolor="none",
        )

    # y=x reference line (theoretical entropy limit)
    m = max(max(p.H_baseline for p in pts), max(p.bp_per_symbol for p in pts))
    m *= 1.05
    ax.plot(
        [0, m],
        [0, m],
        "--",
        color="black",
        alpha=0.6,
        linewidth=1.5,
        label="y = x (entropy limit)",
    )

    # legends: ONLY for algorithms (shapes), quality via colorbar
    from matplotlib.lines import Line2D
    algs = sorted(set(p.alg for p in pts))
    alg_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_for_alg(a),
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            label=a.upper(),
            markersize=8,
        )
        for a in algs
    ]
    if alg_handles:
        ax.legend(
            handles=alg_handles,
            title="Algorithm (marker)",
            loc="upper left",
            frameon=False,
        )

    # colorbar for quality (no legend entries)
    if qualities:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("quality")

    ax.set_title(f"Bits per SYMBOL vs Entropy baseline {title_suffix}".strip())
    ax.set_xlabel("Entropy per symbol (H_rate if available else H0) [bits/sym]")
    ax.set_ylabel("Measured bits per symbol [bits/sym]")
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"bp_per_symbol_vs_entropy{filename_suffix}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.show()
    print(f"[plot] {out}")

def plot_redundancy(
    points: List[Point],
    outdir: Path,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> None:
    pts = [p for p in points if p.H_baseline is not None and p.H_baseline > 0]
    if not pts:
        print("[plot] no points with positive entropy baseline.")
        return

    # color mapping for quality
    qualities = sorted({p.quality for p in pts if p.quality is not None})
    cmap, norm = palette_for_quality(qualities, q_min=0, q_max=16)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    # group for trend-line computation
    by_alg: Dict[str, List[tuple]] = defaultdict(list)

    # scatter: color = quality, marker = algorithm
    for p in pts:
        gap = p.bp_per_symbol - p.H_baseline
        color = cmap(norm(p.quality)) if p.quality is not None else (0.5, 0.5, 0.5, 0.8)
        marker = marker_for_alg(p.alg)

        ax.scatter(
            p.H_baseline,
            gap,
            s=55,
            alpha=0.95,
            color=color,
            marker=marker,
            edgecolor="none",
        )

        by_alg[p.alg].append((p.H_baseline, gap, p.quality))

    # horizontal zero line (ideal = no redundancy)
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7)

    # trend-line colors (you can tweak these)
    trend_colors = {
        "brotli": "navy",
        "lz4": "red",
    }

    # --- trend lines only for highest quality per algorithm ---
    for alg, rows in by_alg.items():
        q_vals = [q for (_, _, q) in rows if q is not None]
        if not q_vals:
            continue
        max_q = max(q_vals)  # should be 11 for Brotli, 16 for LZ4 in your setup

        xs = np.array([H for (H, gap, q) in rows if q == max_q])
        ys = np.array([gap for (H, gap, q) in rows if q == max_q])

        if len(xs) < 2:
            continue  # not enough points to fit a line

        m, b = np.polyfit(xs, ys, deg=1)  # slope, intercept
        x_line = np.linspace(xs.min(), xs.max(), 100)
        y_line = m * x_line + b

        color = trend_colors.get(alg, "black")
        ax.plot(
            x_line,
            y_line,
            linestyle="-",
            linewidth=1.5,
            color=color,
            alpha=0.9,
        )

        # label next to the line: "trend line Brotli (q=11)" etc.
        x_label = x_line[int(len(x_line) * 0.7)]
        y_label = m * x_label + b
        ax.text(
            x_label,
            y_label,
            f"trend line {alg.upper()} (q={max_q})",
            fontsize=8,
            color=color,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    # legend ONLY for algorithms (shapes)
    from matplotlib.lines import Line2D
    algs = sorted(set(p.alg for p in pts))
    alg_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_for_alg(a),
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            label=a.upper(),
            markersize=8,
        )
        for a in algs
    ]
    if alg_handles:
        ax.legend(
            handles=alg_handles,
            title="Algorithm (marker)",
            loc="upper left",
            frameon=False,
        )

    # colorbar for quality
    if qualities:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("quality")

    ax.set_title(f"Redundancy vs Entropy baseline {title_suffix}".strip())
    ax.set_xlabel("Entropy per symbol (H_rate if available else H0) [bits/sym]")
    ax.set_ylabel("Redundancy = (bp/sym − H_baseline) [bits/sym]")
    ax.grid(True, alpha=0.3)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"redundancy_vs_entropy{filename_suffix}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.show()
    print(f"[plot] {out}")

import numpy as np

def plot_redundancy_for_alg(
    points: List[Point],
    alg: str,
    outdir: Path,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> None:
    """
    Redundancy vs entropy baseline for a SINGLE algorithm.
    - color = quality (fixed scale 0–16, shared with LZ4 range)
    - trend line only for highest quality of this algorithm
    - label includes the line equation R = m H + b
    """
    pts = [p for p in points if p.H_baseline is not None and p.H_baseline > 0]
    if not pts:
        print(f"[plot] no points with positive entropy baseline for {alg}.")
        return

    # color mapping for quality (fixed 0–16)
    qualities = sorted({p.quality for p in pts if p.quality is not None})
    cmap, norm = palette_for_quality(qualities, q_min=0, q_max=16)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    # scatter: color = quality, marker = algorithm (all same alg here)
    marker = marker_for_alg(alg)
    for p in pts:
        gap = p.bp_per_symbol - p.H_baseline
        color = cmap(norm(p.quality)) if p.quality is not None else (0.5, 0.5, 0.5, 0.8)

        ax.scatter(
            p.H_baseline,
            gap,
            s=55,
            alpha=0.95,
            color=color,
            marker=marker,
            edgecolor="none",
        )

    # horizontal zero line (ideal = no redundancy)
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7)

    # --- trend line only for highest quality of this algorithm ---
    q_vals = [p.quality for p in pts if p.quality is not None]
    if q_vals:
        max_q = max(q_vals)  # should be 11 for Brotli, 16 for LZ4 given your setup

        xs = np.array([p.H_baseline for p in pts if p.quality == max_q])
        ys = np.array([p.bp_per_symbol - p.H_baseline for p in pts if p.quality == max_q])

        if len(xs) >= 2:
            m, b = np.polyfit(xs, ys, deg=1)  # slope, intercept
            x_line = np.linspace(xs.min(), xs.max(), 100)
            y_line = m * x_line + b

            ax.plot(
                x_line,
                y_line,
                linestyle="-",
                linewidth=1.5,
                color="black",
                alpha=0.9,
            )

            # label near the right part of the line with full equation
            x_label = x_line[int(len(x_line) * 0.7)]
            y_label = m * x_label + b
            ax.text(
                x_label,
                y_label,
                (
                    f"trend line {alg.upper()} (q={max_q})\n"
                    f"R = {m:.3f}·H + {b:.3f}"
                ),
                fontsize=8,
                color="black",
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
        else:
            print(f"[plot] not enough points at max quality {max_q} for {alg} to fit a line.")

    # colorbar for quality (0–16)
    if qualities:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("quality (0–16)")

    ax.set_title(f"{alg.upper()} — Redundancy vs Entropy baseline {title_suffix}".strip())
    ax.set_xlabel("Entropy per symbol (H_rate if available else H0) [bits/sym]")
    ax.set_ylabel("Redundancy = (bp/sym − H_baseline) [bits/sym]")
    ax.grid(True, alpha=0.3)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"redundancy_vs_entropy{filename_suffix}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.show()
    print(f"[plot] {out}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Plot universal per-symbol graphs for k-ary generated sources."
    )
    ap.add_argument("--config", type=Path, default=Path("conf") / "config.yaml")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override manifest path; by default uses karygen.out_root/manifest.csv",
    )

    # Which results to include (default: both if neither is specified)
    ap.add_argument(
        "-b",
        "--brotli",
        action="store_true",
        help="Use Brotli results (results/results_brotli.csv).",
    )
    ap.add_argument(
        "-l",
        "--lz4",
        action="store_true",
        help="Use LZ4 results (results/results_lz4.csv).",
    )

    # Where the result files are
    ap.add_argument(
        "--results-brotli",
        type=Path,
        default=Path("results") / "results_brotli.csv",
    )
    ap.add_argument(
        "--results-lz4",
        type=Path,
        default=Path("results") / "results_lz4.csv",
    )

    args = ap.parse_args()

    # If neither -b nor -l was provided, include both by default
    if not (args.brotli or args.lz4):
        args.brotli = True
        args.lz4 = True

    # Discover manifest path from config if not provided
    if args.manifest is None:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        k = cfg.get("karygen", {})
        out_root = Path(k.get("out_root", "./data/kary"))
        args.manifest = out_root / "manifest.csv"

    if not args.manifest.exists():
        print(f"[err] Manifest CSV not found: {args.manifest}")
        return

    # Load manifest once
    manifest = load_manifest(args.manifest)

    # Accumulate points from the selected result files
    points: List[Point] = []

    if args.brotli:
        if args.results_brotli.exists():
            pts_b = load_results(args.results_brotli, manifest)
            pts_b = [p for p in pts_b if getattr(p, "alg", "brotli").lower() == "brotli"]
            points.extend(pts_b)
        else:
            print(f"[warn] Brotli results not found: {args.results_brotli}")

    if args.lz4:
        if args.results_lz4.exists():
            pts_l = load_results(args.results_lz4, manifest)
            pts_l = [p for p in pts_l if getattr(p, "alg", "lz4").lower() == "lz4"]
            points.extend(pts_l)
        else:
            print(f"[warn] LZ4 results not found: {args.results_lz4}")

    if not points:
        print(
            "[plot] no joined points; manifest present but no results rows matched. "
            "Make sure the 'dataset' in results_* matches manifest 'path'."
        )
        return

    base_outdir = Path("plots") / "kary"
    base_outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Global plot with all models: bits/symbol vs entropy baseline ----
    # This is the "one where are all where is theoretical limit of compression".
    plot_universal_yx(
        points,
        base_outdir,
        title_suffix="(all models)",
        filename_suffix="_all_models",
    )

    # ---- 2) Per-model plots ----
    # We separate into 3 distribution families used in the EE:

    # ---- 2) Per-model plots ----
    model_groups = ["iid_peaked", "zipf", "markov_persistent"]

    for model_name in model_groups:
        pts_m = [p for p in points if p.model == model_name]
        if not pts_m:
            print(f"[plot] no points for model {model_name!r}; skipping.")
            continue

        model_outdir = base_outdir / model_name
        model_outdir.mkdir(parents=True, exist_ok=True)

        # (a) Bits/symbol vs entropy baseline for this model (both algs together)
        plot_universal_yx(
            pts_m,
            model_outdir,
            title_suffix=f"({model_name})",
            filename_suffix=f"_{model_name}",
        )

        # (b) Redundancy vs entropy baseline: SEPARATE per algorithm
        for alg in ("brotli", "lz4"):
            pts_ma = [p for p in pts_m if p.alg == alg]
            if not pts_ma:
                continue
            alg_outdir = model_outdir / alg
            plot_redundancy_for_alg(
                pts_ma,
                alg,
                alg_outdir,
                title_suffix=f"({model_name})",
                filename_suffix=f"_{model_name}_{alg}",
            )


if __name__ == "__main__":
    main()
