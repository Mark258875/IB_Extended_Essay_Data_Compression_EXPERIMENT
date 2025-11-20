#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import yaml

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
    try:
        d = json.loads(params_json)
        if isinstance(d, dict):
            if "quality" in d: return int(d["quality"])
            if "compression_level" in d: return int(d["compression_level"])
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
    """
    Scatter: measured bits/symbol vs entropy baseline.
    Used both globally (all models) and per-model.
    """
    pts = [p for p in points if p.H_baseline is not None]
    if not pts:
        print(f"[plot] no points with entropy baseline available for {title_suffix!r}.")
        return

    colors = palette()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    for p in pts:
        ax.scatter(
            p.H_baseline,
            p.bp_per_symbol,
            s=50,
            alpha=0.9,
            color=colors.get(p.alg, "k"),
            marker=marker_for_model(p.model),
            label=None,
        )

    # y = x theoretical limit
    m = max(max(p.H_baseline for p in pts), max(p.bp_per_symbol for p in pts))
    m *= 1.05
    ax.plot(
        [0, m],
        [0, m],
        "--",
        color="gray",
        alpha=0.8,
        linewidth=1.5,
        label="y = x (entropy limit)",
    )

    # legends
    from matplotlib.lines import Line2D
    alg_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[a],
            label=a.upper(),
            markersize=7,
        )
        for a in sorted(set(p.alg for p in pts))
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_for_model(mname),
            linestyle="none",
            markerfacecolor="black",
            label=mname,
            markersize=7,
        )
        for mname in sorted(set(p.model for p in pts))
    ]

    ax.legend(
        handles=alg_handles + model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=max(3, len(alg_handles) + len(model_handles)),
        frameon=False,
    )

    title = "Bits per SYMBOL vs Entropy baseline"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Entropy per symbol (H_rate if available else H0) [bits/sym]")
    ax.set_ylabel("Measured bits per symbol [bits/sym]")
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"bp_per_symbol_vs_entropy{filename_suffix}.png"
    out = outdir / fname
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out)
    plt.show()
    print(f"[plot] {out}")

def plot_redundancy(
    points: List[Point],
    outdir: Path,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> None:
    """
    Scatter: redundancy (measured − baseline) vs entropy baseline.
    Only used per-model, not globally (to get 7 graphs total).
    """
    pts = [p for p in points if p.H_baseline is not None and p.H_baseline > 0]
    if not pts:
        print(f"[plot] no points with positive entropy baseline for {title_suffix!r}.")
        return

    colors = palette()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    for p in pts:
        gap = p.bp_per_symbol - p.H_baseline
        ax.scatter(
            p.H_baseline,
            gap,
            s=50,
            alpha=0.9,
            color=colors.get(p.alg, "k"),
            marker=marker_for_model(p.model),
        )

    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7)
    title = "Redundancy vs Entropy baseline"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Entropy per symbol (H_rate if available else H0) [bits/sym]")
    ax.set_ylabel("Redundancy = (bp/sym − H_baseline) [bits/sym]")
    ax.grid(True, alpha=0.3)

    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"redundancy_vs_entropy{filename_suffix}.png"
    out = outdir / fname
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
    model_groups = ["iid_peaked", "zipf", "markov_persistent"]

    for model_name in model_groups:
        pts_m = [p for p in points if p.model == model_name]
        if not pts_m:
            print(f"[plot] no points for model {model_name!r}; skipping.")
            continue

        model_outdir = base_outdir / model_name
        model_outdir.mkdir(parents=True, exist_ok=True)

        # (a) Bits/symbol vs entropy baseline for this model
        plot_universal_yx(
            pts_m,
            model_outdir,
            title_suffix=f"({model_name})",
            filename_suffix=f"_{model_name}",
        )

        # (b) Redundancy vs entropy baseline for this model
        plot_redundancy(
            pts_m,
            model_outdir,
            title_suffix=f"({model_name})",
            filename_suffix=f"_{model_name}",
        )

if __name__ == "__main__":
    main()
