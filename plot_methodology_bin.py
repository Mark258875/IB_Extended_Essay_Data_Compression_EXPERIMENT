#!/usr/bin/env python3
"""
One-point demo plot: binary entropy H(p) (navy) with a single red mocked point and an 'entropy gap' arrow.

Usage:
    python mock_entropy_gap_one_point.py --p 0.3 --gap 0.12 --out plots/mock_entropy_gap_one_point.png
"""

from __future__ import annotations
import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----- helpers -----

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def binary_entropy(p: float) -> float:
    """H(p) in bits per source bit (Bernoulli i.i.d.)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


def p_grid(n: int = 800):
    return [i / (n - 1) for i in range(n)]


# ----- main -----

def main():
    ap = argparse.ArgumentParser(description="Plot H(p) with a single mocked entropy gap point.")
    ap.add_argument("--p", type=float, default=0.30, help="Bernoulli parameter p (0<p<1).")
    ap.add_argument("--gap", type=float, default=0.12, help="Absolute gap above H(p).")
    ap.add_argument("--out", type=Path, default=Path("plots") / "mock_entropy_gap_one_point.png",
                    help="Output PNG path.")
    args = ap.parse_args()

    # Clamp p to (0,1) open interval to avoid H=0 edge cases
    p = min(max(args.p, 1e-6), 1 - 1e-6)
    H = binary_entropy(p)
    y_meas = H + args.gap
    gap_val = y_meas - H
    pct = (gap_val / H) * 100.0 if H > 0 else float("inf")

    xs = p_grid(800)
    ys = [binary_entropy(t) for t in xs]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=120)

    # H(p) in navy
    ax.plot(xs, ys, linewidth=2.0, color="navy", label="H(p)")

    # Single point in red
    ax.scatter(p, y_meas, s=60, color="red", zorder=3, label="file")

    # Arrow showing the entropy gap
    ax.annotate(
        "",
        xy=(p, y_meas),
        xytext=(p, H),
        arrowprops=dict(arrowstyle="->", linewidth=1.3, alpha=0.9, color="black"),
        zorder=2,
    )

    # Label the entropy gap on the LEFT of the arrow midpoint, including percentage change
    ax.annotate(
        f"entropy gap: {gap_val:.4f}  ({pct:+.1f}%)",
        xy=(p, (y_meas + H) / 2),   # midpoint of the arrow
        xytext=(-16, 0),            # left of the line
        textcoords="offset points",
        ha="right", va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
    )

    # Legend: H(p) and the mocked point
    handles = [
        Line2D([0], [0], color="navy", lw=2.0, label="H(p)"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="red", markeredgecolor="red", markersize=7, label="file"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=2
    )

    # Axes & layout
    ax.set_title("BROTLI/LZ4 (qualitative level = N) bpb vs. H(p) ⇒ ENTROPY GAP")
    ax.set_xlabel("p (Pr[X=1])")
    ax.set_ylabel("bits per source bit")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1.05, y_meas * 1.10))
    ax.grid(True, alpha=0.3)

    ensure_parent_dir(args.out)
    try:
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    except Exception:
        pass
    fig.savefig(args.out)
    plt.show()


if __name__ == "__main__":
    main()
