#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt

def main():
    # --- Single illustrative point ---
    # x = entropy baseline H (bits per symbol)
    # y = redundancy R = (bp/sym - H) (bits per symbol)
    H_demo = 1.0
    R_demo = 0.20

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

    # zero redundancy line
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7)

    # single red point
    ax.scatter(
        [H_demo],
        [R_demo],
        s=60,
        color="red",
        edgecolor="none",
    )

    # optional annotation
    ax.annotate(
        "example point",
        xy=(H_demo, R_demo),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )

    # axes, grid, title
    ax.set_xlabel("Entropy per symbol $H$ [bits per byte]")
    ax.set_ylabel("Redundancy $R = \\mathrm{bp/sym} - H$ [bits per byte]")
    ax.set_title("Illustrative redundancy vs entropy (methodology example)")

    # pick some reasonable limits so the point is nicely framed
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(-0.05, 0.35)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("redundancy_methodology_demo.png")
    plt.show()

if __name__ == "__main__":
    main()
