#!/usr/bin/env python3
from __future__ import annotations
import math, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_BROTLI = Path("results/results_brotli.csv")
RESULTS_LZ4    = Path("results/results_lz4.csv")
OUT_DIR        = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def h2(p: float) -> float:
    """Binary entropy in bits/bit."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p*math.log2(p) + (1-p)*math.log2(1-p))

def parse_dataset_path(s: str):
    """
    Extract (rep, p, order) from dataset path like:
      data\\binary\\ascii01\\p0.10\\p0.10_random_10000000B.txt
      data/binary/bitpack/p0.30/p0.30_blocks_zeros_then_ones_...bin
    """
    ss = s.replace("\\", "/")
    m = re.search(r"/binary/(ascii01|bitpack)/p([0-9.]+)/([^/]+)$", ss, re.IGNORECASE)
    if not m:
        return None
    rep = m.group(1).lower()
    p = float(m.group(2))
    fname = m.group(3).lower()
    if "random" in fname:
        order = "random"
    elif "alternating" in fname:
        order = "alternating"
    elif "blocks_zeros_then_ones" in fname or "blocks" in fname:
        order = "blocks_zeros_then_ones"
    else:
        order = "unknown"
    return rep, p, order

def extract_quality(alg: str, params_str: str) -> str:
    """
    Extract a small label for 'quality' (brotli) or 'level' (lz4) out of CSV params JSON.
    Falls back to 'q=?' if not found.
    """
    qlab = "q=?"
    try:
        # CSV stores JSON with doubled quotes → still valid JSON for json.loads
        params = json.loads(params_str)
        if alg.lower() == "brotli":
            qval = params.get("quality")
            if qval is not None:
                qlab = f"q={qval}"
        else:
            # our LZ4 sweep uses 'level' or 'compression_level' depending on version
            qval = params.get("level", params.get("compression_level"))
            if qval is not None:
                qlab = f"lvl={qval}"
    except Exception:
        pass
    return qlab

# --- Load & combine any available results ---
frames = []
if RESULTS_BROTLI.exists(): frames.append(pd.read_csv(RESULTS_BROTLI))
if RESULTS_LZ4.exists():    frames.append(pd.read_csv(RESULTS_LZ4))
if not frames:
    raise FileNotFoundError("No results CSVs found. Run sweeps first.")
df = pd.concat(frames, ignore_index=True)

# --- Parse dataset meta + quality/level labels ---
parsed = df["dataset"].apply(parse_dataset_path)
mask = parsed.notnull()
df = df[mask].reset_index(drop=True)
meta = pd.DataFrame(parsed[mask].tolist(), columns=["rep", "p", "order"])
df = pd.concat([df, meta], axis=1)
df["qlab"] = [extract_quality(a, ps) for a, ps in zip(df["alg"], df["params"])]

# Only keep the three designed orders we know how to talk about
df = df[df["order"].isin(["random", "alternating", "blocks_zeros_then_ones"])]

# --- Plot helpers ---
ALG_MARKER = {"brotli": "o", "lz4": "^"}  # circle for brotli, triangle for lz4
EPS = 1e-6  # tiny tolerance for the bound check

def plot_one(rep: str, order: str):
    """
    For a given representation and order:
      - plot theory curve (if order == 'random')
      - plot bpb by p, one color per quality/level, different marker per alg
    """
    sub = df[(df["rep"] == rep) & (df["order"] == order)].copy()
    if sub.empty:
        return None

    # p-grid for the smooth theory curve
    pgrid = np.linspace(0.0, 1.0, 400)
    if rep == "ascii01":
        theory_bpb = h2  # bits/byte = H(p) (each input byte is one Bernoulli bit)
        theory_label = "Theory: H(p)"
        scale = 1.0
    else:
        theory_bpb = lambda p: 8.0 * h2(p)  # bitpacked: 8 bits per input byte
        theory_label = "Theory: 8·H(p)"
        scale = 8.0

    # Build consistent color map by quality label across both algs for this rep+order
    qlabels = sorted(sub["qlab"].unique(),
                     key=lambda s: (s.split("=")[0], int(s.split("=")[1]) if s.split("=")[1].isdigit() else 999))
    cmap = plt.cm.get_cmap("tab20", max(1, len(qlabels)))
    q_to_color = {q: cmap(i) for i, q in enumerate(qlabels)}

    # --- bound violations report (random only) ---
    violations = []
    if order == "random":
        for _, r in sub.iterrows():
            bound = (h2(r["p"]) if rep == "ascii01" else 8.0*h2(r["p"]))
            if r["bpb"] + EPS < bound:
                violations.append((r["alg"], r["qlab"], r["p"], r["bpb"], bound))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    if order == "random":
        ax.plot(pgrid, [theory_bpb(p) for p in pgrid], lw=2, label=theory_label)

    # scatter per (alg, qlab)
    for alg in sorted(sub["alg"].unique()):
        m = ALG_MARKER.get(alg.lower(), "s")
        for q in qlabels:
            d = sub[(sub["alg"] == alg) & (sub["qlab"] == q)].sort_values("p")
            if d.empty: 
                continue
            ax.plot(d["p"], d["bpb"], linestyle="none", marker=m, markersize=5,
                    label=f"{alg} {q}", color=q_to_color[q], alpha=0.9)

    ax.set_xlabel("p = P(bit=1)")
    ax.set_ylabel("bits per byte (bpb)")
    title = f"{rep} — {order} — measured bpb vs. theory"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    # save
    out_png = OUT_DIR / f"{rep}_{order}_bpb_vs_theory_by_quality.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # write violations
    if violations:
        rpt = OUT_DIR / f"{rep}_{order}_bound_violations.txt"
        with rpt.open("w", encoding="utf-8") as f:
            f.write("# Any measured bpb below the theoretical lower bound (random Bernoulli) would appear here.\n")
            f.write("# Columns: alg, quality, p, bpb_measured, bpb_bound\n")
            for (alg, qlab, p, bpb, bound) in violations:
                f.write(f"{alg},{qlab},p={p:.4f},bpb={bpb:.6f},bound={bound:.6f}\n")

    return out_png

# --- Make plots ---
outs = []
for rep in ["ascii01", "bitpack"]:
    for order in ["random", "alternating", "blocks_zeros_then_ones"]:
        out = plot_one(rep, order)
        if out:
            outs.append(out)

print("Wrote plots to:")
for p in outs:
    print(" -", p)
print("Note:")
print(" • Theory curves are only overlaid for 'random' (i.i.d. Bernoulli) data.")
print(" • 'alternating' and 'blocks' are *not* i.i.d.; their entropy rate can be far below H(p),")
print("   so measured bpb legitimately falls below the binary-entropy curve there.")
