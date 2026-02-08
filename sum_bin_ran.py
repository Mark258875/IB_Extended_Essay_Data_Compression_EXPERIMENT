#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

# ---------- parsing helpers ----------
def try_float(x: Any):
    try:
        return float(x)
    except Exception:
        return None

def parse_params(params_json: str) -> Dict[str, Any]:
    try:
        return json.loads(params_json)
    except Exception:
        return {}

def extract_quality_label(params: Dict[str, Any]) -> str:
    """Return a compact setting label like 'quality=6' or 'level=12'."""
    for k in ("quality", "level", "compression_level", "preset", "acceleration"):
        if k in params:
            return f"{k}={params[k]}"
    return json.dumps(params, sort_keys=True) if params else "(default)"

_P_TAG = re.compile(r"(p\d+(?:\.\d+)?)", re.IGNORECASE)

def extract_distribution(dataset: str) -> str:
    """
    Build a distribution key from the dataset path/filename.
    Examples: 'p0.01_random', 'p0.05_alternating', 'p0.01_blocks', etc.
    If no order token is found, just returns 'pX'.
    """
    s = (dataset or "").lower().replace("\\", "/")
    m = _P_TAG.search(s)
    prob = m.group(1) if m else "p?"
    if "random" in s:
        return f"{prob}_random"
    if "alternating" in s:
        return f"{prob}_alternating"
    if "blocks" in s:
        return f"{prob}_blocks"
    if "sequential" in s:
        return f"{prob}_sequential"
    return prob

def read_binary_rows(csv_paths: List[Path], debug: bool = False) -> List[Dict[str, Any]]:
    """
    Load ALL rows from the given CSVs and treat them as part of the binary experiment.
    (We assume you pass in the correct CSVs, e.g. random/results_*.csv, seq/results_*.csv.)
    """
    rows: List[Dict[str, Any]] = []
    files_loaded = 0
    rows_seen = 0

    for p in csv_paths:
        if not p:
            continue
        if not p.exists():
            if debug:
                print(f"[warn] CSV not found: {p}")
            continue
        files_loaded += 1
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                rows_seen += 1
                ds = row.get("dataset") or ""
                alg = (row.get("alg") or "").lower()

                if debug and i < 2:
                    print(f"[peek {p.name}] alg={alg} dataset={ds}")

                params = parse_params(row.get("params") or "{}")
                bpb = try_float(row.get("bpb"))
                H   = try_float(row.get("entropy_bpb"))
                comp_mb_s = try_float(row.get("comp_MB_s"))

                rows.append({
                    "dataset": ds,
                    "distribution": extract_distribution(ds),
                    "alg": alg.upper(),  # normalize for output
                    "bpb": bpb,
                    "entropy_bpb": H,
                    "comp_MB_s": comp_mb_s,
                    "quality": extract_quality_label(params),
                })

    if debug:
        print(f"[stats] files_loaded={files_loaded}, rows_seen={rows_seen}, returned={len(rows)}")
        if files_loaded == 0:
            print("[hint] Check the --brotli-csv / --lz4-csv paths you passed.")

    return rows

def fmt_pct_overH(bpb: float | None, H: float | None) -> str:
    if bpb is None or H is None or H <= 0:
        return "n/a"
    pct = (bpb - H) / H * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"

# ---------- summarization ----------
def summarize_best_worst(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    For each (Algorithm x Distribution), pick:
      - Best/Worst ratio by bpb (lower is better), with % over entropy shown.
      - Best/Worst speed by comp_MB_s (higher is better).
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["alg"], r["distribution"])].append(r)

    summary: List[Dict[str, str]] = []
    for (alg, dist), items in sorted(groups.items()):
        items_bpb = [x for x in items if x["bpb"] is not None and x["entropy_bpb"] is not None]
        items_spd = [x for x in items if x["comp_MB_s"] is not None]

        if not items_bpb or not items_spd:
            continue

        best_ratio = min(items_bpb, key=lambda x: x["bpb"])
        worst_ratio = max(items_bpb, key=lambda x: x["bpb"])
        best_speed = max(items_spd, key=lambda x: x["comp_MB_s"])
        worst_speed = min(items_spd, key=lambda x: x["comp_MB_s"])

        summary.append({
            "Algorithm": alg,
            "Distribution": dist,
            "Best_ratio": f"{best_ratio['quality']} ({best_ratio['bpb']:.3f} bpb)",
            "Best_ratio_overH": fmt_pct_overH(best_ratio["bpb"], best_ratio["entropy_bpb"]),
            "Worst_ratio": f"{worst_ratio['quality']} ({worst_ratio['bpb']:.3f} bpb)",
            "Worst_ratio_overH": fmt_pct_overH(worst_ratio["bpb"], worst_ratio["entropy_bpb"]),
            "Best_speed": f"{best_speed['quality']} ({best_speed['comp_MB_s']:.1f} MB/s)",
            "Worst_speed": f"{worst_speed['quality']} ({worst_speed['comp_MB_s']:.1f} MB/s)",
        })
    return summary

# ---------- output helpers ----------
def print_table(summary: List[Dict[str, str]]) -> None:
    if not summary:
        print("No binary summary to display.")
        return
    cols = [
        "Algorithm", "Distribution",
        "Best_ratio", "Best_ratio_overH",
        "Worst_ratio", "Worst_ratio_overH",
        "Best_speed", "Worst_speed"
    ]
    widths = {c: max(len(c), max(len(str(r[c])) for r in summary)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in summary:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

def write_csv(summary: List[Dict[str, str]], out_path: Path) -> None:
    if not summary:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[OK] wrote {out_path}")

def to_latex(summary: List[Dict[str, str]]) -> str:
    if not summary:
        return ""
    header = r"""\begin{table}[h]
\centering
\caption{Binary experiment: best/worst per distribution (ratio, \% over entropy, and speed).}
\label{tab:binary_best_worst}
\begin{tabular}{l l l l l l l l}
\toprule
Algorithm & Distribution & Best ratio & Over $H$ & Worst ratio & Over $H$ & Best speed & Worst speed \\ 
\midrule
"""
    rows = []
    for r in summary:
        rows.append(f"{r['Algorithm']} & {r['Distribution']} & {r['Best_ratio']} & "
                    f"{r['Best_ratio_overH']} & {r['Worst_ratio']} & {r['Worst_ratio_overH']} & "
                    f"{r['Best_speed']} & {r['Worst_speed']} \\\\")
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return header + "\n".join(rows) + footer

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Summarize Binary experiment: best/worst ratio (with % over entropy) and speed per distribution.")
    ap.add_argument("--brotli-csv", type=Path, default=Path("results") / "results_brotli.csv",
                    help="Path to Brotli CSV (default: results/results_brotli.csv)")
    ap.add_argument("--lz4-csv", type=Path, default=Path("results") / "results_lz4.csv",
                    help="Path to LZ4 CSV (default: results/results_lz4.csv)")
    ap.add_argument("--out-csv", type=Path, default=Path("summaries") / "binary_best_worst.csv",
                    help="Optional output CSV path")
    ap.add_argument("--latex", action="store_true", help="Print LaTeX table to stdout")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    rows = read_binary_rows([args.brotli_csv, args.lz4_csv], debug=args.debug)
    if not rows:
        print("No rows loaded. Check CSV paths or content.")
        return

    summary = summarize_best_worst(rows)
    print_table(summary)
    if args.out_csv:
        write_csv(summary, args.out_csv)
    if args.latex:
        print("\n----- LaTeX -----")
        print(to_latex(summary))

if __name__ == "__main__":
    main()
