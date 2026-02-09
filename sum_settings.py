#!/usr/bin/env python3
from __future__ import annotations
import csv, json, statistics, math, argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# =========================
# ===== helper funcs  =====
# =========================

def try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def parse_params(params_json: str) -> Dict[str, Any]:
    try:
        return json.loads(params_json)
    except Exception:
        return {}

def extract_setting_label(params: Dict[str, Any]) -> str:
    """
    Return a compact setting label like 'quality=6' or 'level=12'.
    Falls back to compact JSON or '(default)'.
    """
    for k in ("quality", "level", "compression_level", "preset", "acceleration"):
        if k in params:
            return f"{k}={params[k]}"
    return json.dumps(params, sort_keys=True) if params else "(default)"

def iqr(values: List[float]) -> float:
    if not values:
        return float("nan")
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return qs[2] - qs[0]  # Q3 - Q1

def numeric_key_from_setting(setting: str) -> Tuple[int, str]:
    """
    Best-effort to sort settings numerically by the number inside 'quality=6',
    'level=12', etc. Falls back to 9999 if no integer found.
    """
    import re
    m = re.search(r"(-?\d+)", setting)
    if m:
        try:
            return (int(m.group(1)), setting)
        except Exception:
            pass
    return (9999, setting)


# =========================
# ===== core loading  =====
# =========================

def read_all_rows(results_root: Path) -> List[Dict[str, Any]]:
    """
    Read all relevant CSVs:
      - random / seq / kary
      - standard_corpora (Calgary, Canterbury, Silesia)
    and return per-row measurements needed for H7:
      (alg, setting, gap, comp_MB_s, decomp_MB_s)
    """
    csv_paths: List[Path] = [
        # Binary random
        results_root / "random" / "results_brotli.csv",
        results_root / "random" / "results_lz4.csv",

        # Binary sequential
        results_root / "seq" / "results_brotli.csv",
        results_root / "seq" / "results_lz4.csv",

        # k-ary synthetic distributions
        results_root / "kary" / "results_brotli.csv",
        results_root / "kary" / "results_lz4.csv",

        # Web corpora: separate CSVs per corpus
        results_root / "standard_corpora" / "results_brotli_calgary.csv",
        results_root / "standard_corpora" / "results_lz4_calgary.csv",
        results_root / "standard_corpora" / "results_brotli_canterbury.csv",
        results_root / "standard_corpora" / "results_lz4_canterbury.csv",
        results_root / "standard_corpora" / "results_brotli_silesia.csv",
        results_root / "standard_corpora" / "results_lz4_silesia.csv",
    ]

    rows: List[Dict[str, Any]] = []
    files_loaded = 0
    for p in csv_paths:
        if not p.exists():
            print(f"[warn] CSV not found, skipping: {p}")
            continue
        files_loaded += 1
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                alg = (row.get("alg") or "").lower()
                if alg not in {"brotli", "lz4"}:
                    continue

                bpb = try_float(row.get("bpb"))
                H   = try_float(row.get("entropy_bpb"))
                if bpb is None or H is None:
                    # need both to define entropy gap
                    continue
                gap = bpb - H

                comp = try_float(row.get("comp_MB_s"))
                decomp = try_float(row.get("decomp_MB_s"))

                params = parse_params(row.get("params") or "{}")
                setting = extract_setting_label(params)

                rows.append({
                    "alg": alg,
                    "setting": setting,
                    "gap": gap,
                    "comp_MB_s": comp,
                    "decomp_MB_s": decomp,
                })

    if files_loaded == 0:
        print("[err] No CSVs found under", results_root)
    else:
        print(f"[info] Loaded from {files_loaded} CSV file(s), {len(rows)} usable rows.")

    return rows


# =========================
# ===== summarization  ====
# =========================

def summarize_per_setting(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by (Algorithm, Setting) across ALL experiments and datasets.
    For each group compute:
      - N (rows)
      - Median entropy gap (bpb - H)
      - IQR of gap
      - Median compression speed (MB/s)
      - Median decompression speed (MB/s)
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["alg"], r["setting"])
        groups[key].append(r)

    summary: List[Dict[str, Any]] = []
    for (alg, setting), items in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], numeric_key_from_setting(kv[0][1]))
    ):
        gaps   = [x["gap"] for x in items if x["gap"] is not None]
        comps  = [x["comp_MB_s"] for x in items if x["comp_MB_s"] is not None]
        decomps= [x["decomp_MB_s"] for x in items if x["decomp_MB_s"] is not None]

        if not gaps:
            continue

        gap_med = statistics.median(gaps)
        gap_iqr = iqr(gaps)

        comp_med = statistics.median(comps) if comps else float("nan")
        decomp_med = statistics.median(decomps) if decomps else float("nan")

        summary.append({
            "Algorithm": alg.upper(),
            "Setting": setting,
            "N_rows": len(items),
            "Gap_median_bpb_minus_H": gap_med,
            "Gap_IQR": gap_iqr,
            "Comp_MB_s_median": comp_med,
            "Decomp_MB_s_median": decomp_med,
        })
    return summary


# =========================
# ===== output helpers ====
# =========================

def write_csv(summary: List[Dict[str, Any]], out_path: Path) -> None:
    if not summary:
        print("[warn] No summary rows to write.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(summary[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"[OK] wrote per-setting summary -> {out_path}")

def print_table(summary: List[Dict[str, Any]]) -> None:
    if not summary:
        print("No per-setting summary to display.")
        return
    cols = [
        "Algorithm",
        "Setting",
        "N_rows",
        "Gap_median_bpb_minus_H",
        "Gap_IQR",
        "Comp_MB_s_median",
        "Decomp_MB_s_median",
    ]
    # compute column widths
    widths = {c: len(c) for c in cols}
    for row in summary:
        for c in cols:
            widths[c] = max(widths[c], len(str(row[c])))

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in summary:
        print(" | ".join(str(row[c]).ljust(widths[c]) for c in cols))


def to_latex(summary: List[Dict[str, Any]]) -> str:
    if not summary:
        return ""
    header = r"""\begin{table}[h]
\centering
\caption{Per-setting summary across all experiments: median entropy gap and speeds.}
\label{tab:per_setting_h7}
\begin{tabular}{l l r r r r r}
\toprule
Algorithm & Setting & $N$ & Gap (bpb$-H$) & Gap IQR & Comp (MB/s) & Decomp (MB/s) \\
\midrule
"""
    lines = []
    for r in summary:
        lines.append(
            f"{r['Algorithm']} & {r['Setting']} & {r['N_rows']} & "
            f"{r['Gap_median_bpb_minus_H']:.4f} & {r['Gap_IQR']:.4f} & "
            f"{(r['Comp_MB_s_median'] or float('nan')):.1f} & "
            f"{(r['Decomp_MB_s_median'] or float('nan')):.1f} \\\\"
        )
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return header + "\n".join(lines) + footer


# =========================
# ========  CLI  =========
# =========================

def main():
    ap = argparse.ArgumentParser(
        description="Summarize per-setting behaviour (for H7) across all experiments."
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory holding random/seq/kary/standard_corpora (default: ./results)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("summaries") / "per_setting_summary.csv",
        help="Output CSV for per-setting summary.",
    )
    ap.add_argument(
        "--latex",
        action="store_true",
        help="Also print LaTeX table to stdout.",
    )
    args = ap.parse_args()

    rows = read_all_rows(args.results_root)
    if not rows:
        print("No rows loaded; nothing to summarize.")
        return

    summary = summarize_per_setting(rows)
    print_table(summary)
    write_csv(summary, args.out_csv)

    if args.latex:
        print("\n----- LaTeX per-setting table (H7) -----")
        print(to_latex(summary))


if __name__ == "__main__":
    main()
