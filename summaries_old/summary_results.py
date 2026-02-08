#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, statistics, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# ---------- helpers ----------
def try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def base_name(p: str) -> str:
    p = p.replace("\\", "/")
    return p.rsplit("/", 1)[-1]

def parse_params(params_json: str) -> Dict[str, Any]:
    try:
        return json.loads(params_json)
    except Exception:
        return {}

def extract_setting_label(alg: str, params: Dict[str, Any]) -> str:
    # tries common keys
    for k in ("quality","level","compression_level","preset","acceleration"):
        if k in params:
            return f"{k}={params[k]}"
    # fallback: compact JSON
    return json.dumps(params, sort_keys=True) if params else "(default)"

def detect_experiment(dataset_path: str) -> str:
    s = dataset_path.lower().replace("\\", "/")
    # Binary experiment: random vs sequential
    if "/binary/" in s:
        if "random" in s:
            return "Binary (IID)"
        if "alternating" in s or "blocks" in s or "sequential" in s:
            return "Binary (Sequential)"
        return "Binary (Other)"
    # Web corpora
    if "standard_corpora" in s or "corpora" in s or "corpus" in s:
        return "Web corpora"
    # k-ary / entropy sweep (adjust to your folder names if needed)
    if "kary" in s or "k-ary" in s or "entropy" in s:
        return "k-ary (entropy sweep)"
    return "Other"

def iqr(values: List[float]) -> float:
    if not values:
        return float("nan")
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return qs[2] - qs[0]  # Q3 - Q1

# ---------- load ----------
def read_rows(csv_paths: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in csv_paths:
        if not p.exists():
            continue
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                alg = (row.get("alg") or "").lower()
                if alg not in {"brotli","lz4"}:
                    continue
                ds = row.get("dataset") or ""
                # core numeric fields
                bpb = try_float(row.get("bpb"))
                H   = try_float(row.get("entropy_bpb"))
                comp = try_float(row.get("comp_MB_s"))
                decomp = try_float(row.get("decomp_MB_s"))
                if bpb is None or H is None:
                    # skip rows lacking required fields
                    continue
                gap = bpb - H
                params = parse_params(row.get("params") or "{}")
                out.append({
                    "dataset": ds,
                    "dataset_base": base_name(ds),
                    "experiment": detect_experiment(ds),
                    "alg": alg,
                    "gap": gap,
                    "bpb": bpb,
                    "H": H,
                    "comp_MB_s": comp,
                    "decomp_MB_s": decomp,
                    "params": params,
                    "setting": extract_setting_label(alg, params),
                })
    return out

# ---------- summaries ----------
def summarize_per_experiment(rows: List[Dict[str,Any]]):
    # group -> (experiment, alg)
    groups: Dict[Tuple[str,str], List[Dict[str,Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["alg"])].append(r)

    tableA = []  # experiment, alg, N, gap_med, gap_iqr, comp_med, decomp_med, best_setting
    for (exp, alg), items in sorted(groups.items()):
        if not items:
            continue
        gaps   = [x["gap"] for x in items if x["gap"] is not None]
        comps  = [x["comp_MB_s"] for x in items if x["comp_MB_s"] is not None]
        decomps= [x["decomp_MB_s"] for x in items if x["decomp_MB_s"] is not None]
        N = len(items)

        gap_med = statistics.median(gaps) if gaps else float("nan")
        gap_iqr = iqr(gaps) if gaps else float("nan")
        comp_med = statistics.median(comps) if comps else float("nan")
        decomp_med = statistics.median(decomps) if decomps else float("nan")

        # best setting by lowest median gap (break ties by higher median speed)
        by_setting: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
        for it in items: by_setting[it["setting"]].append(it)

        def setting_score(recs: List[Dict[str,Any]]):
            g = [x["gap"] for x in recs if x["gap"] is not None]
            c = [x["comp_MB_s"] for x in recs if x["comp_MB_s"] is not None]
            g_med = statistics.median(g) if g else float("inf")
            c_med = statistics.median(c) if c else 0.0
            return (g_med, -c_med)  # sort by gap asc, then speed desc

        best_setting = min(by_setting.items(), key=lambda kv: setting_score(kv[1]))[0] if by_setting else "(n/a)"

        tableA.append({
            "Experiment": exp,
            "Algorithm": alg.upper(),
            "N": N,
            "Gap_median": gap_med,
            "Gap_IQR": gap_iqr,
            "Comp_MB_s_median": comp_med,
            "Decomp_MB_s_median": decomp_med,
            "Best_setting": best_setting
        })
    return tableA

def paired_brotli_vs_lz4(rows: List[Dict[str,Any]]):
    # Pair on (experiment, dataset_base). For each pair, compute diffs (LZ4 - Brotli).
    # If multiple settings per alg exist for same dataset, take the *best setting per alg for that dataset* by lowest gap.
    # Then aggregate median diffs.
    # Step 1: group by (experiment, dataset_base, alg, setting)
    per_key: Dict[Tuple[str,str,str,str], List[Dict[str,Any]]] = defaultdict(list)
    for r in rows:
        per_key[(r["experiment"], r["dataset_base"], r["alg"], r["setting"])].append(r)

    # Step 2: reduce to best setting per (experiment, dataset_base, alg)
    best_per_alg: Dict[Tuple[str,str,str], Dict[str,Any]] = {}
    for (exp, db, alg, setting), recs in per_key.items():
        # pick setting with lowest median gap (tie -> faster)
        g = [x["gap"] for x in recs if x["gap"] is not None]
        c = [x["comp_MB_s"] for x in recs if x["comp_MB_s"] is not None]
        if not g:
            continue
        g_med = statistics.median(g)
        c_med = statistics.median(c) if c else 0.0
        key = (exp, db, alg)
        prev = best_per_alg.get(key)
        if prev is None or (g_med < prev["g_med"] or (math.isclose(g_med, prev["g_med"]) and c_med > prev["c_med"])):
            best_per_alg[key] = {"setting": setting, "g_med": g_med, "c_med": c_med,
                                 "gap_samples": g, "comp_samples": c}

    # Step 3: build paired lists per experiment
    pairs: Dict[str, List[Tuple[float,float]]] = defaultdict(list)  # exp -> [(gap_diff, comp_diff), ...]
    for exp_db_alg, rec in list(best_per_alg.items()):
        exp, db, alg = exp_db_alg
        other_alg = "lz4" if alg == "brotli" else ("brotli" if alg == "lz4" else None)
        if other_alg is None:
            continue
        a = best_per_alg.get((exp, db, "brotli"))
        b = best_per_alg.get((exp, db, "lz4"))
        if a and b:
            gap_diff  = b["g_med"] - a["g_med"]  # positive => LZ4 has larger gap
            comp_diff = statistics.median(b["comp_samples"]) - statistics.median(a["comp_samples"])
            pairs[exp].append((gap_diff, comp_diff))

    # Step 4: aggregate per experiment
    tableB = []
    for exp, diffs in sorted(pairs.items()):
        if not diffs:
            continue
        gap_diffs  = [d[0] for d in diffs]
        comp_diffs = [d[1] for d in diffs]
        row = {
            "Experiment": exp,
            "Paired_N": len(diffs),
            "Gap_diff_median_(LZ4-Brotli)": statistics.median(gap_diffs),
            "Comp_MB_s_diff_median_(LZ4-Brotli)": statistics.median(comp_diffs),
            "Direction_gap": "Brotli better" if statistics.median(gap_diffs) > 0 else "LZ4 better or tie",
            "Direction_speed": "LZ4 faster" if statistics.median(comp_diffs) > 0 else "Brotli faster or tie",
        }
        tableB.append(row)
    return tableB

# ---------- output ----------
def to_csv(rows: List[Dict[str,Any]], path: Path) -> None:
    if not rows: return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def to_latex_tableA(rows: List[Dict[str,Any]]) -> str:
    if not rows: return ""
    header = r"""\begin{table}[h]
\centering
\caption{Per-experiment summary (median [IQR]).}
\label{tab:summary}
\begin{tabular}{l l r r r r l}
\toprule
Experiment & Algorithm & $N$ & Gap (bpb$-H$) & Comp (MB/s) & Decomp (MB/s) & Best setting\\
\midrule
"""
    lines = []
    for r in rows:
        lines.append(f"{r['Experiment']} & {r['Algorithm']} & {r['N']} & "
                     f"{r['Gap_median']:.4f} [{r['Gap_IQR']:.4f}] & "
                     f"{(r['Comp_MB_s_median'] or float('nan')):.1f} & "
                     f"{(r['Decomp_MB_s_median'] or float('nan')):.1f} & "
                     f"{r['Best_setting']} \\\\")
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return header + "\n".join(lines) + footer

def to_latex_tableB(rows: List[Dict[str,Any]]) -> str:
    if not rows: return ""
    header = r"""\begin{table}[h]
\centering
\caption{Paired Brotli vs LZ4 (best-per-dataset settings). Positive diffs mean LZ4 $>$ Brotli.}
\label{tab:paired}
\begin{tabular}{l r r r l l}
\toprule
Experiment & Paired $N$ & Median $\Delta$Gap & Median $\Delta$Comp MB/s & Gap Direction & Speed Direction\\
\midrule
"""
    lines = []
    for r in rows:
        lines.append(f"{r['Experiment']} & {r['Paired_N']} & "
                     f"{r['Gap_diff_median_(LZ4-Brotli)']:.4f} & "
                     f"{r['Comp_MB_s_diff_median_(LZ4-Brotli)']:.1f} & "
                     f"{r['Direction_gap']} & {r['Direction_speed']} \\\\")
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return header + "\n".join(lines) + footer

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Summarize compression results into compact tables.")
    ap.add_argument("--brotli-csv", type=Path, default=Path("results") / "results_brotli.csv")
    ap.add_argument("--lz4-csv", type=Path, default=Path("results") / "results_lz4.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("summaries"))
    args = ap.parse_args()

    rows = read_rows([args.brotli_csv, args.lz4_csv])
    if not rows:
        print("No rows loaded. Check CSV paths.")
        return

    tableA = summarize_per_experiment(rows)
    tableB = paired_brotli_vs_lz4(rows)

    outA = args.out_dir / "summary_per_experiment.csv"
    outB = args.out_dir / "paired_brotli_vs_lz4.csv"
    to_csv(tableA, outA)
    to_csv(tableB, outB)

    print(f"[OK] Wrote: {outA}")
    print(f"[OK] Wrote: {outB}")

    # Also print LaTeX snippets
    print("\n----- LaTeX: Table A -----")
    print(to_latex_tableA(tableA))
    print("\n----- LaTeX: Table B -----")
    print(to_latex_tableB(tableB))

if __name__ == "__main__":
    main()
