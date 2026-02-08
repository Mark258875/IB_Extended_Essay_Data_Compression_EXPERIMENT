#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

# --------- robust matcher for "standard corpora" ----------
STD_PAT = re.compile(r"[\\/](standard_corpora|corpora)[\\/]", re.IGNORECASE)

def is_webcorpus_row(dataset: str) -> bool:
    s = (dataset or "").lower()
    s = s.replace("\\", "/")
    return bool(STD_PAT.search(s))

# --------- parsing helpers ----------
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
    for k in ("quality", "level", "compression_level", "preset", "acceleration"):
        if k in params:
            return f"{k}={params[k]}"
    return json.dumps(params, sort_keys=True) if params else "(default)"

def split_corpus_and_doc(dataset: str) -> Tuple[str, str]:
    """
    dataset like:
      data/standard_corpora/Calgary corpus/book1
    -> corpus = 'Calgary corpus', doc = 'book1'
    If we can't parse, fall back to ('(unknown)', basename).
    """
    s = (dataset or "").replace("\\", "/")
    parts = s.split("/")
    try:
        idx = None
        # find 'standard_corpora' or 'corpora'
        for i, p in enumerate(parts):
            if p.lower() in {"standard_corpora", "corpora"}:
                idx = i
                break
        if idx is not None and idx + 2 < len(parts):
            corpus = parts[idx + 1]
            doc = parts[idx + 2]
            return corpus, doc
    except Exception:
        pass
    # fallback
    return "(unknown)", parts[-1] if parts else "(unknown)"

def fmt_pct_overH(bpb: float | None, H: float | None) -> str:
    if bpb is None or H is None or H <= 0:
        return "n/a"
    pct = (bpb - H) / H * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"

# --------- load rows ----------
def read_web_rows(csv_paths: List[Path], debug: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files_loaded = 0
    rows_seen = 0
    matched = 0

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
                alg = (row.get("alg") or "").upper()

                if debug and i < 2:
                    print(f"[peek {p.name}] alg={alg} dataset={ds}")

                if not is_webcorpus_row(ds):
                    continue
                matched += 1

                params = parse_params(row.get("params") or "{}")
                bpb = try_float(row.get("bpb"))
                H   = try_float(row.get("entropy_bpb"))
                comp_mb_s = try_float(row.get("comp_MB_s"))

                corpus, doc = split_corpus_and_doc(ds)

                rows.append({
                    "dataset": ds,
                    "corpus": corpus,
                    "doc": doc,
                    "alg": alg,
                    "bpb": bpb,
                    "entropy_bpb": H,
                    "comp_MB_s": comp_mb_s,
                    "quality": extract_quality_label(params),
                })

    if debug:
        print(f"[stats] files_loaded={files_loaded}, rows_seen={rows_seen}, "
              f"matched_webcorpora={matched}, returned={len(rows)}")
        if files_loaded == 0:
            print("[hint] Pass explicit CSV paths via --brotli-csv / --lz4-csv.")

    return rows

# --------- summary ----------
def summarize_best_worst(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    For each (Algorithm × Corpus × Document), pick:
      - Best/Worst ratio by bpb (lower is better), plus % over entropy for each.
      - Best/Worst speed by comp_MB_s (higher is better).
    """
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["alg"], r["corpus"], r["doc"])].append(r)

    summary: List[Dict[str, str]] = []
    for (alg, corpus, doc), items in sorted(groups.items()):
        items_bpb = [x for x in items if x["bpb"] is not None and x["entropy_bpb"] is not None]
        items_spd = [x for x in items if x["comp_MB_s"] is not None]

        if not items_bpb and not items_spd:
            continue

        best_ratio_txt = worst_ratio_txt = best_ratio_overH = worst_ratio_overH = "n/a"
        if items_bpb:
            best_ratio = min(items_bpb, key=lambda x: x["bpb"])
            worst_ratio = max(items_bpb, key=lambda x: x["bpb"])
            best_ratio_txt = f"{best_ratio['quality']} ({best_ratio['bpb']:.3f} bpb)"
            worst_ratio_txt = f"{worst_ratio['quality']} ({worst_ratio['bpb']:.3f} bpb)"
            best_ratio_overH = fmt_pct_overH(best_ratio["bpb"], best_ratio["entropy_bpb"])
            worst_ratio_overH = fmt_pct_overH(worst_ratio["bpb"], worst_ratio["entropy_bpb"])

        best_speed_txt = worst_speed_txt = "n/a"
        if items_spd:
            best_speed = max(items_spd, key=lambda x: x["comp_MB_s"])
            worst_speed = min(items_spd, key=lambda x: x["comp_MB_s"])
            best_speed_txt = f"{best_speed['quality']} ({best_speed['comp_MB_s']:.1f} MB/s)"
            worst_speed_txt = f"{worst_speed['quality']} ({worst_speed['comp_MB_s']:.1f} MB/s)"

        summary.append({
            "Algorithm": alg,
            "Corpus": corpus,
            "Document": doc,
            "Best_ratio": best_ratio_txt,
            "Best_ratio_overH": best_ratio_overH,
            "Worst_ratio": worst_ratio_txt,
            "Worst_ratio_overH": worst_ratio_overH,
            "Best_speed": best_speed_txt,
            "Worst_speed": worst_speed_txt,
        })
    return summary

# --------- output ----------
def print_table(summary: List[Dict[str, str]]) -> None:
    if not summary:
        print("No web-corpora summary to display.")
        return
    cols = [
        "Algorithm", "Corpus", "Document",
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
\caption{Web corpora (e.g., Calgary): best/worst per document (ratio, \% over entropy, and speed).}
\label{tab:webcorpora_best_worst}
\begin{tabular}{l l l l l l l l}
\toprule
Algorithm & Corpus & Document & Best ratio & Over $H$ & Worst ratio & Over $H$ & Best speed \\
\midrule
"""
    rows = []
    for r in summary:
        rows.append(f"{r['Algorithm']} & {r['Corpus']} & {r['Document']} & "
                    f"{r['Best_ratio']} & {r['Best_ratio_overH']} & "
                    f"{r['Worst_ratio']} & {r['Worst_ratio_overH']} & "
                    f"{r['Best_speed']} \\\\")
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return header + "\n".join(rows) + footer

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Summarize web corpora (standard corpora): best/worst ratio (with % over entropy) and speed per document."
    )
    ap.add_argument("--brotli-csv", type=Path, default=Path("results") / "results_brotli.csv",
                    help="Path to Brotli CSV (default: results/results_brotli.csv)")
    ap.add_argument("--lz4-csv", type=Path, default=Path("results") / "results_lz4.csv",
                    help="Path to LZ4 CSV (default: results/results_lz4.csv)")
    ap.add_argument("--out-csv", type=Path, default=Path("summaries") / "webcorpora_best_worst.csv",
                    help="Optional output CSV path")
    ap.add_argument("--latex", action="store_true", help="Print LaTeX table to stdout")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    rows = read_web_rows([args.brotli_csv, args.lz4_csv], debug=args.debug)
    if not rows:
        print("No web-corpora rows loaded.")
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
