from __future__ import annotations

import argparse
import glob
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import bench_compress as bench
from config import Config
from runtime_tuning import apply_runtime


def expand_inputs(paths: list[str]) -> list[str]:
    """Accept literal files or glob patterns; return existing files only."""
    out = set()
    for p in paths:
        # Treat as glob first (supports both literal and pattern)
        matches = glob.glob(p, recursive=True)
        if not matches:
            matches = [p]
        for m in matches:
            mp = Path(m)
            if mp.is_file():
                out.add(str(mp))
    return sorted(out)


def write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        old = {}
    old.update(meta)
    meta_path.write_text(json.dumps(old, indent=2), encoding="utf-8")


def run_alg(alg: str, files: list[str], args_ns: SimpleNamespace) -> list[dict]:
    rows = []
    total_files = len(files)
    for i, f in enumerate(files, 1):
        print(f"[{alg}] Processing file {i}/{total_files}: {f}")
        data = Path(f).read_bytes()  # I/O not timed
        start_time = time.time()
        try:
            row = bench.run_one(f, data, alg, args_ns)
            rows.append(row)
            elapsed = time.time() - start_time
            ratio = row.get("ratio", float("nan"))
            bpb = row.get("bpb", float("nan"))
            H = row.get("entropy_bpb", float("nan"))
            comp = row.get("comp_MB_s", float("nan"))
            decomp = row.get("decomp_MB_s", float("nan"))
            print(f"  Done in {elapsed:.1f}s | ratio={ratio:.4f} bpb={bpb:.3f} H_bpb={H:.3f} "
                  f"comp={comp:.1f} MB/s decomp={decomp:.1f} MB/s")
        except Exception as e:
            print(f"[{alg}] ERROR on {f}: {e}")
    return rows



def main():
    # --- CLI: only positional inputs ---
    ap = argparse.ArgumentParser(description="Compress files with Brotli & LZ4 and save to Results.csv")
    ap.add_argument("inputs", nargs="+", help="Files or glob patterns (quote globs on shell)")
    args = ap.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        print("No input files found.")
        return

    # --- Load config & standardize runtime ---
    cfg = Config.load("conf/config.yaml")
    apply_runtime(cfg.runtime)

    # Build bench args (repeats/warmup & alg params come from config)
    bench_args = cfg.to_bench_args()

    # --- Output paths (single CSV + single meta JSON) ---
    out_dir = Path(cfg.paths.output_dir or "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "Results.csv"
    meta_path = out_dir / "Results.meta.json"

    # --- Run both algorithms over the files ---
    rows_all = []
    rows_all += run_alg("brotli", files, bench_args)
    rows_all += run_alg("lz4",    files, bench_args)

    if not rows_all:
        print("No results to write.")
        return

    # Append to a single CSV (create with header if missing)
    bench.write_csv(rows_all, out_csv, append=True)
    print(f"\nWrote {len(rows_all)} row(s) to {out_csv}")

    # Meta (reproducibility)
    meta = {
        "timestamp": time.time(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "files": files,
        "output_csv": str(out_csv),
        "repeats": cfg.timing.repeats,
        "warmup": cfg.timing.warmup,
        "runtime": {
            "cpu_affinity": cfg.runtime.cpu_affinity,
            "priority": cfg.runtime.priority,
        },
        "brotli": {
            "quality": cfg.brotli.quality,
            "mode": cfg.brotli.mode,
            "lgwin": cfg.brotli.lgwin,
        },
        "lz4": {
            "level": cfg.lz4.compression_level,
        },
    }
    try:
        import brotli as _b
        meta["brotli_version"] = getattr(_b, "__version__", "unknown")
    except Exception:
        meta["brotli_version"] = None
    try:
        import lz4
        meta["lz4_version"] = getattr(lz4, "__version__", "unknown")
    except Exception:
        meta["lz4_version"] = None

    write_meta(meta_path, meta)
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
