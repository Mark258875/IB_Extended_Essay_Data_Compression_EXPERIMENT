#!/usr/bin/env python3
"""
Sweep LZ4 compression levels over input files and record results to the
CSV path defined in conf/config.yaml (shared with Brotli).

Usage:
  python sweep_lz4.py

Notes:
- No CLI arguments; everything comes from conf/config.yaml via Config.
- Writes rows with alg="lz4" so they can live in the same CSV as Brotli.
- Requires python-lz4 installed.
"""
from pathlib import Path
import glob
import json
import time
import platform
from copy import deepcopy

from config import Config
import bench_compress as bench


def expand_files(patterns):
    return sorted(set(sum((glob.glob(pat) for pat in patterns), [])))


def main():
    cfg = Config.load("conf/config.yaml")
    base_args = cfg.to_bench_args()
    out_csv = cfg.output_csv_path_for("lz4")

    files = expand_files(cfg.paths.input_globs)
    if not files:
        print("No input files matched. Edit conf/config.yaml -> paths.input_globs")
        return

    rows_all = []
    levels = list(range(0, 17))  # 0..16 inclusive

    for lvl in levels:
        args = deepcopy(base_args)
        args.lz4_level = lvl
        print(f"\n=== Sweeping LZ4 level={lvl} over {len(files)} file(s) ===")
        rows = []
        for f in files:
            data = Path(f).read_bytes()
            row = bench.run_one(f, data, "lz4", args)
            rows.append(row)
            print(f"{f} | L={lvl} | ratio={row['ratio']:.4f} "
                  f"comp={row['comp_MB_s']:.1f} MB/s ({row['comp_MiB_s']:.1f} MiB/s) "
                  f"decomp={row['decomp_MB_s']:.1f} MB/s ({row['decomp_MiB_s']:.1f} MiB/s)")
        # append per-level to avoid large memory usage
        bench.write_csv(rows, out_csv, append=True)
        rows_all.extend(rows)

    # Optional: write meta once for reproducibility
    meta = {
        "timestamp": time.time(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "input_globs": cfg.paths.input_globs,
        "output_csv": str(out_csv),
        "sweep": {
            "lz4_levels": levels,
        },
        "repeats": cfg.timing.repeats,
    }
    try:
        import lz4.frame as _lz4
        meta["lz4_version"] = getattr(_lz4, "__version__", "unknown")
    except Exception:
        try:
            import lz4 as _lz4
            meta["lz4_version"] = getattr(_lz4, "__version__", "unknown")
        except Exception:
            meta["lz4_version"] = None

    meta_path = out_csv.with_suffix(".lz4.meta.json")
    # merge if exists
    try:
        if meta_path.exists():
            old = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            old = {}
    except Exception:
        old = {}
    old.update(meta)
    meta_path.write_text(json.dumps(old, indent=2), encoding="utf-8")

    print(f"\nDone. Appended {len(rows_all)} row(s) to {out_csv}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
