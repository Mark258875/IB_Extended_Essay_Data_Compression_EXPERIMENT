#!/usr/bin/env python3
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
    out_csv = cfg.output_csv_path()

    files = expand_files(cfg.paths.input_globs)
    if not files:
        print("No input files matched. Edit conf/config.yaml -> paths.input_globs")
        return

    rows_all = []
    qualities = list(range(0, 12))  # 0..11 inclusive

    for q in qualities:
        args = deepcopy(base_args)
        args.brotli_q = q
        print(f"\n=== Sweeping quality={q} over {len(files)} file(s) ===")
        rows = []
        for f in files:
            data = Path(f).read_bytes()
            row = bench.run_one(f, data, "brotli", args)
            rows.append(row)
            print(f"{f} | Q={q} | ratio={row['ratio']:.4f} "
                  f"comp={row['comp_MB_s']:.1f} MB/s ({row['comp_MiB_s']:.1f} MiB/s) "
                  f"decomp={row['decomp_MB_s']:.1f} MB/s ({row['decomp_MiB_s']:.1f} MiB/s)")
        # append per-quality to avoid large memory usage
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
            "qualities": qualities,
        },
        "brotli_base": {
            "mode": cfg.brotli.mode,
            "lgwin": cfg.brotli.lgwin,
        },
        "repeats": cfg.timing.repeats,
    }
    try:
        import brotli as _b
        meta["brotli_version"] = getattr(_b, "__version__", "unknown")
    except Exception:
        meta["brotli_version"] = None

    meta_path = out_csv.with_suffix(".meta.json")
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
