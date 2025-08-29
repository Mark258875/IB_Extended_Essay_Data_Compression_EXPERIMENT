#!/usr/bin/env python3
from pathlib import Path
import glob, json, time, platform

from config import Config
import bench_compress as bench

def main():
    # 1) Load config
    cfg = Config.load("conf/config.yaml")
    args = cfg.to_bench_args()
    out_csv = cfg.output_csv_path()

    # 2) Expand input files
    files = sorted(set(sum((glob.glob(pat) for pat in cfg.paths.input_globs), [])))
    if not files:
        print("No input files matched. Edit conf/config.yaml -> paths.input_globs")
        return

    # 3) Run Brotli on each file
    rows = []
    for f in files:
        data = Path(f).read_bytes()
        row = bench.run_one(f, data, "brotli", args)
        rows.append(row)
        print(f"{f} | brotli | ratio={row.get('ratio')} bpb={row.get('bpb')} "
              f"H={row.get('entropy_bpb')} over%={row.get('%_over_entropy')} "
              f"compMB/s={row.get('comp_MB_s')}")

    # 4) Write CSV + meta (reproducibility)
    bench.write_csv(rows, out_csv)
    meta = {
        "timestamp": time.time(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "input_globs": cfg.paths.input_globs,
        "output_csv": str(out_csv),
        "brotli": {
            "quality": cfg.brotli.quality,
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

    (out_csv.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows to {out_csv} and {out_csv.with_suffix('.meta.json')}")

if __name__ == "__main__":
    main()
