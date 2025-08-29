#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import json
import time
import platform
import subprocess
from copy import deepcopy
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError

import psutil
from cpuinfo import get_cpu_info
import yaml
import brotli  # for version in meta

from config import Config
import bench_compress as bench
from runtime_tuning import apply_runtime


def expand_files(patterns):
    return sorted(set(sum((glob.glob(pat) for pat in patterns), [])))


def _safe_dist_version(dist_name: str, default: str = "unknown") -> str:
    try:
        return pkg_version(dist_name)
    except PackageNotFoundError:
        return default


def main():
    cfg = Config.load("conf/config.yaml")
    apply_runtime(cfg.runtime)
    base_args = cfg.to_bench_args()
    out_csv = cfg.output_csv_path_for("brotli")

    files = expand_files(cfg.paths.input_globs)
    if not files:
        print("No input files matched. Edit conf/config.yaml -> paths.input_globs")
        return

    # Warn if inputs are small (throughput gets noisy)
    min_bytes = int(getattr(cfg.runtime, "warn_small_bytes", 10_000_000))
    small = [f for f in files if Path(f).stat().st_size < min_bytes]
    if small:
        print(f"[warn] {len(small)} file(s) smaller than {min_bytes} bytes – throughput may be noisy.")

    rows_all = []
    qualities = list(range(0, 12))  # 0..11 inclusive

    for q in qualities:
        args = deepcopy(base_args)
        args.brotli_q = q
        print(f"\n=== Sweeping quality={q} over {len(files)} file(s) ===")
        rows = []
        for f in files:
            data = Path(f).read_bytes()  # I/O outside timing
            row = bench.run_one(f, data, "brotli", args)
            rows.append(row)
            print(
                f"{f} | Q={q} | ratio={row['ratio']:.4f} "
                f"comp={row['comp_MB_s']:.1f} MB/s ({row['comp_MiB_s']:.1f} MiB/s) "
                f"decomp={row['decomp_MB_s']:.1f} MB/s ({row['decomp_MiB_s']:.1f} MiB/s)"
            )
        bench.write_csv(rows, out_csv, append=True)
        rows_all.extend(rows)

    # ---- Meta for reproducibility ----
    meta = {
        "timestamp": time.time(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "input_globs": cfg.paths.input_globs,
        "output_csv": str(out_csv),
        "sweep": {"qualities": qualities},
        "brotli_base": {"mode": cfg.brotli.mode, "lgwin": cfg.brotli.lgwin},
        "repeats": cfg.timing.repeats,
        "warmup": cfg.timing.warmup,
        "runtime": {
            "cpu_affinity": cfg.runtime.cpu_affinity,
            "priority": cfg.runtime.priority,
        },
        "libs": {
            "PyYAML": _safe_dist_version("PyYAML"),
            "psutil": _safe_dist_version("psutil"),
            "py-cpuinfo": _safe_dist_version("py-cpuinfo"),
            "brotli": getattr(brotli, "__version__", "unknown"),
        },
    }

    # Hardware & power info (best-effort; never crash the run)
    try:
        meta["cpu"] = {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
        }
        f = psutil.cpu_freq()
        if f:
            meta["cpu"]["freq_mhz"] = {"current": f.current, "min": f.min, "max": f.max}
        meta["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception as e:
        meta["cpu_info_error"] = str(e)

    try:
        ci = get_cpu_info()
        meta.setdefault("cpu", {}).update({
            "brand": ci.get("brand_raw"),
            "arch": ci.get("arch_string_raw"),
            "bits": ci.get("bits"),
            "flags": ci.get("flags"),
        })
    except Exception as e:
        meta["cpuinfo_error"] = str(e)

    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powercfg", "/GETACTIVESCHEME"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            meta["power_plan"] = out.strip()
        else:
            gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            if os.path.exists(gov_path):
                meta["governor"] = Path(gov_path).read_text().strip()
    except Exception as e:
        meta["power_info_error"] = str(e)

    meta_path = out_csv.with_suffix(".meta.json")
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        old = {}
    old.update(meta)
    meta_path.write_text(json.dumps(old, indent=2), encoding="utf-8")

    print(f"\nDone. Appended {len(rows_all)} row(s) to {out_csv}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
