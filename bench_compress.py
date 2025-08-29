#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Tuple, Optional
import json, math, random, time, statistics, platform
import csv

try:
    import brotli
except Exception:
    brotli = None

MB = 1_000_000.0  # decimal MB for throughput (MB/s)
MiB = 1_048_576.9 

# ---------- Metrics helpers ----------

def entropy_bpb(data: bytes) -> float:
    """Zero-order Shannon entropy estimate in bits per byte."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    H = 0.0
    for c in counts:
        if c:
            p = c / n
            H -= p * (math.log(p, 2))
    return H

def time_op(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)

# ---------- (Optional) synthetic data generators ----------

def gen_uniform(size: int, seed: Optional[int]) -> bytes:
    rnd = random.Random(seed)
    return bytes(rnd.randrange(256) for _ in range(size))

def gen_gaussian(size: int, mu: float, sigma: float, seed: Optional[int]) -> bytes:
    rnd = random.Random(seed)
    out = bytearray(size)
    for i in range(size):
        x = int(rnd.gauss(mu, sigma))
        if x < 0: x = 0
        if x > 255: x = 255
        out[i] = x
    return bytes(out)

def gen_zipf(size: int, s: float, seed: Optional[int]) -> bytes:
    weights = [1.0 / ((k + 1) ** s) for k in range(256)]
    total = sum(weights)
    probs = [w / total for w in weights]
    cdf = []
    csum = 0.0
    for p in probs:
        csum += p
        cdf.append(csum)
    rnd = random.Random(seed)
    out = bytearray(size)
    for i in range(size):
        r = rnd.random()
        lo, hi = 0, 255
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        out[i] = lo
    return bytes(out)

# ---------- Brotli wrapper + single-ALG runner ----------

def compress_brotli(data: bytes, quality: int = 6,
                    mode: Optional[str] = None,
                    window: Optional[int] = None) -> Tuple[bytes, float, float, Dict]:
    if brotli is None:
        raise RuntimeError("brotli package not installed. pip install brotli")
    kwargs = {"quality": quality}
    if mode:
        kwargs["mode"] = {
            "generic": brotli.MODE_GENERIC,
            "text": brotli.MODE_TEXT,
            "font": brotli.MODE_FONT
        }.get(mode, brotli.MODE_GENERIC)
    # ⬇⬇ important: only include lgwin if not None
    if window is not None:
        kwargs["lgwin"] = window
    comp, comp_s = time_op(brotli.compress, data, **kwargs)
    decomp, decomp_s = time_op(brotli.decompress, comp)
    assert decomp == data
    params = {"quality": quality, "mode": mode or "generic", "lgwin": window if window is not None else "default"}
    return comp, comp_s, decomp_s, params


def _percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def run_one(dataset_label: str, data: bytes, alg: str, args):
    if alg != "brotli":
        raise ValueError("This trimmed module supports only 'brotli'.")

    H = entropy_bpb(data)
    original = len(data)

    # warm-up once; then repeat timings and take medians
    comp, ct, dt, params = compress_brotli(
        data,
        quality=getattr(args, "brotli_q", 6),
        mode=getattr(args, "brotli_mode", "generic"),
        window=getattr(args, "brotli_lgwin", None),
    )
    comp_times = [ct]; decomp_times = [dt]
    repeats = max(1, int(getattr(args, "repeats", 3)))
    for _ in range(repeats - 1):
        kwargs = {
            "quality": getattr(args, "brotli_q", 6),
            "mode": {
                "generic": brotli.MODE_GENERIC,
                "text": brotli.MODE_TEXT,
                "font": brotli.MODE_FONT
            }[getattr(args, "brotli_mode", "generic")]
        }
        lgwin_val = getattr(args, "brotli_lgwin", None)
        if lgwin_val is not None:
            kwargs["lgwin"] = lgwin_val

        _, ct2 = time_op(brotli.compress, data, **kwargs)
        _, dt2 = time_op(brotli.decompress, comp)
        comp_times.append(ct2); decomp_times.append(dt2)


    ct_med = statistics.median(comp_times)
    dt_med = statistics.median(decomp_times)

    compressed = len(comp)
    ratio = compressed / original if original else 0.0
    bpb = 8.0 * ratio if original else 0.0
    over = ((bpb - H) / H * 100.0) if H > 0 else None

    # Throughput in MB/s (decimal) and MiB/s (binary)
    comp_MB_s = (original / MB) / ct_med if ct_med > 0 else float("inf")
    decomp_MB_s = (original / MB) / dt_med if dt_med > 0 else float("inf")
    comp_MiB_s = (original / MiB) / ct_med if ct_med > 0 else float("inf")
    decomp_MiB_s = (original / MiB) / dt_med if dt_med > 0 else float("inf")

    row = {
        "dataset": dataset_label,
        "alg": "brotli",
        "params": json.dumps(params, sort_keys=True),
        "original_bytes": original,
        "compressed_bytes": compressed,
        "ratio": ratio,
        "bpb": bpb,
        "entropy_bpb": H,
        "%_over_entropy": over,
        "comp_time_s": ct_med,
        "comp_MB_s": comp_MB_s,
        "comp_MiB_s": comp_MiB_s,
        "decomp_time_s": dt_med,
        "decomp_MB_s": decomp_MB_s,
        "decomp_MiB_s": decomp_MiB_s,
        "repeats": repeats,
        "env_python": platform.python_version(),
        "env_platform": platform.platform(),
        "env_brotli": getattr(brotli, "__version__", "unknown"),
    }
    return row

def write_csv(rows, path: Path, append: bool = False):
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    mode = "a" if append and file_exists else "w"
    with path.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow(r)
