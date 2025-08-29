#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Tuple, Optional
import json, math, random, time, statistics, platform
import csv

try:
    import brotli
except Exception:
    brotli = None

try:
    import lz4.frame as lz4
except ImportError:
    lz4 = None

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
def compress_brotli(data: bytes, args) -> Tuple[bytes, float, float, Dict]:
    """Compress+decompress once and return (comp_bytes, comp_time, decomp_time, params)."""
    if brotli is None:
        raise RuntimeError("brotli package not installed. pip install brotli")

    q = int(getattr(args, "brotli_q", 6))
    mode = (getattr(args, "brotli_mode", "generic") or "generic").lower()
    lgwin = getattr(args, "brotli_lgwin", None)

    kwargs = {"quality": q}
    if mode in {"generic", "text", "font"}:
        kwargs["mode"] = {
            "generic": brotli.MODE_GENERIC,
            "text": brotli.MODE_TEXT,
            "font": brotli.MODE_FONT,
        }[mode]
    if lgwin is not None:
        kwargs["lgwin"] = int(lgwin)

    comp, comp_s = time_op(brotli.compress, data, **kwargs)
    decomp, decomp_s = time_op(brotli.decompress, comp)
    assert decomp == data

    params = {"quality": q, "mode": mode, "lgwin": lgwin if lgwin is not None else "default"}
    return comp, comp_s, decomp_s, params

def compress_lz4(data: bytes, args):
    if lz4 is None:
        raise RuntimeError("lz4 package not installed. pip install lz4")
    comp, comp_s = time_op(lz4.compress, data, compression_level=int(args.lz4_level))
    decomp, decomp_s = time_op(lz4.decompress, comp)
    assert decomp == data
    params = {"level": int(args.lz4_level)}
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
    compress_fn = {
        "brotli": compress_brotli,
        "lz4": compress_lz4,
    }.get(alg)
    if compress_fn is None:
        raise ValueError(f"Unknown algorithm: {alg}")

    H = entropy_bpb(data)
    original = len(data)

    # Warm-up
    warm = max(0, getattr(args, "warmup", 1))
    for _ in range(warm):
        compress_fn(data, args)

    comp_times, decomp_times = [], []
    repeats = max(1, args.repeats)

    last_comp = None
    last_params = None
    for _ in range(repeats):
        comp, ct, _, params = compress_fn(data, args)
        _, dt = time_op({"brotli": brotli.decompress, "lz4": lz4.decompress}[alg], comp)
        comp_times.append(ct)
        decomp_times.append(dt)
        last_comp = comp
        last_params = params

    ct_med = statistics.median(comp_times)
    dt_med = statistics.median(decomp_times)

    comp_MB_s   = (original / MB)  / ct_med if ct_med > 0 else float("inf")
    decomp_MB_s = (original / MB)  / dt_med if dt_med > 0 else float("inf")
    comp_MiB_s  = (original / MiB) / ct_med if ct_med > 0 else float("inf")
    decomp_MiB_s= (original / MiB) / dt_med if dt_med > 0 else float("inf")


    compressed = len(last_comp)
    ratio = compressed / original if original else 0.0
    bpb = 8.0 * ratio if original else 0.0
    over = ((bpb - H) / H * 100.0) if H > 0 else None
    comp_MB_s = (original / MB) / ct_med if ct_med > 0 else float("inf")
    decomp_MB_s = (original / MB) / dt_med if dt_med > 0 else float("inf")

    row = {
        "dataset": dataset_label,
        "alg": alg,
        "params": json.dumps(last_params, sort_keys=True),
        "original_bytes": original,
        "compressed_bytes": compressed,
        "ratio": ratio,
        "bpb": bpb,
        "entropy_bpb": H,
        "%_over_entropy": over,
        "comp_time_s": ct_med,
        "comp_MB_s": comp_MB_s,
        "decomp_time_s": dt_med,
        "decomp_MB_s": decomp_MB_s,
        "repeats": repeats,
        "comp_time_s": ct_med,
        "comp_MB_s": comp_MB_s,
        "comp_MiB_s": comp_MiB_s,
        "decomp_time_s": dt_med,
        "decomp_MB_s": decomp_MB_s,
        "decomp_MiB_s": decomp_MiB_s, 
        "env_python": platform.python_version(),
        "env_platform": platform.platform(),
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
