#!/usr/bin/env python3
from __future__ import annotations

import bisect
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# =========================
# ===== math helpers  =====
# =========================

def shannon_H(probs: List[float]) -> float:
    """Shannon entropy in bits for a discrete distribution."""
    H = 0.0
    for p in probs:
        if p > 0.0:
            H -= p * math.log2(p)
    return H


def H0_of_hist(counts: List[int]) -> float:
    """Zero-order entropy (bits/symbol) for integer counts."""
    n = sum(counts)
    if n == 0:
        return 0.0
    inv_n = 1.0 / n
    H = 0.0
    for c in counts:
        if c:
            p = c * inv_n
            H -= p * math.log2(p)
    return H


def Hrate_from_transitions(trans: List[List[int]]) -> float:
    """
    First-order entropy *rate* estimate from transition counts.
    H_rate = sum_i pi_i * H(T_i) where
      - pi_i ~ row-sum_i / total
      - T_i(j) ~ trans[i][j] / row-sum_i
    Returns bits/symbol. If some rows are empty, they contribute 0.
    """
    rowsum = [sum(row) for row in trans]
    total = sum(rowsum)
    if total == 0:
        return 0.0
    H = 0.0
    for i, rsum in enumerate(rowsum):
        if rsum == 0:
            continue
        pi = rsum / total
        row = trans[i]
        hrow = 0.0
        for c in row:
            if c:
                p = c / rsum
                hrow -= p * math.log2(p)
        H += pi * hrow
    return H


# =================================================
# ===== distributions, targets and calibration =====
# =================================================

def peaked_flat_probs(K: int, q: float) -> List[float]:
    """
    p = [1-(K-1)q, q, q, ..., q]; q in (0, 1/K].
    Entropy increases with q on [0, 1/K].
    """
    q = max(0.0, min(q, 1.0 / K))
    p0 = 1.0 - (K - 1) * q
    return [p0] + [q] * (K - 1)


def peaked_flat_q_for_target_H(K: int, H_target: float, iters: int = 70) -> float:
    """Find q s.t. H(peaked_flat_probs(K,q)) ~= H_target (bisection)."""
    Hmax = math.log2(K)
    H_target = max(0.0, min(H_target, Hmax))
    lo, hi = 0.0, 1.0 / K
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        Hmid = shannon_H(peaked_flat_probs(K, mid))
        if Hmid < H_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def zipf_probs(K: int, s: float) -> List[float]:
    """Zipf distribution p_i ∝ i^{-s}, i=1..K (1-indexed rank)."""
    weights = [1.0 / ((i + 1) ** s) for i in range(K)]
    Z = sum(weights)
    return [w / Z for w in weights]


def zipf_s_for_target_H(K: int, H_target: float, s_lo: float = 0.0, s_hi: float = 4.0, iters: int = 70) -> float:
    """
    Solve for s such that H(zipf(K,s)) ~= H_target.
    H decreases with s (s=0 -> uniform -> log2 K).
    """
    Hmax = math.log2(K)
    H_target = max(0.0, min(H_target, Hmax))
    lo, hi = s_lo, s_hi
    # widen hi if needed
    while shannon_H(zipf_probs(K, hi)) > H_target and hi < 100.0:
        hi *= 2.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        Hmid = shannon_H(zipf_probs(K, mid))
        if Hmid > H_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def markov_rho_for_target_rate(K: int, H_target: float, iters: int = 70) -> float:
    """
    Uniform stationary Markov with persistence rho:
      T_i = rho * e_i + (1-rho) * uniform
    Entropy rate:
      stay = rho + (1-rho)/K
      move = (1-rho)/K
      H_rate = -[stay log2 stay + (K-1)*move log2 move]
    Solve for rho to match H_target.
    """
    Hmax = math.log2(K)
    H_target = max(0.0, min(H_target, Hmax))

    def Hrate(rho: float) -> float:
        stay = rho + (1.0 - rho) / K
        move = (1.0 - rho) / K
        # guard: stay/move > 0
        stay = max(1e-18, min(1.0, stay))
        move = max(1e-18, min(1.0, move))
        return -(stay * math.log2(stay) + (K - 1) * move * math.log2(move))

    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        Hmid = Hrate(mid)
        if Hmid > H_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ======================================
# ===== sampling & sequence builders ===
# ======================================

def multinomial_exact_counts(p: List[float], n: int) -> List[int]:
    """
    Largest-remainder rounding to get integer counts summing to n, closely matching p.
    """
    raw = [pi * n for pi in p]
    base = [int(math.floor(x)) for x in raw]
    rems = [x - b for x, b in zip(raw, base)]
    need = n - sum(base)
    order = sorted(range(len(p)), key=lambda i: rems[i], reverse=True)
    for i in range(need):
        base[order[i]] += 1
    return base


def cdf_from_probs(p: List[float]) -> List[float]:
    cdf, a = [], 0.0
    for pi in p:
        a += pi
        cdf.append(a)
    cdf[-1] = 1.0
    return cdf


def sample_iid_symbols(p: List[float], n: int, rng: random.Random) -> List[int]:
    cdf = cdf_from_probs(p)
    out = [0] * n
    for i in range(n):
        r = rng.random()
        out[i] = bisect.bisect_left(cdf, r)
    return out


def symbols_from_counts_ordered(counts: List[int], order: str, rng: random.Random) -> List[int]:
    """
    Build a sequence with EXACT histogram 'counts', arranged by 'order':
      - 'random': random permutation of the multiset
      - 'blocks': all 0s, then all 1s, then ...
      - 'alt':    cycle 0,1,2,... until counts exhausted
    """
    K = len(counts)
    if sum(counts) == 0:
        return []

    if order == "blocks":
        out = []
        for sym, c in enumerate(counts):
            if c > 0:
                out.extend([sym] * c)
        return out

    if order == "alt":
        out = []
        rem = counts[:]
        sym = 0
        while sum(rem) > 0:
            if rem[sym] > 0:
                out.append(sym)
                rem[sym] -= 1
            sym = (sym + 1) % K
        return out

    # random (default)
    multiset = []
    for sym, c in enumerate(counts):
        multiset.extend([sym] * c)
    rng.shuffle(multiset)
    return multiset


def simulate_markov_uniform(K: int, n: int, rho: float, rng: random.Random) -> Tuple[List[int], List[List[int]]]:
    """
    Simulate order-1 Markov with uniform stationary and persistence rho.
    Returns (symbols, transition_counts[K][K]).
    """
    # Build per-state CDF
    u = 1.0 / K
    rows_cdf = []
    for i in range(K):
        row = [(rho + (1 - rho) * u) if j == i else (1 - rho) * u for j in range(K)]
        rows_cdf.append(cdf_from_probs(row))

    x = rng.randrange(K)
    seq = [x]
    trans = [[0] * K for _ in range(K)]
    for _ in range(1, n):
        r = rng.random()
        nx = bisect.bisect_left(rows_cdf[x], r)
        trans[x][nx] += 1
        x = nx
        seq.append(x)
    return seq, trans


# =========================
# ===== bit packing   =====
# =========================

def bits_per_symbol(K: int) -> int:
    return max(1, math.ceil(math.log2(K)))


def pack_symbols_to_bytes(symbols: List[int], K: int) -> bytes:
    """
    Pack symbols in [0,K-1] into ceil(log2 K) bits per symbol, MSB->LSB in each byte.
    """
    bps = bits_per_symbol(K)
    acc = 0
    used = 0
    out = bytearray()
    for s in symbols:
        if s < 0 or s >= K:
            raise ValueError(f"symbol {s} out of range for K={K}")
        # append bps bits of s into acc (MSB first)
        for k in reversed(range(bps)):
            bit = (s >> k) & 1
            acc = ((acc << 1) | bit) & 0xFF
            used += 1
            if used == 8:
                out.append(acc)
                acc = 0
                used = 0
    if used:
        acc <<= (8 - used)
        out.append(acc)
    return bytes(out)


# =========================
# ===== config model  =====
# =========================

@dataclass
class KaryGenConfig:
    out_root: Path
    encoding: str                  # "byte" | "pack"
    size_bytes: int                # desired file size (approx for pack)
    seeds: List[int]               # list of seeds
    overwrite: bool
    datasets: List[Dict]           # list of dataset specs (see sample YAML)

    @staticmethod
    def load_from_yaml(path: Path) -> "KaryGenConfig":
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "karygen" not in d:
            raise ValueError("conf/config.yaml must contain a top-level 'karygen' section.")
        g = d["karygen"]
        out_root = Path(g.get("out_root", "./data/kary"))
        encoding = str(g.get("encoding", "byte")).lower()
        if encoding not in {"byte", "pack"}:
            raise ValueError("karygen.encoding must be 'byte' or 'pack'.")
        size_bytes = int(g.get("size_bytes", 10_000_000))
        seeds = list(map(int, g.get("seeds", [123])))
        overwrite = bool(g.get("overwrite", False))
        datasets = g.get("datasets", [])
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("karygen.datasets must be a non-empty list.")
        return KaryGenConfig(out_root, encoding, size_bytes, seeds, overwrite, datasets)


# =========================
# ===== writers & I/O =====
# =========================

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def write_bytes(p: Path, data: bytes, overwrite: bool) -> None:
    ensure_parent(p)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File exists: {p} (set overwrite: true in config to replace)")
    p.write_bytes(data)


def open_manifest(root: Path) -> Tuple[csv.DictWriter, any]:
    mf = root / "manifest.csv"
    ensure_parent(mf)
    write_header = not mf.exists() or mf.stat().st_size == 0
    f = mf.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=[
        "path",
        "model",
        "K",
        "encoding",
        "seed",
        "target_bits_per_symbol",
        "measured_H0_bits_per_symbol",
        "measured_Hrate_bits_per_symbol",
        "n_symbols",
        "bytes_written",
        "source_bits_per_symbol",
        "params_json",
    ])
    if write_header:
        w.writeheader()
    return w, f


# =========================
# ===== main pipeline  ====
# =========================

def main():
    cfg = KaryGenConfig.load_from_yaml(Path("conf/config.yaml"))
    out_root = cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_writer, manifest_file = open_manifest(out_root)

    total_files = 0

    try:
        for spec in cfg.datasets:
            model = str(spec.get("model", "")).lower()
            if model not in {"iid_peaked", "zipf", "markov_persistent", "histperm"}:
                print(f"[skip] unknown model: {model}")
                continue

            K = int(spec.get("K", 256))
            if K < 2:
                print(f"[skip] K must be >= 2 (got {K})")
                continue

            encode = cfg.encoding
            bps = bits_per_symbol(K)
            if encode == "byte" and K > 256:
                raise ValueError(f"karygen.encoding='byte' requires K<=256 (got {K}).")

            # derive n_symbols from target file size and encoding
            if encode == "byte":
                n_symbols = cfg.size_bytes
            else:
                n_symbols = (cfg.size_bytes * 8) // bps  # pack mode: approximate
                if n_symbols == 0:
                    raise ValueError("size_bytes too small for packed encoding.")

            seeds = cfg.seeds

            # ---- model-specific parameter grids ----
            # Accepted keys per model:
            # - iid_peaked: targets_bits_per_symbol: [..]
            # - zipf: s_values: [..]  OR targets_bits_per_symbol: [..] (then solve s)
            # - markov_persistent: targets_bits_per_symbol: [..]
            # - histperm: target_bits_per_symbol: H0*, order: one or list of {"random","blocks","alt"}
            #
            # Optional common: name_prefix

            name_prefix = spec.get("name_prefix", "")

            if model == "iid_peaked":
                targets = list(map(float, spec.get("targets_bits_per_symbol", [])))
                if not targets:
                    print("[iid_peaked] no targets_bits_per_symbol provided; skipping.")
                    continue

                for Ht in targets:
                    q = peaked_flat_q_for_target_H(K, Ht)
                    p = peaked_flat_probs(K, q)

                    for seed in seeds:
                        rng = random.Random(seed)
                        # sample EXACT histogram via counts to reduce variance
                        counts = multinomial_exact_counts(p, n_symbols)
                        symbols = symbols_from_counts_ordered(counts, "random", rng)
                        # encode
                        if encode == "byte":
                            data = bytes(symbols)  # K<=256 guaranteed
                        else:
                            data = pack_symbols_to_bytes(symbols, K)

                        # stats (symbol domain)
                        H0_meas = H0_of_hist(counts)
                        Hrate_meas = ""  # N/A for i.i.d.

                        # write
                        out_dir = out_root / f"K{K}" / "iid_peaked"
                        fname = f"{name_prefix}H{Ht:.3f}_q{q:.6f}_N{len(symbols)}_seed{seed}.bin"
                        opath = out_dir / fname
                        write_bytes(opath, data, cfg.overwrite)

                        # manifest
                        manifest_writer.writerow({
                            "path": str(opath),
                            "model": "iid_peaked",
                            "K": K,
                            "encoding": encode,
                            "seed": seed,
                            "target_bits_per_symbol": f"{Ht:.6f}",
                            "measured_H0_bits_per_symbol": f"{H0_meas:.6f}",
                            "measured_Hrate_bits_per_symbol": "",
                            "n_symbols": len(symbols),
                            "bytes_written": len(data),
                            "source_bits_per_symbol": bps if encode == "pack" else 8,  # file-domain; for byte, 8 bits/byte
                            "params_json": json.dumps({"q": q, "peaked": True}, sort_keys=True),
                        })
                        total_files += 1
                        print(f"[write] {opath}  (H0≈{H0_meas:.4f} bits/sym)")

            elif model == "zipf":
                s_values = spec.get("s_values")
                targets = spec.get("targets_bits_per_symbol")
                grid: List[Tuple[float, float]] = []  # list of (s, H_target)

                if s_values:
                    for s in s_values:
                        s = float(s)
                        grid.append((s, shannon_H(zipf_probs(K, s))))
                if targets:
                    for Ht in targets:
                        s = zipf_s_for_target_H(K, float(Ht))
                        grid.append((s, float(Ht)))

                if not grid:
                    print("[zipf] provide s_values or targets_bits_per_symbol; skipping.")
                    continue

                for s, Ht in grid:
                    p = zipf_probs(K, s)
                    for seed in seeds:
                        rng = random.Random(seed)
                        counts = multinomial_exact_counts(p, n_symbols)
                        symbols = symbols_from_counts_ordered(counts, "random", rng)

                        if encode == "byte":
                            data = bytes(symbols)
                        else:
                            data = pack_symbols_to_bytes(symbols, K)

                        H0_meas = H0_of_hist(counts)

                        out_dir = out_root / f"K{K}" / "zipf"
                        fname = f"{name_prefix}H{Ht:.3f}_s{s:.4f}_N{len(symbols)}_seed{seed}.bin"
                        opath = out_dir / fname
                        write_bytes(opath, data, cfg.overwrite)

                        manifest_writer.writerow({
                            "path": str(opath),
                            "model": "zipf",
                            "K": K,
                            "encoding": encode,
                            "seed": seed,
                            "target_bits_per_symbol": f"{Ht:.6f}",
                            "measured_H0_bits_per_symbol": f"{H0_meas:.6f}",
                            "measured_Hrate_bits_per_symbol": "",
                            "n_symbols": len(symbols),
                            "bytes_written": len(data),
                            "source_bits_per_symbol": bps if encode == "pack" else 8,
                            "params_json": json.dumps({"s": s}, sort_keys=True),
                        })
                        total_files += 1
                        print(f"[write] {opath}  (H0≈{H0_meas:.4f} bits/sym)")

            elif model == "markov_persistent":
                targets = list(map(float, spec.get("targets_bits_per_symbol", [])))
                if not targets:
                    print("[markov_persistent] no targets_bits_per_symbol provided; skipping.")
                    continue

                for Ht in targets:
                    rho = markov_rho_for_target_rate(K, Ht)
                    for seed in seeds:
                        rng = random.Random(seed)
                        symbols, trans = simulate_markov_uniform(K, n_symbols, rho, rng)

                        if encode == "byte":
                            data = bytes(symbols)
                        else:
                            data = pack_symbols_to_bytes(symbols, K)

                        # stats
                        # histogram H0 (for info)
                        counts = [0] * K
                        for s in symbols:
                            counts[s] += 1
                        H0_meas = H0_of_hist(counts)
                        Hrate_meas = Hrate_from_transitions(trans)

                        out_dir = out_root / f"K{K}" / "markov_persistent"
                        fname = f"{name_prefix}Hrate{Ht:.3f}_rho{rho:.6f}_N{len(symbols)}_seed{seed}.bin"
                        opath = out_dir / fname
                        write_bytes(opath, data, cfg.overwrite)

                        manifest_writer.writerow({
                            "path": str(opath),
                            "model": "markov_persistent",
                            "K": K,
                            "encoding": encode,
                            "seed": seed,
                            "target_bits_per_symbol": f"{Ht:.6f}",
                            "measured_H0_bits_per_symbol": f"{H0_meas:.6f}",
                            "measured_Hrate_bits_per_symbol": f"{Hrate_meas:.6f}",
                            "n_symbols": len(symbols),
                            "bytes_written": len(data),
                            "source_bits_per_symbol": bps if encode == "pack" else 8,
                            "params_json": json.dumps({"rho": rho}, sort_keys=True),
                        })
                        total_files += 1
                        print(f"[write] {opath}  (Hrate≈{Hrate_meas:.4f} bits/sym)")

            elif model == "histperm":
                # Same histogram (target H0), different *order*.
                Ht = float(spec.get("target_bits_per_symbol", 0.0))
                shape = str(spec.get("shape", "peaked")).lower()  # 'peaked' or 'zipf'
                orders = spec.get("orders", ["random", "blocks", "alt"])
                if isinstance(orders, str):
                    orders = [orders]

                if shape == "peaked":
                    q = peaked_flat_q_for_target_H(K, Ht)
                    p = peaked_flat_probs(K, q)
                    shape_params = {"shape": "peaked", "q": q}
                elif shape == "zipf":
                    s = zipf_s_for_target_H(K, Ht)
                    p = zipf_probs(K, s)
                    shape_params = {"shape": "zipf", "s": s}
                else:
                    raise ValueError("histperm.shape must be 'peaked' or 'zipf'.")

                for seed in seeds:
                    rng = random.Random(seed)
                    counts = multinomial_exact_counts(p, n_symbols)
                    for order in orders:
                        symbols = symbols_from_counts_ordered(counts, order, rng)
                        if encode == "byte":
                            data = bytes(symbols)
                        else:
                            data = pack_symbols_to_bytes(symbols, K)

                        H0_meas = H0_of_hist(counts)

                        out_dir = out_root / f"K{K}" / "histperm"
                        fname = f"{name_prefix}H0{Ht:.3f}_{order}_N{len(symbols)}_seed{seed}.bin"
                        opath = out_dir / fname
                        write_bytes(opath, data, cfg.overwrite)

                        params = {"order": order}
                        params.update(shape_params)

                        manifest_writer.writerow({
                            "path": str(opath),
                            "model": "histperm",
                            "K": K,
                            "encoding": encode,
                            "seed": seed,
                            "target_bits_per_symbol": f"{Ht:.6f}",
                            "measured_H0_bits_per_symbol": f"{H0_meas:.6f}",
                            "measured_Hrate_bits_per_symbol": "",
                            "n_symbols": len(symbols),
                            "bytes_written": len(data),
                            "source_bits_per_symbol": bps if encode == "pack" else 8,
                            "params_json": json.dumps(params, sort_keys=True),
                        })
                        total_files += 1
                        print(f"[write] {opath}  (H0≈{H0_meas:.4f} bits/sym | {order})")

    finally:
        manifest_file.close()

    print(f"\nDone. Wrote {total_files} file(s). Manifest -> {out_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
