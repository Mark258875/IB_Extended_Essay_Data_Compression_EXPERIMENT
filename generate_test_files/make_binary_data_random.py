#!/usr/bin/env python3
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Tuple

# -------- configurable defaults --------
OUT_ROOT = Path("data/binary")
SIZE_BYTES = 10_000_000  # per file, for BOTH ascii01 and bitpack
P_VALUES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.99]
SEED_BASE = 12345        # base seed; per-file seed derived from p for reproducibility


# -------- utilities --------

def counts_for_p(p: float, n_bits: int) -> Tuple[int, int]:
    """Return (zeros, ones) counts for n_bits given Pr[1]=p (exact count via rounding)."""
    ones = round(p * n_bits)
    zeros = n_bits - ones
    return zeros, ones


def iter_random_bits_counts(zeros: int, ones: int, rng: random.Random):
    """
    Stream a random permutation of a multiset with exactly `zeros` zeros and `ones` ones.
    This generates each bit online with probability equal to the remaining fraction:
        P(next=1) = ones / (zeros + ones)
    which yields a uniform random sequence among all sequences with those counts.
    """
    z, o = zeros, ones
    while z + o:
        if z == 0:
            o -= 1
            yield 1
        elif o == 0:
            z -= 1
            yield 0
        else:
            if rng.random() < (o / (z + o)):
                o -= 1
                yield 1
            else:
                z -= 1
                yield 0


# -------- writers (ascii01 and bitpack) --------

def write_ascii_random(path: Path, zeros: int, ones: int, seed: int) -> None:
    """
    Write randomized sequence of '0' and '1' (ASCII bytes) with exact counts.
    File size (bytes) == zeros + ones.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    zb, ob = b"0", b"1"
    chunk = bytearray()
    CHUNK_TARGET = 1_000_000  # bytes per flush

    with path.open("wb") as f:
        for bit in iter_random_bits_counts(zeros, ones, rng):
            chunk += ob if bit else zb
            if len(chunk) >= CHUNK_TARGET:
                f.write(chunk)
                chunk.clear()
        if chunk:
            f.write(chunk)

    # sanity
    assert path.stat().st_size == zeros + ones, "ascii01 size mismatch"


def write_bitpack_random(path: Path, zeros: int, ones: int, seed: int) -> None:
    """
    Write randomized bits packed MSB→LSB into bytes with exact counts.
    File size (bytes) == ceil((zeros+ones)/8).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    acc = 0
    used = 0
    total_bits = zeros + ones

    with path.open("wb") as f:
        for bit in iter_random_bits_counts(zeros, ones, rng):
            acc = ((acc << 1) | (1 if bit else 0)) & 0xFF
            used += 1
            if used == 8:
                f.write(bytes([acc]))
                acc = 0
                used = 0

        if used:
            acc <<= (8 - used)  # pad remaining high bits with zeros
            f.write(bytes([acc]))

    expect_bytes = (total_bits + 7) // 8
    assert path.stat().st_size == expect_bytes, "bitpack size mismatch"


# -------- main generator --------

def make_all() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "manifest.csv"

    # Append if exists; write header only for a new/empty file
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        if write_header:
            w.writerow(["rep", "order", "p", "size_bytes", "zeros", "ones", "seed", "path"])

        for p in P_VALUES:
            # ascii01: SIZE_BYTES bytes ⇒ SIZE_BYTES bits
            n_bits_ascii = SIZE_BYTES
            z_ascii, o_ascii = counts_for_p(p, n_bits_ascii)
            seed_ascii = SEED_BASE + int(round(p * 100))

            base_dir_a = OUT_ROOT / "ascii01" / f"p{p:.2f}"
            f_rand_a = base_dir_a / f"p{p:.2f}_random_{SIZE_BYTES}B.txt"
            write_ascii_random(f_rand_a, z_ascii, o_ascii, seed_ascii)
            w.writerow(["ascii01", "random", f"{p:.2f}", SIZE_BYTES, z_ascii, o_ascii, seed_ascii, str(f_rand_a)])

            # bitpack: SIZE_BYTES bytes ⇒ SIZE_BYTES*8 bits
            n_bits_pack = SIZE_BYTES * 8
            z_pack, o_pack = counts_for_p(p, n_bits_pack)
            seed_pack = SEED_BASE * 2 + int(round(p * 100))

            base_dir_b = OUT_ROOT / "bitpack" / f"p{p:.2f}"
            f_rand_b = base_dir_b / f"p{p:.2f}_random_{SIZE_BYTES}B.bin"
            write_bitpack_random(f_rand_b, z_pack, o_pack, seed_pack)
            w.writerow(["bitpack", "random", f"{p:.2f}", SIZE_BYTES, z_pack, o_pack, seed_pack, str(f_rand_b)])

            print(f"p={p:.2f} -> ascii01: {f_rand_a.name} | bitpack: {f_rand_b.name}")

    print(f"\nAppended randomized binary sources to -> {manifest_path}")


if __name__ == "__main__":
    make_all()
