#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Tuple

# -------- configurable defaults --------
OUT_ROOT = Path("data/binary")
SIZE_BYTES = 10_000_000  # per file, for BOTH ascii01 and bitpack
P_VALUES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.99]

# -------- utilities --------

def counts_for_p(p: float, n_bits: int) -> Tuple[int, int]:
    """Return (zeros, ones) counts for n_bits given Pr[1]=p."""
    ones = round(p * n_bits)
    zeros = n_bits - ones
    return zeros, ones

def write_ascii_blocks(path: Path, zeros: int, ones: int, zeros_first: bool=True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        if zeros_first:
            if zeros: f.write(b"0" * zeros)
            if ones:  f.write(b"1" * ones)
        else:
            if ones:  f.write(b"1" * ones)
            if zeros: f.write(b"0" * zeros)

def write_ascii_alternating(path: Path, zeros: int, ones: int) -> None:
    """Write 0101... until one symbol runs out, then finish with remainder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    zb, ob = b"0", b"1"
    with path.open("wb") as f:
        # choose start symbol to roughly match the majority
        start_one = ones >= zeros
        z, o = zeros, ones
        chunk = bytearray()
        CHUNK_TARGET = 1_000_000  # bytes per flush

        cur = 1 if start_one else 0
        while z > 0 or o > 0:
            if cur == 1 and o > 0:
                chunk += ob
                o -= 1
                cur = 0
            elif cur == 0 and z > 0:
                chunk += zb
                z -= 1
                cur = 1
            else:
                # one symbol exhausted: flush remainder in one go
                if z > 0:
                    chunk += b"0" * z
                    z = 0
                if o > 0:
                    chunk += b"1" * o
                    o = 0
            if len(chunk) >= CHUNK_TARGET:
                f.write(chunk)
                chunk.clear()
        if chunk:
            f.write(chunk)

def write_bitpack_blocks(path: Path, zeros: int, ones: int, zeros_first: bool=True) -> None:
    """Write blocks in bit-packed form (MSB first in each byte)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total_bits = zeros + ones
    with path.open("wb") as f:
        def emit_bits(bits: Iterable[int]):
            acc = 0
            used = 0
            for b in bits:
                acc = ((acc << 1) | (1 if b else 0)) & 0xFF
                used += 1
                if used == 8:
                    f.write(bytes([acc]))
                    acc = 0
                    used = 0
            if used:  # pad remaining high bits with zeros
                acc <<= (8 - used)
                f.write(bytes([acc]))

        if zeros_first:
            emit_bits([0] * zeros)
            emit_bits([1] * ones)
        else:
            emit_bits([1] * ones)
            emit_bits([0] * zeros)

    # sanity: ensure exact byte length
    expect_bytes = (total_bits + 7) // 8
    assert path.stat().st_size == expect_bytes, "bitpack size mismatch"

def write_bitpack_alternating(path: Path, zeros: int, ones: int) -> None:
    """Write alternating in bit-packed form, then finish remainder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total_bits = zeros + ones
    with path.open("wb") as f:
        acc = 0
        used = 0
        z, o = zeros, ones
        # start with majority symbol to match counts better
        cur = 1 if o >= z else 0
        while z > 0 or o > 0:
            if cur == 1 and o > 0:
                bit = 1
                o -= 1
                cur = 0
            elif cur == 0 and z > 0:
                bit = 0
                z -= 1
                cur = 1
            else:
                # one exhausted; dump remainder
                if z > 0:
                    # fast-path: output zeros in runs
                    take = min(z, 8 - used)
                    acc = ((acc << take) & 0xFF)
                    used += take
                    z -= take
                    if used == 8:
                        f.write(bytes([acc])); acc = 0; used = 0
                    continue
                if o > 0:
                    # output ones in runs
                    take = min(o, 8 - used)
                    mask = (0xFF ^ (0xFF >> take))  # take leading ones
                    acc = ((acc << take) | mask) & 0xFF
                    used += take
                    o -= take
                    if used == 8:
                        f.write(bytes([acc])); acc = 0; used = 0
                    continue
                break

            acc = ((acc << 1) | bit) & 0xFF
            used += 1
            if used == 8:
                f.write(bytes([acc])); acc = 0; used = 0

        if used:
            acc <<= (8 - used)
            f.write(bytes([acc]))

    expect_bytes = (total_bits + 7) // 8
    assert path.stat().st_size == expect_bytes, "bitpack size mismatch"

def make_all() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["rep","order","p","size_bytes","zeros","ones","path"])

        for p in P_VALUES:
            # ascii01: file size is SIZE_BYTES (one byte per bit)
            n_bits_ascii = SIZE_BYTES
            z_ascii, o_ascii = counts_for_p(p, n_bits_ascii)

            # bitpack: also SIZE_BYTES, but that’s bytes; bits are 8×
            n_bits_packed = SIZE_BYTES * 8
            z_pack, o_pack = counts_for_p(p, n_bits_packed)

            # --- ascii01 / blocks + alternating ---
            base_dir = OUT_ROOT / "ascii01" / f"p{p:.2f}"
            f_blocks = base_dir / f"p{p:.2f}_blocks_zeros_then_ones_{SIZE_BYTES}B.txt"
            f_alt    = base_dir / f"p{p:.2f}_alternating_{SIZE_BYTES}B.txt"
            write_ascii_blocks(f_blocks, z_ascii, o_ascii, zeros_first=True)
            write_ascii_alternating(f_alt, z_ascii, o_ascii)
            w.writerow(["ascii01","blocks_zeros_then_ones",f"{p:.2f}",SIZE_BYTES,z_ascii,o_ascii,str(f_blocks)])
            w.writerow(["ascii01","alternating",          f"{p:.2f}",SIZE_BYTES,z_ascii,o_ascii,str(f_alt)])

            # --- bitpack / blocks + alternating ---
            base_dir = OUT_ROOT / "bitpack" / f"p{p:.2f}"
            f_blocks_b = base_dir / f"p{p:.2f}_blocks_zeros_then_ones_{SIZE_BYTES}B.bin"
            f_alt_b    = base_dir / f"p{p:.2f}_alternating_{SIZE_BYTES}B.bin"
            write_bitpack_blocks(f_blocks_b, z_pack, o_pack, zeros_first=True)
            write_bitpack_alternating(f_alt_b, z_pack, o_pack)
            w.writerow(["bitpack","blocks_zeros_then_ones",f"{p:.2f}",SIZE_BYTES,z_pack,o_pack,str(f_blocks_b)])
            w.writerow(["bitpack","alternating",          f"{p:.2f}",SIZE_BYTES,z_pack,o_pack,str(f_alt_b)])

            print(f"p={p:.2f} -> ascii01: {f_blocks.name}, {f_alt.name} | bitpack: {f_blocks_b.name}, {f_alt_b.name}")

    print(f"\nWrote binary sources and manifest -> {manifest_path}")

if __name__ == "__main__":
    make_all()
