#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Any, Dict
import yaml

_VALID_MODES = {"generic", "text", "font"}

@dataclass
class Paths:
    input_globs: List[str]
    output_dir: str = "./results"
    # Prefer per-alg outputs; kept backward-compat with legacy output_filename
    output_brotli: str = "results_brotli.csv"
    output_lz4: str = "results_lz4.csv"

    def _ensure_dir(self) -> Path:
        outdir = Path(self.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def output_path_for(self, alg: str) -> Path:
        outdir = self._ensure_dir()
        if alg.lower() == "brotli":
            return outdir / self.output_brotli
        if alg.lower() == "lz4":
            return outdir / self.output_lz4
        # Fallback: unknown alg -> put in generic results.csv
        return outdir / "results.csv"

@dataclass
class RuntimeCfg:
    cpu_affinity: Optional[int] = None
    priority: str = "normal"
    warn_small_bytes: int = 10_000_000
    
@dataclass
class BrotliCfg:
    quality: int = 6              # 0..11
    mode: str = "generic"         # generic|text|font
    lgwin: Optional[int] = None   # 10..24 or None

@dataclass
class Lz4Cfg:
    compression_level: int = 1  # 0..16

@dataclass
class Timing:
    repeats: int = 5              # >=1
    warmup: int = 1               # >=0

@dataclass
class Config:
    paths: Paths
    brotli: BrotliCfg = field(default_factory=BrotliCfg)
    lz4: Lz4Cfg = field(default_factory=Lz4Cfg)
    timing: Timing = field(default_factory=Timing)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)

    # ---- loading & validation ----
    @staticmethod
    def load(path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")

        # apply defaults and coerce to dataclasses
        paths = Config._parse_paths(data.get("paths", {}))
        brotli = Config._parse_brotli(data.get("brotli", {}))
        lz4 = Config._parse_lz4(data.get("lz4", {}))
        timing = Config._parse_timing(data.get("timing", {}))
        runtime = Config._parse_runtime(data.get("runtime", {}))
        cfg = Config(paths=paths, brotli=brotli, lz4=lz4, timing=timing, runtime=runtime)
        cfg._validate()
        return cfg

    @staticmethod
    def _parse_runtime(d: Dict[str, Any]) -> RuntimeCfg:
        return RuntimeCfg(
            cpu_affinity=(int(d["cpu_affinity"]) if d.get("cpu_affinity") is not None else None),
            priority=str(d.get("priority", "normal")).lower(),
            warn_small_bytes=int(d.get("warn_small_bytes", 10_000_000)),  # new
        )

    @staticmethod
    def _parse_paths(d: Dict[str, Any]) -> Paths:
        input_globs = d.get("input_globs")
        if not input_globs or not isinstance(input_globs, list):
            raise ValueError("paths.input_globs must be a non-empty list of glob patterns.")

        # Back-compat: if legacy 'output_filename' is present, use it for both
        legacy = d.get("output_filename")
        output_brotli = str(d.get("output_brotli", legacy if legacy else "results_brotli.csv"))
        output_lz4    = str(d.get("output_lz4",    legacy if legacy else "results_lz4.csv"))

        return Paths(
            input_globs=[str(p) for p in input_globs],
            output_dir=str(d.get("output_dir", "./results")),
            output_brotli=output_brotli,
            output_lz4=output_lz4,
        )


    @staticmethod
    def _parse_brotli(d: Dict[str, Any]) -> BrotliCfg:
        quality = int(d.get("quality", 6))
        mode = str(d.get("mode", "generic")).lower()
        lgwin = d.get("lgwin", None)
        lgwin = int(lgwin) if lgwin is not None else None
        return BrotliCfg(quality=quality, mode=mode, lgwin=lgwin)

    @staticmethod
    def _parse_lz4(d: Dict[str, Any]) -> Lz4Cfg:
        level = int(d.get("compression_level", 1))
        return Lz4Cfg(compression_level=level)

    @staticmethod
    def _parse_timing(d: Dict[str, Any]) -> Timing:
        repeats = int(d.get("repeats", 5))
        warmup = int(d.get("warmup", 1))
        return Timing(repeats=repeats, warmup=warmup)

    def _validate(self) -> None:
        if not (0 <= self.brotli.quality <= 11):
            raise ValueError("brotli.quality must be in [0, 11].")
        if not (0 <= self.lz4.compression_level <= 16):
            raise ValueError("lz4.compression_level must be in [0, 16].")
        if self.brotli.mode not in _VALID_MODES:
            raise ValueError(f"brotli.mode must be one of {_VALID_MODES}.")
        if self.brotli.lgwin is not None and not (10 <= self.brotli.lgwin <= 24):
            raise ValueError("brotli.lgwin must be in [10, 24] or null.")
        if self.timing.repeats < 1:
            raise ValueError("timing.repeats must be >= 1.")
        if self.timing.warmup < 0:
            raise ValueError("timing.warmup must be >= 0.")
        

    # ---- helpers used by your main/bench code ----
    def to_bench_args(self) -> SimpleNamespace:
        """Flattened args object consumed by bench.run_one()."""
        return SimpleNamespace(
            brotli_q=self.brotli.quality,
            brotli_mode=self.brotli.mode,
            brotli_lgwin=self.brotli.lgwin,
            lz4_level=self.lz4.compression_level,
            repeats=self.timing.repeats,
            warmup=self.timing.warmup,
        )


    def output_csv_path_for(self, alg: str) -> Path:
        return self.paths.output_path_for(alg)

