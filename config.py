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
    output_filename: str = "brotli_results.csv"

    def output_path(self) -> Path:
        outdir = Path(self.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir / self.output_filename

@dataclass
class BrotliCfg:
    quality: int = 6              # 0..11
    mode: str = "generic"         # generic|text|font
    lgwin: Optional[int] = None   # 10..24 or None

@dataclass
class Timing:
    repeats: int = 5              # >=1
    warmup: int = 1               # >=0

@dataclass
class Config:
    paths: Paths
    brotli: BrotliCfg = field(default_factory=BrotliCfg)
    timing: Timing = field(default_factory=Timing)

    # ---- loading & validation ----
    @staticmethod
    def load(path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")

        # apply defaults and coerce to dataclasses
        paths = Config._parse_paths(data.get("paths", {}))
        brotli = Config._parse_brotli(data.get("brotli", {}))
        timing = Config._parse_timing(data.get("timing", {}))
        cfg = Config(paths=paths, brotli=brotli, timing=timing)
        cfg._validate()
        return cfg

    @staticmethod
    def _parse_paths(d: Dict[str, Any]) -> Paths:
        input_globs = d.get("input_globs")
        if not input_globs or not isinstance(input_globs, list):
            raise ValueError("paths.input_globs must be a non-empty list of glob patterns.")
        return Paths(
            input_globs=[str(p) for p in input_globs],
            output_dir=str(d.get("output_dir", "./results")),
            output_filename=str(d.get("output_filename", "brotli_results.csv")),
        )

    @staticmethod
    def _parse_brotli(d: Dict[str, Any]) -> BrotliCfg:
        quality = int(d.get("quality", 6))
        mode = str(d.get("mode", "generic")).lower()
        lgwin = d.get("lgwin", None)
        lgwin = int(lgwin) if lgwin is not None else None
        return BrotliCfg(quality=quality, mode=mode, lgwin=lgwin)

    @staticmethod
    def _parse_timing(d: Dict[str, Any]) -> Timing:
        repeats = int(d.get("repeats", 5))
        warmup = int(d.get("warmup", 1))
        return Timing(repeats=repeats, warmup=warmup)

    def _validate(self) -> None:
        if not (0 <= self.brotli.quality <= 11):
            raise ValueError("brotli.quality must be in [0, 11].")
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
            repeats=self.timing.repeats,
            warmup=self.timing.warmup,
        )

    def output_csv_path(self) -> Path:
        return self.paths.output_path()
