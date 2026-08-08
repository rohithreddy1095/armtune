"""Benchmark runners.

A runner takes a Config and returns a Result (tokens/sec for prompt processing
and generation). Two implementations:

* LlamaBenchRunner - shells out to llama.cpp's llama-bench (real numbers).
* MockRunner       - deterministic analytical model of Arm64 inference behaviour,
                     used for development, CI, and demos on non-Arm hosts.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional

QUANTS = ["Q4_0", "Q4_K_M", "Q8_0", "F16"]

# Rough bytes-per-weight for memory estimates
QUANT_BPW = {"Q4_0": 0.56, "Q4_K_M": 0.60, "Q8_0": 1.06, "F16": 2.0}


@dataclass(frozen=True)
class Config:
    quant: str
    threads: int
    batch: int = 512
    flash_attn: bool = False

    def label(self) -> str:
        fa = "+fa" if self.flash_attn else ""
        return f"{self.quant} t{self.threads} b{self.batch}{fa}"


@dataclass
class Result:
    config: Config
    pp_tps: float  # prompt processing tokens/sec
    tg_tps: float  # text generation tokens/sec
    runtime_s: float
    ok: bool = True
    error: str = ""

    def score(self) -> float:
        # Generation speed dominates interactive serving; prompt speed matters for RAG.
        return 0.7 * self.tg_tps + 0.3 * self.pp_tps

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score(), 2)
        return d


class LlamaBenchRunner:
    """Real benchmarks via llama.cpp llama-bench."""

    def __init__(self, model_path: str, llama_bench: str = "llama-bench"):
        self.model_path = model_path
        self.llama_bench = llama_bench

    def available(self) -> bool:
        return shutil.which(self.llama_bench) is not None

    def run(self, cfg: Config, n_prompt: int = 128, n_gen: int = 64) -> Result:
        import time

        cmd = [
            self.llama_bench,
            "-m", self.model_path,
            "-t", str(cfg.threads),
            "-b", str(cfg.batch),
            "-p", str(n_prompt),
            "-n", str(n_gen),
            "-o", "json",
        ]
        if cfg.flash_attn:
            cmd += ["-fa", "1"]
        t0 = time.time()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            rows = json.loads(out.stdout)
            pp = next((r["avg_ts"] for r in rows if r.get("n_prompt", 0) > 0), 0.0)
            tg = next((r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0), 0.0)
            return Result(cfg, round(pp, 2), round(tg, 2), round(time.time() - t0, 1))
        except Exception as e:  # noqa: BLE001
            return Result(cfg, 0.0, 0.0, round(time.time() - t0, 1), ok=False, error=str(e))


class MockRunner:
    """Analytical model calibrated on published Arm64 llama.cpp numbers.

    Captures the real shape of the tuning landscape so the search logic can be
    developed and demonstrated anywhere:
    - int4 quants beat F16 on memory-bound generation
    - Q4_K_M slightly beats Q4_0 with dotprod/i8mm kernels
    - throughput scales with threads until memory bandwidth saturates
    - oversubscribing logical cores hurts
    """

    def __init__(self, physical_cores: int = 8, mem_bw_gbs: float = 60.0,
                 model_params_b: float = 8.0, has_dotprod: bool = True):
        self.cores = physical_cores
        self.bw = mem_bw_gbs
        self.params = model_params_b
        self.dotprod = has_dotprod

    def available(self) -> bool:
        return True

    def run(self, cfg: Config, n_prompt: int = 128, n_gen: int = 64) -> Result:
        bpw = QUANT_BPW[cfg.quant]
        model_gb = self.params * bpw
        # Generation is memory-bandwidth bound: one full weight read per token.
        peak_tg = self.bw / model_gb
        # Thread scaling: linear-ish to a bandwidth knee, flat after physical cores.
        eff = min(cfg.threads, self.cores)
        knee = max(2, int(self.cores * 0.75))
        scale = min(1.0, 0.15 + 0.85 * eff / knee)
        oversub = 0.85 if cfg.threads > self.cores else 1.0
        kernel = 1.0
        if cfg.quant in ("Q4_0", "Q4_K_M", "Q8_0"):
            kernel = 1.25 if self.dotprod else 0.9
            if cfg.quant == "Q4_K_M" and self.dotprod:
                kernel *= 1.06
        tg = peak_tg * scale * oversub * kernel
        # Prompt processing is compute bound: scales better with cores & batch.
        pp = tg * (6.0 + 2.0 * min(cfg.batch, 1024) / 1024) * (eff / self.cores)
        if cfg.flash_attn:
            pp *= 1.08
            tg *= 1.02
        return Result(cfg, round(pp, 2), round(tg, 2), runtime_s=0.01)
