"""armtune CLI: detect | tune"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import sysinfo
from .runner import LlamaBenchRunner, MockRunner
from .search import TuneLog, build_space, successive_halving
from .report import emit_deploy, render_report, save_json
from .runner import QUANT_BPW


def cmd_detect(_args) -> int:
    si = sysinfo.detect()
    print(sysinfo.summarize(si))
    return 0


def cmd_tune(args) -> int:
    si = sysinfo.detect()
    print(sysinfo.summarize(si))
    print()

    log = TuneLog()
    mock = args.mock or not si.is_arm64
    if mock and not args.mock:
        log.note("Host is not Arm64 - falling back to mock mode for development.")

    if mock:
        has_dp = si.features.get("asimddp", {}).get("present", True)
        runner = MockRunner(
            physical_cores=si.physical_cores or 8,
            mem_bw_gbs=args.mem_bw,
            model_params_b=args.params,
            has_dotprod=has_dp if si.is_arm64 else True,
        )
        log.note(f"Mock runner: {si.physical_cores} cores, {args.mem_bw} GB/s assumed bandwidth, "
                 f"{args.params}B params.")
    else:
        if not args.model:
            print("error: --model is required for real benchmarks", file=sys.stderr)
            return 2
        runner = LlamaBenchRunner(args.model, args.llama_bench)
        if not runner.available():
            print(f"error: '{args.llama_bench}' not found on PATH. Install llama.cpp "
                  "or pass --llama-bench /path/to/llama-bench", file=sys.stderr)
            return 2
        log.note(f"Real benchmarks via {args.llama_bench} on {args.model}")

    space = build_space(si, args.params)
    log.note(f"Planned search space: {len(space)} configs after hardware-aware pruning.")
    print(f"Tuning over {len(space)} configs...\n")

    results = successive_halving(space, runner.run, rounds=args.rounds, log=log)
    if not results:
        print("error: all benchmark runs failed", file=sys.stderr)
        return 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model_gb = args.params * QUANT_BPW[results[0].config.quant]
    emit_deploy(results[0], model_gb, outdir / "deploy")
    report = render_report(si, results, log, args.model or f"{args.params}B model", mock)
    (outdir / "report.md").write_text(report)
    save_json(si, results, log, outdir / "results.json")

    print(report)
    print(f"\nArtifacts written to {outdir}/ (report.md, results.json, deploy/)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="armtune",
                                description="Agentic auto-tuner for LLM inference on Arm64")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="Show hardware capabilities relevant to inference")
    d.set_defaults(fn=cmd_detect)

    t = sub.add_parser("tune", help="Search for the fastest serving config")
    t.add_argument("--model", help="Path to a .gguf model (real mode)")
    t.add_argument("--params", type=float, default=8.0, help="Model size in billions (default 8)")
    t.add_argument("--llama-bench", default="llama-bench", help="Path to llama-bench binary")
    t.add_argument("--rounds", type=int, default=2, help="Successive-halving rounds (default 2)")
    t.add_argument("--mock", action="store_true", help="Force mock mode")
    t.add_argument("--mem-bw", type=float, default=60.0,
                   help="Assumed memory bandwidth GB/s for mock mode (default 60)")
    t.add_argument("--out", default="armtune-out", help="Output directory")
    t.set_defaults(fn=cmd_tune)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
