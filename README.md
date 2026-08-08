# armtune

**Agentic auto-tuner for LLM inference on Arm64.** Point it at an Arm machine and a GGUF
model; it detects the silicon, plans a hardware-aware search space, benchmarks its way to
the fastest serving config, and ships the result as a ready-to-deploy Docker image +
Helm values pinned to that config.

Built for the **Arm Create: AI Optimization Challenge 2026** (Cloud AI track).

## Why

The same llama.cpp binary can vary 2-4x in throughput on the same Arm host depending on
quantization, thread count, batch size, and kernel features (NEON dotprod, i8mm, SVE).
Most teams guess. armtune measures - cheaply - using a successive-halving agent loop:

```
plan (prune space from CPU features + RAM)
  -> act (short llama-bench runs on every candidate)
  -> observe & prune (keep top 40%)
  -> act (longer runs on survivors)
  -> ship (Dockerfile + Helm values for the winner)
```

## Quick start

```bash
pip install -e .

# What silicon am I on?
armtune detect

# Real tuning on any Arm64 host (Graviton, Axion, Ampere, Apple Silicon, Raspberry Pi)
# Requires llama.cpp's llama-bench on PATH and a .gguf model:
armtune tune --model llama-3.1-8b.Q4_K_M.gguf --params 8 --out results/

# No Arm hardware handy? Development/demo mode with a calibrated analytical model:
armtune tune --mock --params 8 --mem-bw 100 --out results/
```

Outputs:

- `report.md` - host capabilities, leaderboard, agent decision log
- `results.json` - full benchmark data for every round
- `deploy/Dockerfile` - arm64 llama.cpp server image built with `-mcpu=native`
- `deploy/values.yaml` - Helm values with the winning threads/batch/quant and an
  `arm64` nodeSelector

See [`examples/mock-dev-host-8B/`](examples/mock-dev-host-8B/) for a sample run.

## How the search works

`build_space()` prunes before running anything: quants that don't fit in 60% of RAM are
dropped, thread candidates are derived from physical core count (memory-bandwidth-bound
generation rarely benefits from oversubscription), and flash-attention variants are only
added where they matter. `successive_halving()` then spends its benchmark budget like a
bandit: cheap 64/32-token runs across the whole space, full 256/128-token runs only on
the top 40%.

Scoring is `0.7 * generation + 0.3 * prompt` tokens/sec - weighted for interactive
serving; adjust in `runner.py` for RAG-heavy workloads.

## Mock mode

`MockRunner` is an analytical model of Arm64 inference calibrated to the published
behaviour of llama.cpp on Neoverse cores: generation throughput = memory bandwidth /
model bytes, int-quant kernel bonus when dotprod/i8mm are present, thread scaling with a
bandwidth knee, oversubscription penalty. It exists so the agent logic is testable and
demoable anywhere - all real numbers come from `llama-bench`.

## Tests

```bash
python -m pytest tests/ -q
```

## Roadmap

- ONNX Runtime + KleidiAI runner
- Multi-instance throughput mode (batch serving)
- Cost/perf mode: $/1M tokens across Graviton instance types

## License

Apache-2.0
