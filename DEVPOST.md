# Devpost submission - Arm Create: AI Optimization Challenge

**Project name:** armtune
**Track:** Cloud AI
**Elevator pitch (200 chars):**
An agent that benchmarks its way to the fastest LLM serving config on any Arm64 host -
then ships it as a ready-to-deploy Docker image and Helm chart. Stop guessing, start measuring.

## Inspiration

The same 8B model on the same Graviton instance can serve 2-4x faster or slower
depending on quantization, threads, batch size, and whether the build actually uses
dotprod/i8mm/SVE kernels. Most teams pick defaults and move on - leaving real
performance (and real money) on the table.

## What it does

armtune is an agentic auto-tuner. It detects the Arm silicon it's running on (NEON,
dotprod, i8mm, SVE/SVE2, core count, RAM), plans a hardware-aware search space, runs a
successive-halving benchmark loop via llama.cpp's llama-bench, and emits: a Markdown
report with a leaderboard and the agent's decision log, plus deployment artifacts -
an arm64 Dockerfile built with `-mcpu=native` and Helm values pinned to the winning
config with an arm64 nodeSelector.

## How we built it

Pure-Python agent core (plan -> act -> observe -> prune), pluggable runners (real:
llama-bench subprocess; mock: an analytical model of memory-bandwidth-bound generation
calibrated to Neoverse behaviour, for CI and non-Arm dev hosts), pytest suite covering
the search logic, memory pruning, oversubscription penalties, and CLI end-to-end.

## Challenges

Benchmark budget: exhaustive search is too slow to be practical, so we spend budget
like a bandit - short runs everywhere, long runs only on survivors. And honest
reporting: mock numbers are clearly labeled; only Arm hardware produces real ones.

## What's next

ONNX Runtime + KleidiAI runner, batch-serving throughput mode, and a cost mode that
recommends the cheapest Graviton instance type per 1M tokens.
