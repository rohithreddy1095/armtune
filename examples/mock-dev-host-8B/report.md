# armtune report - 8.0B model
_Generated 2026-08-08 03:49 UTC_

## Host
```
Architecture : x86_64 (not Arm64 - mock/dev mode)
CPU          : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
Cores        : 4 physical / 4 logical
Memory       : 3.8 GB
Platform     : Generic
Features:
  [-] asimd    NEON SIMD (baseline Arm64 vector math)
  [-] asimddp  SDOT/UDOT int8 dot product (big int4/int8 quant speedup)
  [-] i8mm     Int8 matrix multiply (KleidiAI / smmla kernels)
  [-] sve      Scalable Vector Extension
  [-] sve2     SVE2 (Graviton3+/Axion)
  [-] bf16     BFloat16 arithmetic
  [-] fphp     FP16 arithmetic
```

> **Mock mode** - numbers come from armtune's calibrated analytical model (non-Arm dev host). Run on Arm64 for real measurements.

## Winner

| | |
|---|---|
| Config | **Q4_0 t4 b512+fa** |
| Generation | **28.46 tok/s** |
| Prompt processing | **210.94 tok/s** |
| Speedup vs worst evaluated | **1.1x** |

## Leaderboard (final round)

| # | Config | Gen tok/s | Prompt tok/s | Score |
|---|--------|-----------|--------------|-------|
| 1 | Q4_0 t4 b512+fa | 28.46 | 210.94 | 83.2 |
| 2 | Q4_0 t4 b512 | 27.9 | 195.31 | 78.1 |

## Agent decisions

- Mock runner: 4 cores, 100.0 GB/s assumed bandwidth, 8.0B params.
- Planned search space: 7 configs after hardware-aware pruning.
- Round 0: evaluated 7 configs at budget pp=64/tg=32; keeping top 2 (best: Q4_0 t4 b512+fa @ 83.2)
- Round 1: evaluated 2 configs at budget pp=256/tg=128; keeping top 2 (best: Q4_0 t4 b512+fa @ 83.2)
- Final winner: Q4_0 t4 b512+fa (gen 28.46 tok/s, prompt 210.94 tok/s)

## Ship it

Deployment artifacts generated in `deploy/`:
- `Dockerfile` - arm64 llama.cpp server image built with `-mcpu=native`
- `values.yaml` - Helm values pinned to the winning config, `arm64` nodeSelector
