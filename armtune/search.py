"""Agentic tuning loop: plan -> act (benchmark) -> observe -> prune -> repeat.

Successive-halving search over quant x threads x batch x flash_attn.
Cheap short runs first over the full space, keep the top fraction, re-run the
survivors with longer, more reliable benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .runner import QUANTS, QUANT_BPW, Config, Result
from .sysinfo import SystemInfo


@dataclass
class TuneLog:
    rounds: List[dict] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.decisions.append(msg)


def build_space(si: SystemInfo, model_params_b: float, quants: Optional[List[str]] = None) -> List[Config]:
    """Plan step: prune the config space using hardware knowledge before running anything."""
    quants = quants or list(QUANTS)
    space: List[Config] = []
    # Memory guard: skip quants whose weights alone exceed 60% of RAM.
    viable = []
    for q in quants:
        need = model_params_b * QUANT_BPW[q]
        if si.mem_gb and need > si.mem_gb * 0.6:
            continue
        viable.append(q)
    if not viable:
        viable = ["Q4_0"]

    pc = max(1, si.physical_cores)
    thread_opts = sorted({max(1, pc // 2), max(1, int(pc * 0.75)), pc, min(si.logical_cores, pc + 2)})
    batch_opts = [256, 512]
    for q in viable:
        for t in thread_opts:
            for b in batch_opts:
                space.append(Config(quant=q, threads=t, batch=b))
    # flash-attn variant only for the largest batch (where it matters)
    space += [Config(quant=q, threads=pc, batch=512, flash_attn=True) for q in viable]
    return space


def successive_halving(
    space: List[Config],
    run: Callable[[Config, int, int], Result],
    rounds: int = 2,
    keep: float = 0.4,
    log: Optional[TuneLog] = None,
) -> List[Result]:
    log = log if log is not None else TuneLog()
    budgets = [(64, 32), (256, 128), (512, 256)][:rounds + 1]
    candidates = list(space)
    results: List[Result] = []
    for rnd, (n_prompt, n_gen) in enumerate(budgets):
        results = [run(c, n_prompt, n_gen) for c in candidates]
        results = [r for r in results if r.ok]
        results.sort(key=lambda r: r.score(), reverse=True)
        log.rounds.append({
            "round": rnd,
            "budget": {"n_prompt": n_prompt, "n_gen": n_gen},
            "evaluated": len(candidates),
            "results": [r.to_dict() for r in results],
        })
        if rnd < len(budgets) - 1:
            n_keep = max(2, int(len(results) * keep))
            log.note(
                f"Round {rnd}: evaluated {len(candidates)} configs at budget "
                f"pp={n_prompt}/tg={n_gen}; keeping top {n_keep} "
                f"(best: {results[0].config.label()} @ {results[0].score():.1f})"
            )
            candidates = [r.config for r in results[:n_keep]]
    if results:
        log.note(f"Final winner: {results[0].config.label()} "
                 f"(gen {results[0].tg_tps} tok/s, prompt {results[0].pp_tps} tok/s)")
    return results
