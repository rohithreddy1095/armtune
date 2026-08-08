import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from armtune.runner import MockRunner, Config, QUANTS
from armtune.search import build_space, successive_halving, TuneLog
from armtune.sysinfo import SystemInfo, detect
from armtune.report import emit_deploy, render_report


def make_si(cores=8, mem=16.0):
    si = SystemInfo(arch="aarch64", is_arm64=True, cpu_model="test", physical_cores=cores,
                    logical_cores=cores, mem_gb=mem)
    si.features = {f: {"present": True, "why": ""} for f in
                   ("asimd", "asimddp", "i8mm", "sve", "sve2", "bf16", "fphp")}
    return si


def test_detect_runs():
    si = detect()
    assert si.arch
    assert si.logical_cores >= 1


def test_space_prunes_by_memory():
    si = make_si(mem=6.0)  # 8B F16 = 16GB -> must be pruned
    space = build_space(si, model_params_b=8.0)
    quants = {c.quant for c in space}
    assert "F16" not in quants
    assert "Q4_0" in quants


def test_quant_beats_f16_on_generation():
    r = MockRunner(physical_cores=8, mem_bw_gbs=60, model_params_b=8)
    q4 = r.run(Config("Q4_K_M", 8))
    f16 = r.run(Config("F16", 8))
    assert q4.tg_tps > f16.tg_tps


def test_oversubscription_penalty():
    r = MockRunner(physical_cores=8, mem_bw_gbs=60, model_params_b=8)
    ok = r.run(Config("Q4_0", 8))
    over = r.run(Config("Q4_0", 12))
    assert ok.tg_tps > over.tg_tps


def test_search_finds_sensible_winner():
    si = make_si()
    r = MockRunner(physical_cores=8, mem_bw_gbs=60, model_params_b=8)
    log = TuneLog()
    results = successive_halving(build_space(si, 8.0), r.run, rounds=2, log=log)
    best = results[0]
    assert best.config.quant in ("Q4_K_M", "Q4_0")
    assert best.config.threads <= 8
    assert len(log.rounds) == 3
    assert any("winner" in d.lower() for d in log.decisions)


def test_artifacts(tmp_path):
    si = make_si()
    r = MockRunner()
    log = TuneLog()
    results = successive_halving(build_space(si, 8.0), r.run, rounds=1, log=log)
    files = emit_deploy(results[0], 5.0, tmp_path / "deploy")
    assert all(f.exists() for f in files)
    dockerfile = (tmp_path / "deploy" / "Dockerfile").read_text()
    assert "linux/arm64" in dockerfile
    assert f'"-t", "{results[0].config.threads}"' in dockerfile
    md = render_report(si, results, log, "test-model", mock=True)
    assert "Winner" in md and "Leaderboard" in md


def test_cli_end_to_end(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "armtune.cli", "tune", "--mock", "--out", str(out)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "report.md").exists()
    assert (out / "deploy" / "Dockerfile").exists()
    data = json.loads((out / "results.json").read_text())
    assert data["results"], "expected benchmark results"
