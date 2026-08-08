"""Hardware detection for Arm64 (and everything else, honestly)."""
from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass, field, asdict

ARM_FEATURES = [
    # feature flag -> why it matters for inference
    ("asimd", "NEON SIMD (baseline Arm64 vector math)"),
    ("asimddp", "SDOT/UDOT int8 dot product (big int4/int8 quant speedup)"),
    ("i8mm", "Int8 matrix multiply (KleidiAI / smmla kernels)"),
    ("sve", "Scalable Vector Extension"),
    ("sve2", "SVE2 (Graviton3+/Axion)"),
    ("bf16", "BFloat16 arithmetic"),
    ("fphp", "FP16 arithmetic"),
]


@dataclass
class SystemInfo:
    arch: str = ""
    is_arm64: bool = False
    cpu_model: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    mem_gb: float = 0.0
    features: dict = field(default_factory=dict)
    platform_hint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _read_cpuinfo() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return f.read()
    except OSError:
        return ""


def _mem_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(re.search(r"(\d+)", line).group(1))
                    return round(kb / 1024 / 1024, 1)
    except OSError:
        pass
    # macOS fallback
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return round(int(out) / 1024**3, 1)
    except Exception:
        return 0.0


def _platform_hint(cpu_model: str, features: dict) -> str:
    m = cpu_model.lower()
    if "apple" in m or platform.system() == "Darwin":
        return "Apple Silicon (macOS)"
    if features.get("sve2"):
        return "Neoverse V2-class (e.g. AWS Graviton4, NVIDIA Grace, Google Axion)"
    if features.get("sve"):
        return "Neoverse V1-class (e.g. AWS Graviton3)"
    if features.get("asimddp"):
        return "Neoverse N1-class (e.g. AWS Graviton2, Ampere Altra) or newer"
    return "Generic"


def detect() -> SystemInfo:
    arch = platform.machine().lower()
    is_arm64 = arch in ("arm64", "aarch64")
    info = _read_cpuinfo()

    cpu_model = ""
    for pat in (r"model name\s*:\s*(.+)", r"Hardware\s*:\s*(.+)"):
        m = re.search(pat, info)
        if m:
            cpu_model = m.group(1).strip()
            break
    if not cpu_model and platform.system() == "Darwin":
        try:
            import subprocess

            cpu_model = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            cpu_model = platform.processor() or "unknown"
    cpu_model = cpu_model or platform.processor() or "unknown"

    flags = ""
    m = re.search(r"(?:Features|flags)\s*:\s*(.+)", info)
    if m:
        flags = m.group(1)
    feature_map = {}
    for feat, why in ARM_FEATURES:
        present = bool(re.search(rf"\b{feat}\b", flags))
        if platform.system() == "Darwin" and is_arm64 and feat in ("asimd", "asimddp", "fphp", "i8mm"):
            present = True  # Apple Silicon M1+ has these
        feature_map[feat] = {"present": present, "why": why}

    logical = os.cpu_count() or 1
    physical = logical
    try:
        cores = set()
        for block in info.split("\n\n"):
            pid = re.search(r"physical id\s*:\s*(\d+)", block)
            cid = re.search(r"core id\s*:\s*(\d+)", block)
            if pid and cid:
                cores.add((pid.group(1), cid.group(1)))
        if cores:
            physical = len(cores)
    except Exception:
        pass

    si = SystemInfo(
        arch=arch,
        is_arm64=is_arm64,
        cpu_model=cpu_model,
        physical_cores=physical,
        logical_cores=logical,
        mem_gb=_mem_gb(),
        features=feature_map,
    )
    si.platform_hint = _platform_hint(cpu_model, {k: v["present"] for k, v in feature_map.items()})
    return si


def summarize(si: SystemInfo) -> str:
    lines = [
        f"Architecture : {si.arch} ({'Arm64 - lets tune!' if si.is_arm64 else 'not Arm64 - mock/dev mode'})",
        f"CPU          : {si.cpu_model}",
        f"Cores        : {si.physical_cores} physical / {si.logical_cores} logical",
        f"Memory       : {si.mem_gb} GB",
        f"Platform     : {si.platform_hint}",
        "Features:",
    ]
    for feat, d in si.features.items():
        mark = "+" if d["present"] else "-"
        lines.append(f"  [{mark}] {feat:<8} {d['why']}")
    return "\n".join(lines)
