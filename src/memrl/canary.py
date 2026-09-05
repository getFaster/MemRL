"""Run and measure the two-process learning concurrency canary."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from memrl.checkpointing import restore_checkpoint


@dataclass
class CanaryConfig:
    seeds: tuple[int, int] = (901, 902)
    total_timesteps: int = 200_000
    output_dir: Path = Path("canary-runs")
    checkpoint_dir: Path = Path("canary-checkpoints")
    report: Path = Path("canary-resources.json")
    sample_interval: float = 1.0


def _meminfo_bytes(name: str) -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, unit = line.split()
        if key.removesuffix(":") == name:
            return int(value) * (1024 if unit == "kB" else 1)
    raise RuntimeError(f"/proc/meminfo does not contain {name}")


def _swap_used_bytes() -> int:
    return _meminfo_bytes("SwapTotal") - _meminfo_bytes("SwapFree")


def _gpu_used_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(int(line.strip()) * 1024**2 for line in result.stdout.splitlines() if line.strip())


def _checkpoint_for_seed(root: Path, seed: int) -> Path | None:
    candidates = sorted(root.glob(f"*__learned__seed{seed}__*/final"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _checkpoint_status(root: Path, seeds: tuple[int, int]) -> tuple[bool, dict[str, Any]]:
    statuses = []
    for seed in seeds:
        checkpoint = _checkpoint_for_seed(root, seed)
        if checkpoint is None:
            statuses.append({"seed": seed, "found": False})
            continue
        bundle = restore_checkpoint(checkpoint)
        statuses.append(
            {
                "seed": seed,
                "found": True,
                "frames_saved": bundle.metadata.get("frames_saved", False),
                "frame_coverage": bundle.frame_coverage,
            }
        )
    complete = all(
        item.get("found") and item.get("frames_saved") and item.get("frame_coverage") == 1.0 for item in statuses
    )
    return complete, {"checkpoints": statuses}


def run_canary(config: CanaryConfig) -> dict[str, Any]:
    if config.sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    if len(config.seeds) != 2 or config.seeds[0] == config.seeds[1]:
        raise ValueError("canary requires two distinct seeds")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline_swap = _swap_used_bytes()
    processes = []
    for seed in config.seeds:
        command = [
            sys.executable,
            "-m",
            "memrl.train",
            "--retrieval-mode",
            "learned",
            "--seed",
            str(seed),
            "--total-timesteps",
            str(config.total_timesteps),
            "--wandb-mode",
            "disabled",
            "--output-dir",
            str(config.output_dir),
            "--checkpoint-dir",
            str(config.checkpoint_dir),
        ]
        log = (config.output_dir / f"seed{seed}.log").open("w")
        processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))

    peak_vram = 0
    min_mem_available = _meminfo_bytes("MemAvailable")
    peak_swap_growth = 0
    try:
        while any(process.poll() is None for process, _ in processes):
            peak_vram = max(peak_vram, _gpu_used_bytes())
            min_mem_available = min(min_mem_available, _meminfo_bytes("MemAvailable"))
            peak_swap_growth = max(peak_swap_growth, _swap_used_bytes() - baseline_swap)
            time.sleep(config.sample_interval)
        peak_vram = max(peak_vram, _gpu_used_bytes())
        min_mem_available = min(min_mem_available, _meminfo_bytes("MemAvailable"))
        peak_swap_growth = max(peak_swap_growth, _swap_used_bytes() - baseline_swap)
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()

    return_codes = [process.returncode for process, _ in processes]
    frame_complete, checkpoint_details = _checkpoint_status(config.checkpoint_dir, config.seeds)
    payload = {
        "concurrency_canary": {
            "learned_processes": sum(code == 0 for code in return_codes),
            "full_host_frames": frame_complete,
            "frame_complete_checkpoint": frame_complete,
            "combined_peak_vram_bytes": peak_vram,
            "min_mem_available_bytes": min_mem_available,
            "swap_growth_bytes": max(0, peak_swap_growth),
            **checkpoint_details,
        }
    }
    config.report.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    config = tyro.cli(CanaryConfig)
    print(json.dumps(run_canary(config), indent=2))


if __name__ == "__main__":
    main()