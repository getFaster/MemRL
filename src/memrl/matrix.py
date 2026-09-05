"""Run the seed-matched experiment matrix with bounded process concurrency."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tyro

MAX_COMBINED_VRAM_BYTES = int(4.8 * 1024**3)
MIN_MEM_AVAILABLE_BYTES = int(0.4 * 1024**3)
MAX_SWAP_GROWTH_BYTES = 256 * 1024**2


@dataclass
class MatrixConfig:
    seeds: tuple[int, ...] = (1, 2, 3)
    total_timesteps: int = 10_000_000
    wandb_project: str = "memrl-frostbite"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    output_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    max_parallel: Literal[1, 2] = 1
    resource_report: Path | None = None
    timing_run: bool = False


def validate_concurrency_report(payload: dict[str, Any]) -> None:
    report = payload.get("concurrency_canary", payload)
    required = {
        "learned_processes",
        "full_host_frames",
        "frame_complete_checkpoint",
        "combined_peak_vram_bytes",
        "min_mem_available_bytes",
        "swap_growth_bytes",
    }
    missing = required - report.keys()
    if missing:
        raise ValueError(f"resource report missing canary fields: {sorted(missing)}")
    failures = []
    if int(report["learned_processes"]) < 2:
        failures.append("fewer than two concurrent learned processes")
    if not bool(report["full_host_frames"]):
        failures.append("host frames were not fully enabled")
    if not bool(report["frame_complete_checkpoint"]):
        failures.append("checkpoint was not frame-complete")
    if int(report["combined_peak_vram_bytes"]) > MAX_COMBINED_VRAM_BYTES:
        failures.append("combined peak VRAM exceeded 4.8 GiB")
    if int(report["min_mem_available_bytes"]) < MIN_MEM_AVAILABLE_BYTES:
        failures.append("MemAvailable fell below 2 GiB")
    if int(report["swap_growth_bytes"]) > MAX_SWAP_GROWTH_BYTES:
        failures.append("swap growth exceeded 256 MiB")
    if failures:
        raise ValueError("resource report does not permit max_parallel=2: " + "; ".join(failures))


def validate_config(config: MatrixConfig) -> None:
    if config.max_parallel not in (1, 2):
        raise ValueError("max_parallel must be 1 or 2")
    if config.timing_run and config.max_parallel != 1:
        raise ValueError("timing jobs must execute exclusively with max_parallel=1")
    if config.max_parallel == 2:
        if config.resource_report is None:
            raise ValueError("max_parallel=2 requires --resource-report from a passing concurrency canary")
        validate_concurrency_report(json.loads(config.resource_report.read_text()))


def build_commands(config: MatrixConfig) -> list[list[str]]:
    group = "frostbite-retrieval-ppo-10m"
    commands = []
    modes = ("none", "random", "learned")
    for seed_index, seed in enumerate(config.seeds):
        for offset in range(len(modes)):
            mode = modes[(seed_index + offset) % len(modes)]
            command = [
                sys.executable,
                "-m",
                "memrl.train",
                "--retrieval-mode",
                mode,
                "--seed",
                str(seed),
                "--total-timesteps",
                str(config.total_timesteps),
                "--wandb-project",
                config.wandb_project,
                "--wandb-mode",
                config.wandb_mode,
                "--wandb-group",
                group,
                "--output-dir",
                str(config.output_dir),
                "--checkpoint-dir",
                str(config.checkpoint_dir),
            ]
            if config.wandb_entity is not None:
                command.extend(("--wandb-entity", config.wandb_entity))
            commands.append(command)
    return commands


def run_commands(commands: list[list[str]], max_parallel: int) -> None:
    pending = list(commands)
    active: list[tuple[list[str], subprocess.Popen]] = []
    while pending or active:
        while pending and len(active) < max_parallel:
            command = pending.pop(0)
            active.append((command, subprocess.Popen(command)))
        command, process = active.pop(0)
        returncode = process.wait()
        if returncode:
            for _, other in active:
                other.terminate()
            for _, other in active:
                other.wait()
            raise subprocess.CalledProcessError(returncode, command)


def run_matrix(config: MatrixConfig) -> None:
    validate_config(config)
    run_commands(build_commands(config), config.max_parallel)


def main() -> None:
    run_matrix(tyro.cli(MatrixConfig))


if __name__ == "__main__":
    main()
