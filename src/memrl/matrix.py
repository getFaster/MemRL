"""Run the seed-matched A/B/C experiment matrix sequentially."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class MatrixConfig:
    seeds: tuple[int, ...] = (1, 2, 3)
    total_timesteps: int = 10_000_000
    wandb_project: str = "memrl-frostbite"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    output_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")


def main() -> None:
    config = tyro.cli(MatrixConfig)
    group = "frostbite-retrieval-ppo-10m"
    for mode in ("none", "random", "learned"):
        for seed in config.seeds:
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
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
