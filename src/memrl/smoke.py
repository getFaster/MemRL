"""Run one tiny PPO iteration in all three retrieval modes."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from memrl.config import TrainConfig
from memrl.train import train


@dataclass
class SmokeConfig:
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    wandb_project: str = "memrl-frostbite-smoke"
    output_dir: Path = Path("runs/smoke")
    checkpoint_dir: Path = Path("checkpoints/smoke")
    num_envs: int = 1
    num_steps: int = 8
    seed: int = 101


def run_smoke(config: SmokeConfig) -> None:
    base = TrainConfig(
        total_timesteps=config.num_envs * config.num_steps,
        num_envs=config.num_envs,
        num_steps=config.num_steps,
        num_minibatches=1,
        update_epochs=1,
        memory_capacity=128,
        retrieval_k=4,
        diagnostics_interval=1,
        diagnostics_top_k=2,
        checkpoint_interval=0,
        save_memory_frames=False,
        wandb_mode=config.wandb_mode,
        wandb_project=config.wandb_project,
        output_dir=config.output_dir,
        checkpoint_dir=config.checkpoint_dir,
        seed=config.seed,
    )
    completed = {}
    for offset, mode in enumerate(("none", "random", "learned")):
        checkpoint = train(replace(base, retrieval_mode=mode, seed=config.seed + offset))
        metadata = json.loads(checkpoint.with_suffix(".json").read_text())
        metrics_path = Path(metadata["config"]["output_dir"]) / metadata["run_name"] / "metrics.jsonl"
        metrics = json.loads(metrics_path.read_text().splitlines()[-1])
        numeric = [value for value in metrics.values() if isinstance(value, (int, float))]
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"{mode} produced non-finite metrics")
        completed[mode] = {"checkpoint": str(checkpoint), "SPS": metrics["charts/SPS"]}
    print(json.dumps({"smoke_test": "passed", "modes": completed}, indent=2))


def main() -> None:
    run_smoke(tyro.cli(SmokeConfig))


if __name__ == "__main__":
    main()
