"""Run real EnvPool smokes in all modes and two learned PPO iterations."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import tyro

from memrl.checkpointing import restore_checkpoint
from memrl.config import TrainConfig
from memrl.train import train


@dataclass
class SmokeConfig:
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    wandb_project: str = "memrl-frostbite-smoke"
    output_dir: Path = Path("runs/smoke")
    checkpoint_dir: Path = Path("checkpoints/smoke")
    num_envs: int = 8
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
        save_memory_frames=True,
        wandb_mode=config.wandb_mode,
        wandb_project=config.wandb_project,
        output_dir=config.output_dir,
        checkpoint_dir=config.checkpoint_dir,
        seed=config.seed,
    )
    completed = {}
    for offset, mode in enumerate(("none", "random", "learned")):
        iterations = 2 if mode == "learned" else 1
        checkpoint = train(
            replace(
                base,
                retrieval_mode=mode,
                # Legacy baselines may retain this unused retrieval setting.
                memory_dim=256 if mode == "none" else None,
                seed=config.seed + offset,
                total_timesteps=iterations * config.num_envs * config.num_steps,
            )
        )
        bundle = restore_checkpoint(checkpoint)
        if int(bundle.metadata["completed_rollouts"]) != iterations:
            raise RuntimeError(f"{mode} did not complete the expected rollout count")
        if mode == "learned" and bundle.frame_coverage <= 0:
            raise RuntimeError("learned smoke checkpoint did not preserve diagnostic frames")
        completed[mode] = {
            "checkpoint": str(checkpoint),
            "rollouts": iterations,
            "frame_coverage": bundle.frame_coverage,
        }

    frame_less_initial = train(
        replace(
            base,
            retrieval_mode="learned",
            seed=config.seed + 3,
            save_memory_frames=False,
            total_timesteps=config.num_envs * config.num_steps,
        )
    )
    initial_bundle = restore_checkpoint(frame_less_initial)
    if initial_bundle.frame_coverage != 0.0:
        raise RuntimeError("frame-less checkpoint unexpectedly contains diagnostic frames")
    resumed_checkpoint = train(
        replace(
            base,
            retrieval_mode="learned",
            seed=config.seed + 3,
            save_memory_frames=False,
            total_timesteps=2 * config.num_envs * config.num_steps,
            resume_from=frame_less_initial,
        )
    )
    resumed_bundle = restore_checkpoint(resumed_checkpoint)
    if int(resumed_bundle.metadata["completed_rollouts"]) != 2:
        raise RuntimeError("frame-less recovery did not continue training")
    history = resumed_bundle.metadata.get("resume_history", [])
    if not history or history[-1].get("environment_reset_discontinuity") is not True:
        raise RuntimeError("frame-less recovery did not record the environment reset discontinuity")
    metrics_path = config.output_dir / resumed_checkpoint.parent.name / "metrics.jsonl"
    final_metrics = json.loads(metrics_path.read_text().splitlines()[-1])
    coverage = float(final_metrics["diagnostics/frame_coverage"])
    if not 0.0 < coverage < 1.0:
        raise RuntimeError(f"frame-less recovery should report incomplete frame coverage, got {coverage}")
    completed["learned_frame_less_resume"] = {
        "checkpoint": str(resumed_checkpoint),
        "rollouts": 2,
        "frame_coverage": coverage,
    }
    print(json.dumps({"smoke_test": "passed", "modes": completed}, indent=2))


def main() -> None:
    run_smoke(tyro.cli(SmokeConfig))


if __name__ == "__main__":
    main()
