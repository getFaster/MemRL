from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class TrainConfig:
    exp_name: str = "retrieval_ppo_atari_jax"
    seed: int = 1
    env_id: str = "Frostbite-v5"
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    retrieval_mode: Literal["none", "random", "learned"] = "none"
    memory_capacity: int = 100_000
    memory_dim: int = 256
    retrieval_k: int = 64
    temperature: float = 0.1
    diagnostics_interval: int = 10
    diagnostics_top_k: int = 8

    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = "memrl-frostbite"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    capture_video: bool = False
    output_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_interval: int = 1_000
    checkpointing: bool = True
    save_memory_frames: bool = True
    retain_periodic_checkpoints: int = 2
    resume_from: Path | None = None
    benchmark_run: bool = False
    xla_memory_fraction: float = 0.55

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return self.total_timesteps // self.batch_size

    def validate(self) -> None:
        if self.retrieval_mode not in {"none", "random", "learned"}:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")
        if self.num_envs < 1 or self.num_steps < 1:
            raise ValueError("num_envs and num_steps must be positive")
        if self.num_minibatches < 1 or self.update_epochs < 1:
            raise ValueError("num_minibatches and update_epochs must be positive")
        if self.total_timesteps < self.batch_size:
            raise ValueError("total_timesteps must be at least num_envs * num_steps")
        if self.batch_size % self.num_minibatches:
            raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
        if self.memory_capacity < 1 or self.memory_dim < 1 or self.retrieval_k < 1:
            raise ValueError("memory capacity, dimension, and retrieval K must be positive")
        if self.retrieval_k > self.memory_capacity:
            raise ValueError("retrieval K cannot exceed memory capacity")
        if self.diagnostics_interval < 1 or self.diagnostics_top_k < 1:
            raise ValueError("diagnostics interval and top K must be positive")
        if self.diagnostics_top_k > self.retrieval_k:
            raise ValueError("diagnostics top K cannot exceed retrieval K")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.target_kl is not None:
            raise ValueError("target_kl early stopping is incompatible with the fixed four-epoch compiled PPO gate")
        if not 0 < self.xla_memory_fraction <= 1:
            raise ValueError("xla_memory_fraction must be in (0, 1]")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.retain_periodic_checkpoints < 1:
            raise ValueError("retain_periodic_checkpoints must be positive")
        if self.benchmark_run:
            if self.total_timesteps != 200_000:
                raise ValueError("benchmark runs must use the locked 200,000-step target")
            if self.num_envs != 8 or self.num_steps != 128 or self.memory_capacity != 100_000:
                raise ValueError("benchmark runs must use the locked rollout and memory dimensions")
            if self.wandb_mode != "disabled" or self.checkpointing:
                raise ValueError("benchmark runs require disabled W&B and checkpointing")

    def to_dict(self) -> dict:
        values = dataclasses.asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
