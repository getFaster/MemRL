"""Evaluation from one rewritten Orbax checkpoint directory."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tyro


@dataclass
class EvalConfig:
    checkpoint: Path
    episodes: int = 10
    seed: int = 10_001
    num_envs: int = 8
    deterministic_actions: bool = False
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = "memrl-frostbite-eval"
    output_dir: Path = Path("runs/evaluation")
    xla_memory_fraction: float = 0.55


def evaluate(config: EvalConfig) -> dict[str, Any]:
    if config.episodes < 1 or config.num_envs < 1:
        raise ValueError("episodes and num_envs must be positive")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.xla_memory_fraction))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from memrl.checkpointing import restore_checkpoint
    from memrl.envs import initial_episode_statistics, make_envpool, update_episode_statistics
    from memrl.memory import DeviceMemoryState, evaluation_retrieval, sample_batch
    from memrl.models import RetrievalAgent

    bundle = restore_checkpoint(config.checkpoint)
    train_config = bundle.metadata["config"]
    mode = train_config["retrieval_mode"]
    retrieval_k = int(train_config["retrieval_k"])
    temperature = float(train_config["temperature"])
    action_dim = int(bundle.metadata["action_dim"])
    memory_dim = 512 if mode == "none" else 512 + action_dim + 1
    if mode != "none" and train_config.get("memory_dim") != memory_dim:
        raise ValueError(f"retrieval checkpoint requires transition memory width {memory_dim}")
    state_payload = bundle.training["state"]
    params = state_payload.params if hasattr(state_payload, "params") else state_payload["params"]
    agent = RetrievalAgent(action_dim=action_dim, retrieval_mode=mode, temperature=temperature)

    memory = None
    whole_memory = None
    if mode != "none":
        raw_memory = bundle.training["memory"]
        memory = raw_memory if isinstance(raw_memory, DeviceMemoryState) else DeviceMemoryState(**raw_memory)
        size = int(memory.size)
        if size == 0:
            raise ValueError("retrieval checkpoint memory is empty")
        oldest = int(memory.next_index) if size == memory.capacity else 0
        physical = (oldest + np.arange(size, dtype=np.int32)) % memory.capacity
        whole_memory = memory.embeddings[jnp.asarray(physical)]

    env = make_envpool(train_config["env_id"], num_envs=config.num_envs, seed=config.seed)
    handle, _recv, _send, step_env = env.xla()
    obs, _ = env.reset()
    obs = jnp.asarray(obs)
    action_key = jax.random.PRNGKey(config.seed)
    retrieval_key = jax.random.PRNGKey(config.seed + 1)
    episode_statistics = initial_episode_statistics(config.num_envs)
    returns: list[float] = []
    lengths: list[int] = []
    retrieval_entropies: list[float] = []
    episode_rows: list[dict[str, int | float | str]] = []
    global_step = 0

    @jax.jit
    def evaluation_step(env_handle, observation, stats, policy_key, retrieval_rng):
        entropy = jnp.asarray(0.0)
        if mode == "none":
            output = agent.apply(params, observation)
            logits = output.logits
        elif mode == "random":
            sample = sample_batch(memory, retrieval_rng, config.num_envs, retrieval_k)
            retrieval_rng = sample.key
            output = agent.apply(params, observation, sample.embeddings)
            logits = output.logits
        else:
            dummy = jnp.zeros((config.num_envs, retrieval_k, memory_dim), dtype=jnp.float32)
            preliminary = agent.apply(params, observation, dummy)
            retrieval_rng, sample_key = jax.random.split(retrieval_rng)
            retrieved = evaluation_retrieval(sample_key, preliminary.query, whole_memory, retrieval_k, temperature)
            logits = agent.apply(params, observation, retrieved.context, method=agent.apply_retrieved_context)[0]
            entropy = -jnp.sum(
                retrieved.probabilities * jnp.log(jnp.maximum(retrieved.probabilities, 1e-8)), axis=-1
            ).mean()
        policy_key, sample_key = jax.random.split(policy_key)
        action = (
            jnp.argmax(logits, axis=-1)
            if config.deterministic_actions
            else jax.random.categorical(sample_key, logits, axis=-1)
        ).astype(jnp.int32)
        env_handle, (observation, _reward, _terminated, truncated, info) = step_env(env_handle, action)
        stats, events = update_episode_statistics(stats, info, truncated)
        return env_handle, observation, stats, policy_key, retrieval_rng, events, entropy

    run = None
    if config.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=config.wandb_project,
            mode=config.wandb_mode,
            name=f"eval__{mode}__{int(time.time())}",
            config={
                **{key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()},
                "training_config": train_config,
            },
        )

    try:
        while len(returns) < config.episodes:
            handle, obs, episode_statistics, action_key, retrieval_key, events, entropy = evaluation_step(
                handle, obs, episode_statistics, action_key, retrieval_key
            )
            event_mask, event_returns, event_lengths, host_entropy = jax.device_get(
                (events.mask, events.returns, events.lengths, entropy)
            )
            global_step += config.num_envs
            if mode == "learned":
                retrieval_entropies.append(float(host_entropy))
            for env_slot in np.flatnonzero(event_mask):
                episode_return = float(event_returns[env_slot])
                episode_length = int(event_lengths[env_slot])
                returns.append(episode_return)
                lengths.append(episode_length)
                episode_rows.append(
                    {
                        "mode": mode,
                        "seed": config.seed,
                        "env_slot": int(env_slot),
                        "completion_step": global_step,
                        "raw_return": episode_return,
                        "length": episode_length,
                    }
                )
                if run is not None:
                    run.log(
                        {"eval/episodic_return": episode_return, "eval/episodic_length": episode_length},
                        step=global_step,
                    )
                if len(returns) >= config.episodes:
                    break
    finally:
        env.close()
        if run is not None:
            run.finish()

    result = {
        "training_mode": mode,
        "training_seed": int(train_config["seed"]),
        "evaluation_seed": config.seed,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes": float(len(returns)),
        "diagnostics_frame_coverage": bundle.frame_coverage,
    }
    if retrieval_entropies:
        result["mean_whole_memory_entropy"] = float(np.mean(retrieval_entropies))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (config.output_dir / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episode_rows))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    evaluate(tyro.cli(EvalConfig))


if __name__ == "__main__":
    main()
