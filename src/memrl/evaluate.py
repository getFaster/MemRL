"""Checkpoint evaluation, including exact whole-memory learned retrieval."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro


@dataclass
class EvalConfig:
    checkpoint: Path
    memory_checkpoint: Path | None = None
    episodes: int = 10
    seed: int = 10_001
    retrieval_k: int = 64
    temperature: float = 0.1
    deterministic_actions: bool = False
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = "memrl-frostbite-eval"
    output_dir: Path = Path("runs/evaluation")
    xla_memory_fraction: float = 0.55


def evaluate(config: EvalConfig) -> dict[str, float]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.xla_memory_fraction))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import flax.serialization
    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    import numpy as np

    from memrl.envs import extract_final_episode_stats, make_env
    from memrl.memory import EpisodicMemory, evaluation_retrieval
    from memrl.models import RetrievalAgent

    metadata_path = config.checkpoint.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    train_config = metadata["config"]
    mode = train_config["retrieval_mode"]
    env_id = train_config["env_id"]
    action_dim = int(metadata["action_dim"])
    observation_shape = tuple(metadata["observation_shape"])
    agent = RetrievalAgent(action_dim=action_dim, retrieval_mode=mode, temperature=config.temperature)
    dummy_obs = jnp.zeros((1, *observation_shape), dtype=jnp.uint8)
    if mode == "none":
        variables = agent.init(jax.random.PRNGKey(0), dummy_obs)
    else:
        dummy_candidates = jnp.zeros((1, config.retrieval_k, 256), dtype=jnp.float32)
        variables = agent.init(jax.random.PRNGKey(0), dummy_obs, dummy_candidates)
    params = flax.serialization.from_bytes(variables, config.checkpoint.read_bytes())

    memory = None
    all_memory = None
    if mode != "none":
        memory_path = config.memory_checkpoint or config.checkpoint.with_name("memory.npz")
        memory = EpisodicMemory.load(memory_path, seed=config.seed)
        all_memory = memory.all(include_frames=False).embeddings
        if len(memory) == 0:
            raise ValueError("retrieval checkpoint memory is empty")

    env = gym.vector.SyncVectorEnv(
        [make_env(env_id, config.seed, 0, False, "evaluation", config.output_dir)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs, _ = env.reset(seed=[config.seed])
    action_key = jax.random.PRNGKey(config.seed)
    retrieval_key = jax.random.PRNGKey(config.seed + 1)
    returns: list[float] = []
    lengths: list[int] = []
    retrieval_entropies: list[float] = []
    run = None
    if config.wandb_mode != "disabled":
        import wandb

        eval_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()}
        run = wandb.init(
            project=config.wandb_project,
            mode=config.wandb_mode,
            name=f"eval__{mode}__{int(time.time())}",
            config={**eval_config, "training_config": train_config},
        )

    try:
        while len(returns) < config.episodes:
            if mode == "none":
                output = agent.apply(params, obs)
                logits = output.logits
            elif mode == "random":
                assert memory is not None
                sample = memory.sample_candidates(config.retrieval_k, include_frames=False)
                candidates = sample.embeddings[None, ...]
                output = agent.apply(params, obs, candidates)
                logits = output.logits
            else:
                assert all_memory is not None
                # Obtain the current query, score every resident memory exactly,
                # sample K from p(i|k), then aggregate the sampled h by plain mean.
                dummy_candidates = jnp.zeros((1, config.retrieval_k, 256), dtype=jnp.float32)
                preliminary = agent.apply(params, obs, dummy_candidates)
                retrieval_key, sample_key = jax.random.split(retrieval_key)
                retrieved = evaluation_retrieval(
                    sample_key, preliminary.query, jnp.asarray(all_memory), config.retrieval_k, config.temperature
                )
                logits, _, _, _ = agent.apply(params, obs, retrieved.context, method=agent.apply_retrieved_context)
                probabilities = np.asarray(retrieved.probabilities)
                entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-8))).sum(axis=-1)
                retrieval_entropies.append(float(entropy.mean()))

            action_key, sample_key = jax.random.split(action_key)
            if config.deterministic_actions:
                action = jnp.argmax(logits, axis=-1)
            else:
                action = jax.random.categorical(sample_key, logits, axis=-1)
            obs, _, _, _, infos = env.step(np.asarray(action))
            for episode in extract_final_episode_stats(infos):
                returns.append(float(episode["return"]))
                lengths.append(int(episode["length"]))
                if run is not None:
                    run.log({"eval/episodic_return": episode["return"], "eval/episodic_length": episode["length"]})
                if len(returns) >= config.episodes:
                    break
    finally:
        env.close()
        if run is not None:
            run.finish()

    result = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes": float(len(returns)),
    }
    if retrieval_entropies:
        result["mean_whole_memory_entropy"] = float(np.mean(retrieval_entropies))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    evaluate(tyro.cli(EvalConfig))


if __name__ == "__main__":
    main()
