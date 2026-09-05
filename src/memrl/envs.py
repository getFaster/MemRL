"""EnvPool construction and episode accounting for Atari rollouts.

Training and evaluation intentionally share this module so they cannot drift to
different preprocessing settings. EnvPool is imported lazily: pure contract
tests and checkpoint inspection do not require its native extension to load.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from numbers import Integral
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

FROSTBITE_ENV_ID = "Frostbite-v5"
FROSTBITE_ALIASES = frozenset(
    {
        FROSTBITE_ENV_ID,
        "ALE/Frostbite-v5",
        "FrostbiteNoFrameskip-v4",
        "ALE/FrostbiteNoFrameskip-v4",
    }
)
OBSERVATION_SHAPE = (4, 84, 84)


class EpisodeStatistics(NamedTuple):
    """Per-environment real-game accumulators kept in compiled rollout state."""

    returns: jax.Array
    lengths: jax.Array


class EpisodeEvents(NamedTuple):
    """Fixed-shape episode completions emitted by one EnvPool step."""

    mask: jax.Array
    returns: jax.Array
    lengths: jax.Array


def normalize_env_id(env_id: str) -> str:
    """Map every supported Frostbite spelling to EnvPool's canonical task ID."""

    return FROSTBITE_ENV_ID if env_id in FROSTBITE_ALIASES else env_id


def _environment_seeds(seed: int | Sequence[int], num_envs: int) -> list[int]:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if isinstance(seed, int):
        return [seed + index for index in range(num_envs)]
    seeds = [int(value) for value in seed]
    if len(seeds) != num_envs:
        raise ValueError(f"expected exactly {num_envs} environment seeds, got {len(seeds)}")
    return seeds


def envpool_config(env_id: str, num_envs: int = 8, seed: int | Sequence[int] = 1) -> dict[str, Any]:
    """Return the complete, locked EnvPool Atari preprocessing configuration."""

    return {
        "task_id": normalize_env_id(env_id),
        "env_type": "gym",
        "num_envs": num_envs,
        "batch_size": num_envs,
        "seed": _environment_seeds(seed, num_envs),
        "frame_skip": 4,
        "stack_num": 4,
        "noop_max": 30,
        "use_fire_reset": True,
        "reward_clip": True,
        "episodic_life": True,
        "img_height": 84,
        "img_width": 84,
        "gray_scale": True,
        "use_inter_area_resize": True,
        "repeat_action_probability": 0.0,
    }


def make_envpool(
    env_id: str = FROSTBITE_ENV_ID,
    *,
    num_envs: int = 8,
    seed: int | Sequence[int] = 1,
):
    """Create the sole environment backend used by training and evaluation."""

    envpool = import_module("envpool")
    config = envpool_config(env_id, num_envs=num_envs, seed=seed)
    task_id = config.pop("task_id")
    env = envpool.make(task_id, **config)

    # These conventional vector-environment attributes are useful to shared
    # policy initialization code and are absent on some older EnvPool versions.
    # EnvPool 1.2.5 exposes them as read-only properties, so never overwrite an
    # attribute that the native object already provides.
    compatibility_attributes = {
        "num_envs": num_envs,
        "single_action_space": env.action_space,
        "single_observation_space": env.observation_space,
        "is_vector_env": True,
    }
    for name, value in compatibility_attributes.items():
        if not hasattr(env, name):
            setattr(env, name, value)
    validate_envpool_contract(env, num_envs=num_envs)
    return env


def validate_envpool_contract(env: Any, *, num_envs: int) -> None:
    """Fail early if an EnvPool instance does not match the policy interface."""

    if getattr(env, "num_envs", None) != num_envs:
        raise ValueError(f"expected {num_envs} environments, got {getattr(env, 'num_envs', None)}")
    action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
    if action_space is None or not isinstance(getattr(action_space, "n", None), Integral):
        raise TypeError("EnvPool must expose a discrete action space")
    observation_space = getattr(env, "single_observation_space", getattr(env, "observation_space", None))
    observation_shape = tuple(getattr(observation_space, "shape", ()))
    if observation_shape != OBSERVATION_SHAPE:
        raise ValueError(f"expected EnvPool observations {OBSERVATION_SHAPE}, got {observation_shape}")
    if not callable(getattr(env, "xla", None)):
        raise TypeError("EnvPool backend must expose the XLA interface")


def validate_envpool_step(
    observation: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Mapping[str, Any],
    *,
    num_envs: int,
) -> None:
    """Validate one host-visible step, including raw-reward/game-end fields."""

    expected_observation_shape = (num_envs, *OBSERVATION_SHAPE)
    if np.shape(observation) != expected_observation_shape:
        raise ValueError(f"expected observation shape {expected_observation_shape}, got {np.shape(observation)}")
    for name, value in (("reward", reward), ("terminated", terminated), ("truncated", truncated)):
        if np.shape(value) != (num_envs,):
            raise ValueError(f"expected {name} shape ({num_envs},), got {np.shape(value)}")
    for key in ("reward", "terminated"):
        if key not in info:
            raise KeyError(f"EnvPool step info is missing {key!r}")
        if np.shape(info[key]) != (num_envs,):
            raise ValueError(f"expected info[{key!r}] shape ({num_envs},), got {np.shape(info[key])}")


def initial_episode_statistics(num_envs: int) -> EpisodeStatistics:
    """Create empty real-game accumulators on the JAX device."""

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    return EpisodeStatistics(
        returns=jnp.zeros((num_envs,), dtype=jnp.float32),
        lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
    )


def real_episode_done(info: Mapping[str, Any], truncated: Any) -> jax.Array:
    """Return true game-over/time-limit boundaries, excluding life loss."""

    # With EnvPool 1.2.5's Gymnasium contract, returned ``terminated`` includes
    # episodic-life boundaries while info["terminated"] is true only on actual
    # game over. Time-limit truncation is returned separately from info.
    return jnp.logical_or(jnp.asarray(info["terminated"]), jnp.asarray(truncated))


def update_episode_statistics(
    statistics: EpisodeStatistics, info: Mapping[str, Any], truncated: Any
) -> tuple[EpisodeStatistics, EpisodeEvents]:
    """Accumulate raw EnvPool rewards and emit exact real-game completions.

    EnvPool's returned reward can be sign-clipped and its ``done`` can represent
    episodic-life boundaries. The raw ``info['reward']`` and real termination
    fields are therefore deliberately the only inputs used here.
    """

    raw_reward = jnp.asarray(info["reward"], dtype=jnp.float32)
    completed = real_episode_done(info, truncated)
    episode_returns = statistics.returns + raw_reward
    # EnvPool performs an auto-reset on the call after a real terminal state;
    # that call reports elapsed_step == 0 and discards the supplied action.
    # Do not count it as a game action. Synthetic callers without this native
    # field retain the ordinary one-call/one-step convention.
    if "elapsed_step" in info:
        length_increment = (jnp.asarray(info["elapsed_step"]) > 0).astype(jnp.int32)
    else:
        length_increment = jnp.ones_like(statistics.lengths)
    episode_lengths = statistics.lengths + length_increment
    events = EpisodeEvents(mask=completed, returns=episode_returns, lengths=episode_lengths)
    next_statistics = EpisodeStatistics(
        returns=jnp.where(completed, jnp.zeros_like(episode_returns), episode_returns),
        lengths=jnp.where(completed, jnp.zeros_like(episode_lengths), episode_lengths),
    )
    return next_statistics, events


__all__ = [
    "EpisodeEvents",
    "EpisodeStatistics",
    "FROSTBITE_ALIASES",
    "FROSTBITE_ENV_ID",
    "OBSERVATION_SHAPE",
    "envpool_config",
    "initial_episode_statistics",
    "make_envpool",
    "normalize_env_id",
    "real_episode_done",
    "update_episode_statistics",
    "validate_envpool_contract",
    "validate_envpool_step",
]
