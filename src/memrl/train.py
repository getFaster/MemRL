"""Compiled EnvPool/XLA PPO training with device-resident retrieval memory."""

from __future__ import annotations

import importlib.metadata
import json
import os
import resource
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

import tyro

from memrl.config import TrainConfig


def _finite_float(value: Any) -> float:
    import numpy as np

    result = float(np.asarray(value))
    if not np.isfinite(result):
        raise FloatingPointError(f"non-finite PPO update metric: {result}")
    return result


def record_transition(memory, observation_embedding, action, info, episode_ids, timesteps, action_dim):
    """Store completed transitions; EnvPool reset calls do not execute actions.

    The embedding and metadata belong to the observation *before* the step.
    Call only after retrieval/action selection and the environment response.
    """
    import jax.numpy as jnp

    from memrl.memory import insert_batch
    from memrl.models import transition_embedding

    embeddings = transition_embedding(observation_embedding, action, info["reward"], action_dim)
    valid = jnp.asarray(info["elapsed_step"]) > 0
    return insert_batch(memory, embeddings, episode_ids, timesteps, valid=valid)


def _provenance() -> dict[str, Any]:
    dependencies = {}
    for name in ("envpool", "jax", "flax", "optax", "orbax-checkpoint", "numpy"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"dependencies": dependencies, "git_commit": commit, "git_dirty": dirty}


def train(config: TrainConfig) -> Path:
    config.validate()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.xla_memory_fraction))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import struct
    from flax.training.train_state import TrainState

    from memrl.checkpointing import MEMORY_LAYOUT, MemRLCheckpointManager, restore_checkpoint
    from memrl.diagnostics import AGE_BIN_EDGES
    from memrl.envs import OBSERVATION_SHAPE, initial_episode_statistics, make_envpool, update_episode_statistics
    from memrl.logging import RunLogger
    from memrl.memory import HostFrameRing, create_device_memory, sample_batch
    from memrl.models import RetrievalAgent

    retrieval = config.retrieval_mode != "none"
    candidate_count = config.retrieval_k if retrieval else 0
    if config.capture_video:
        raise ValueError("capture_video is not supported by the compiled EnvPool training path")

    @struct.dataclass
    class RolloutStorage:
        obs: jax.Array
        candidates: jax.Array
        candidate_physical: jax.Array
        candidate_insertion_ids: jax.Array
        candidate_episode_ids: jax.Array
        candidate_timesteps: jax.Array
        candidate_valid: jax.Array
        memory_embeddings: jax.Array
        actions: jax.Array
        logprobs: jax.Array
        dones: jax.Array
        values: jax.Array
        rewards: jax.Array
        event_mask: jax.Array
        event_returns: jax.Array
        event_lengths: jax.Array
        next_candidates: jax.Array

    @struct.dataclass
    class EpisodeState:
        statistics: Any
        episode_ids: jax.Array
        previous_episode_ids: jax.Array
        timesteps: jax.Array
        next_episode_id: jax.Array

    @struct.dataclass
    class RetrievalSummary:
        candidate_count: jax.Array
        env_count: jax.Array
        entropy_sum: jax.Array
        max_weight_sum: jax.Array
        effective_sum: jax.Array
        similarity_sum: jax.Array
        similarity_sumsq: jax.Array
        similarity_min: jax.Array
        similarity_max: jax.Array
        same_episode_count: jax.Array
        previous_episode_count: jax.Array
        temporal_sum: jax.Array
        temporal_count: jax.Array
        age_sum: jax.Array
        recent_count: jax.Array
        observation_norm_sum: jax.Array
        action_norm_sum: jax.Array
        reward_norm_sum: jax.Array
        memory_observation_norm_sum: jax.Array
        context_norm_sum: jax.Array
        norm_ratio_sum: jax.Array
        age_histogram: jax.Array
        final_physical: jax.Array
        final_insertion_ids: jax.Array
        final_episode_ids: jax.Array
        final_timesteps: jax.Array
        final_valid: jax.Array
        final_weights: jax.Array
        final_similarities: jax.Array
        final_actions: jax.Array
        final_rewards: jax.Array

    def empty_storage() -> RolloutStorage:
        prefix = (config.num_steps, config.num_envs)
        candidate_shape = (*prefix, candidate_count)
        return RolloutStorage(
            obs=jnp.zeros((*prefix, *OBSERVATION_SHAPE), dtype=jnp.uint8),
            candidates=jnp.zeros((*candidate_shape, config.memory_dim), dtype=jnp.float32),
            candidate_physical=jnp.full(candidate_shape, -1, dtype=jnp.int32),
            candidate_insertion_ids=jnp.full(candidate_shape, -1, dtype=jnp.int32),
            candidate_episode_ids=jnp.full(candidate_shape, -1, dtype=jnp.int32),
            candidate_timesteps=jnp.full(candidate_shape, -1, dtype=jnp.int32),
            candidate_valid=jnp.zeros(candidate_shape, dtype=jnp.bool_),
            # Encoder features are always 512D, including legacy none-mode configs.
            memory_embeddings=jnp.zeros((*prefix, 512), dtype=jnp.float32),
            actions=jnp.zeros(prefix, dtype=jnp.int32),
            logprobs=jnp.zeros(prefix, dtype=jnp.float32),
            dones=jnp.zeros(prefix, dtype=jnp.float32),
            values=jnp.zeros(prefix, dtype=jnp.float32),
            rewards=jnp.zeros(prefix, dtype=jnp.float32),
            event_mask=jnp.zeros(prefix, dtype=jnp.bool_),
            event_returns=jnp.zeros(prefix, dtype=jnp.float32),
            event_lengths=jnp.zeros(prefix, dtype=jnp.int32),
            next_candidates=jnp.zeros((config.num_envs, candidate_count, config.memory_dim), dtype=jnp.float32),
        )

    def empty_summary() -> RetrievalSummary:
        shape = (config.num_envs, candidate_count)
        return RetrievalSummary(
            candidate_count=jnp.asarray(0, dtype=jnp.int32),
            env_count=jnp.asarray(0, dtype=jnp.int32),
            entropy_sum=jnp.asarray(0.0),
            max_weight_sum=jnp.asarray(0.0),
            effective_sum=jnp.asarray(0.0),
            similarity_sum=jnp.asarray(0.0),
            similarity_sumsq=jnp.asarray(0.0),
            similarity_min=jnp.asarray(jnp.inf),
            similarity_max=jnp.asarray(-jnp.inf),
            same_episode_count=jnp.asarray(0, dtype=jnp.int32),
            previous_episode_count=jnp.asarray(0, dtype=jnp.int32),
            temporal_sum=jnp.asarray(0.0),
            temporal_count=jnp.asarray(0, dtype=jnp.int32),
            age_sum=jnp.asarray(0.0),
            recent_count=jnp.asarray(0, dtype=jnp.int32),
            observation_norm_sum=jnp.asarray(0.0),
            action_norm_sum=jnp.asarray(0.0),
            reward_norm_sum=jnp.asarray(0.0),
            memory_observation_norm_sum=jnp.asarray(0.0),
            context_norm_sum=jnp.asarray(0.0),
            norm_ratio_sum=jnp.asarray(0.0),
            age_histogram=jnp.zeros((len(AGE_BIN_EDGES) - 1,), dtype=jnp.int32),
            final_physical=jnp.full(shape, -1, dtype=jnp.int32),
            final_insertion_ids=jnp.full(shape, -1, dtype=jnp.int32),
            final_episode_ids=jnp.full(shape, -1, dtype=jnp.int32),
            final_timesteps=jnp.full(shape, -1, dtype=jnp.int32),
            final_valid=jnp.zeros(shape, dtype=jnp.bool_),
            final_weights=jnp.zeros(shape, dtype=jnp.float32),
            final_similarities=jnp.zeros(shape, dtype=jnp.float32),
            final_actions=jnp.full(shape, -1, dtype=jnp.int32),
            final_rewards=jnp.zeros(shape, dtype=jnp.float32),
        )

    envs = make_envpool(config.env_id, num_envs=config.num_envs, seed=config.seed)
    action_dim = int(envs.single_action_space.n)
    config.resolve_memory_dim(action_dim)
    handle, _recv, _send, step_env = envs.xla()
    next_obs, _ = envs.reset()
    next_obs = jnp.asarray(next_obs)
    next_done = jnp.zeros((config.num_envs,), dtype=jnp.float32)

    agent = RetrievalAgent(action_dim=action_dim, retrieval_mode=config.retrieval_mode, temperature=config.temperature)
    root_key = jax.random.PRNGKey(config.seed)
    init_key, action_key, retrieval_key, update_key, diagnostic_key = jax.random.split(root_key, 5)
    dummy_obs = jnp.zeros((1, *OBSERVATION_SHAPE), dtype=jnp.uint8)
    if retrieval:
        params = agent.init(
            init_key, dummy_obs, jnp.zeros((1, config.retrieval_k, config.memory_dim), dtype=jnp.float32)
        )
    else:
        params = agent.init(init_key, dummy_obs)

    def schedule(count: jax.Array) -> jax.Array:
        if not config.anneal_lr:
            return jnp.asarray(config.learning_rate)
        total_updates = max(1, config.num_iterations * config.update_epochs * config.num_minibatches)
        return config.learning_rate * jnp.maximum(0.0, 1.0 - count / total_updates)

    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(schedule, eps=1e-5))
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=optimizer)
    memory = create_device_memory(config.memory_capacity, config.memory_dim) if retrieval else None
    episode_state = EpisodeState(
        statistics=initial_episode_statistics(config.num_envs),
        episode_ids=jnp.arange(config.num_envs, dtype=jnp.int32),
        previous_episode_ids=jnp.full((config.num_envs,), -1, dtype=jnp.int32),
        timesteps=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        next_episode_id=jnp.asarray(config.num_envs, dtype=jnp.int32),
    )

    def advance_episodes(current: EpisodeState, info: dict[str, Any], truncated: jax.Array):
        statistics, events = update_episode_statistics(current.statistics, info, truncated)
        completed = events.mask
        offsets = jnp.cumsum(completed.astype(jnp.int32)) - 1
        new_ids = current.next_episode_id + offsets
        return (
            current.replace(
                statistics=statistics,
                previous_episode_ids=jnp.where(completed, current.episode_ids, current.previous_episode_ids),
                episode_ids=jnp.where(completed, new_ids, current.episode_ids),
                timesteps=jnp.where(completed, 0, current.timesteps + (jnp.asarray(info["elapsed_step"]) > 0)),
                next_episode_id=current.next_episode_id + completed.astype(jnp.int32).sum(),
            ),
            events,
        )

    def sample_action(policy_params, observation, candidates, key):
        output = (
            agent.apply(policy_params, observation, candidates)
            if retrieval
            else agent.apply(policy_params, observation)
        )
        key, sample_key = jax.random.split(key)
        action = jax.random.categorical(sample_key, output.logits, axis=-1).astype(jnp.int32)
        logprob = jnp.take_along_axis(jax.nn.log_softmax(output.logits), action[:, None], axis=1)[:, 0]
        return output, action, logprob, key

    def accumulate_summary(summary, output, sample, episodes, total_insertions):
        valid = sample.valid
        env_valid = jnp.any(valid, axis=1)
        candidate_n = valid.astype(jnp.int32).sum()
        env_n = env_valid.astype(jnp.int32).sum()
        ages = jnp.where(valid, total_insertions - sample.insertion_ids - 1, 0)
        same = valid & (sample.episode_ids == episodes.episode_ids[:, None])
        previous = valid & (sample.episode_ids == episodes.previous_episode_ids[:, None])
        temporal = jnp.abs(sample.timesteps - episodes.timesteps[:, None])
        similarities = output.retrieval.similarities
        masked_min = jnp.min(jnp.where(valid, similarities, jnp.inf))
        masked_max = jnp.max(jnp.where(valid, similarities, -jnp.inf))
        z_norm = jnp.linalg.norm(output.observation_embedding, axis=-1)
        context_norm = jnp.linalg.norm(output.retrieval.context, axis=-1)
        histogram = jnp.stack(
            [
                jnp.sum(valid & (ages >= lower) & (ages < upper), dtype=jnp.int32)
                for lower, upper in zip(AGE_BIN_EDGES[:-1], AGE_BIN_EDGES[1:], strict=True)
            ]
        )
        return summary.replace(
            candidate_count=summary.candidate_count + candidate_n,
            env_count=summary.env_count + env_n,
            entropy_sum=summary.entropy_sum + jnp.sum(jnp.where(env_valid, output.retrieval.entropy, 0.0)),
            max_weight_sum=summary.max_weight_sum + jnp.sum(jnp.where(env_valid, output.retrieval.max_weight, 0.0)),
            effective_sum=summary.effective_sum
            + jnp.sum(jnp.where(env_valid, output.retrieval.effective_num_memories, 0.0)),
            similarity_sum=summary.similarity_sum + jnp.sum(jnp.where(valid, similarities, 0.0)),
            similarity_sumsq=summary.similarity_sumsq + jnp.sum(jnp.where(valid, similarities**2, 0.0)),
            similarity_min=jnp.minimum(summary.similarity_min, masked_min),
            similarity_max=jnp.maximum(summary.similarity_max, masked_max),
            same_episode_count=summary.same_episode_count + same.astype(jnp.int32).sum(),
            previous_episode_count=summary.previous_episode_count + previous.astype(jnp.int32).sum(),
            temporal_sum=summary.temporal_sum + jnp.sum(jnp.where(same, temporal, 0.0)),
            temporal_count=summary.temporal_count + same.astype(jnp.int32).sum(),
            age_sum=summary.age_sum + jnp.sum(jnp.where(valid, ages, 0.0)),
            recent_count=summary.recent_count + (valid & (ages < 500)).astype(jnp.int32).sum(),
            observation_norm_sum=summary.observation_norm_sum + jnp.sum(jnp.where(env_valid, z_norm, 0.0)),
            action_norm_sum=summary.action_norm_sum
            + jnp.sum(jnp.where(valid, jnp.linalg.norm(sample.embeddings[..., 512:-1], axis=-1), 0.0)),
            reward_norm_sum=summary.reward_norm_sum
            + jnp.sum(jnp.where(valid, jnp.abs(sample.embeddings[..., -1]), 0.0)),
            memory_observation_norm_sum=summary.memory_observation_norm_sum
            + jnp.sum(jnp.where(valid, jnp.linalg.norm(sample.embeddings[..., :512], axis=-1), 0.0)),
            context_norm_sum=summary.context_norm_sum + jnp.sum(jnp.where(env_valid, context_norm, 0.0)),
            norm_ratio_sum=summary.norm_ratio_sum
            + jnp.sum(jnp.where(env_valid, context_norm / jnp.maximum(z_norm, 1e-8), 0.0)),
            age_histogram=summary.age_histogram + histogram,
            final_physical=sample.physical_indices,
            final_insertion_ids=sample.insertion_ids,
            final_episode_ids=sample.episode_ids,
            final_timesteps=sample.timesteps,
            final_valid=sample.valid,
            final_weights=output.retrieval.weights,
            final_similarities=output.retrieval.similarities,
            final_actions=jnp.where(valid, jnp.argmax(sample.embeddings[..., 512:-1], axis=-1), -1),
            final_rewards=sample.embeddings[..., -1],
        )

    @partial(jax.jit, donate_argnums=(4, 8))
    def rollout_retrieval(policy_params, episodes, observation, done, storage, rng, sample_rng, env_handle, mem):
        summary = empty_summary()
        frame_batch = jnp.zeros((config.num_steps, config.num_envs, 84, 84), dtype=jnp.uint8)
        frame_slots = jnp.zeros((config.num_steps, config.num_envs), dtype=jnp.int32)
        frame_insertions = jnp.zeros((config.num_steps, config.num_envs), dtype=jnp.int32)

        def rollout_step(carry, step):
            (
                episodes,
                observation,
                done,
                storage,
                rng,
                sample_rng,
                env_handle,
                mem,
                summary,
                frame_batch,
                frame_slots,
                frame_insertions,
            ) = carry
            sample = sample_batch(mem, sample_rng, config.num_envs, config.retrieval_k)
            sample_rng = sample.key
            output, action, logprob, rng = sample_action(policy_params, observation, sample.embeddings, rng)
            storage = storage.replace(
                obs=storage.obs.at[step].set(observation),
                candidates=storage.candidates.at[step].set(sample.embeddings),
                candidate_physical=storage.candidate_physical.at[step].set(sample.physical_indices),
                candidate_insertion_ids=storage.candidate_insertion_ids.at[step].set(sample.insertion_ids),
                candidate_episode_ids=storage.candidate_episode_ids.at[step].set(sample.episode_ids),
                candidate_timesteps=storage.candidate_timesteps.at[step].set(sample.timesteps),
                candidate_valid=storage.candidate_valid.at[step].set(sample.valid),
                memory_embeddings=storage.memory_embeddings.at[step].set(output.memory_embedding),
                actions=storage.actions.at[step].set(action),
                logprobs=storage.logprobs.at[step].set(logprob),
                dones=storage.dones.at[step].set(done),
                values=storage.values.at[step].set(output.value[:, 0]),
            )
            summary = accumulate_summary(summary, output, sample, episodes, mem.total_insertions)
            frame_batch = frame_batch.at[step].set(observation[:, -1])
            env_handle, (observation, reward, terminated, truncated, info) = step_env(env_handle, action)
            inserted = record_transition(
                mem, output.memory_embedding, action, info, episodes.episode_ids, episodes.timesteps, action_dim
            )
            mem = inserted.state
            slots = inserted.physical_indices
            frame_slots = frame_slots.at[step].set(slots)
            frame_insertions = frame_insertions.at[step].set(
                jnp.where(slots >= 0, mem.insertion_ids[jnp.maximum(slots, 0)], -1)
            )
            episodes, events = advance_episodes(episodes, info, truncated)
            done = jnp.logical_or(terminated, truncated).astype(jnp.float32)
            storage = storage.replace(
                rewards=storage.rewards.at[step].set(reward),
                event_mask=storage.event_mask.at[step].set(events.mask),
                event_returns=storage.event_returns.at[step].set(events.returns),
                event_lengths=storage.event_lengths.at[step].set(events.lengths),
            )
            return (
                episodes,
                observation,
                done,
                storage,
                rng,
                sample_rng,
                env_handle,
                mem,
                summary,
                frame_batch,
                frame_slots,
                frame_insertions,
            ), None

        (
            (
                episodes,
                observation,
                done,
                storage,
                rng,
                sample_rng,
                env_handle,
                mem,
                summary,
                frame_batch,
                frame_slots,
                frame_insertions,
            ),
            _,
        ) = jax.lax.scan(
            rollout_step,
            (
                episodes,
                observation,
                done,
                storage,
                rng,
                sample_rng,
                env_handle,
                mem,
                summary,
                frame_batch,
                frame_slots,
                frame_insertions,
            ),
            jnp.arange(config.num_steps, dtype=jnp.int32),
        )
        bootstrap = sample_batch(mem, sample_rng, config.num_envs, config.retrieval_k)
        storage = storage.replace(next_candidates=bootstrap.embeddings)
        return (
            episodes,
            observation,
            done,
            storage,
            rng,
            bootstrap.key,
            env_handle,
            mem,
            summary,
            frame_batch,
            frame_slots,
            frame_insertions,
        )

    @partial(jax.jit, donate_argnums=(4,))
    def rollout_none(policy_params, episodes, observation, done, storage, rng, env_handle):
        for step in range(config.num_steps):
            output, action, logprob, rng = sample_action(policy_params, observation, None, rng)
            storage = storage.replace(
                obs=storage.obs.at[step].set(observation),
                memory_embeddings=storage.memory_embeddings.at[step].set(output.memory_embedding),
                actions=storage.actions.at[step].set(action),
                logprobs=storage.logprobs.at[step].set(logprob),
                dones=storage.dones.at[step].set(done),
                values=storage.values.at[step].set(output.value[:, 0]),
            )
            env_handle, (observation, reward, terminated, truncated, info) = step_env(env_handle, action)
            episodes, events = advance_episodes(episodes, info, truncated)
            done = jnp.logical_or(terminated, truncated).astype(jnp.float32)
            storage = storage.replace(
                rewards=storage.rewards.at[step].set(reward),
                event_mask=storage.event_mask.at[step].set(events.mask),
                event_returns=storage.event_returns.at[step].set(events.returns),
                event_lengths=storage.event_lengths.at[step].set(events.lengths),
            )
        return episodes, observation, done, storage, rng, env_handle

    def policy_loss(policy_params, obs, candidates, actions, old_logprobs, advantages, returns, old_values, bias):
        output = (
            agent.apply(policy_params, obs, candidates, similarity_bias=bias)
            if retrieval
            else agent.apply(policy_params, obs)
        )
        log_probs = jax.nn.log_softmax(output.logits)
        new_logprob = jnp.take_along_axis(log_probs, actions[:, None], axis=1)[:, 0]
        probs = jax.nn.softmax(output.logits)
        entropy = -(probs * log_probs).sum(axis=-1).mean()
        logratio = new_logprob - old_logprobs
        ratio = jnp.exp(logratio)
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1.0) - logratio).mean()
        clipfrac = (jnp.abs(ratio - 1.0) > config.clip_coef).mean()
        if config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss = jnp.maximum(
            -advantages * ratio,
            -advantages * jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef),
        ).mean()
        values = output.value[:, 0]
        if config.clip_vloss:
            unclipped = jnp.square(values - returns)
            clipped = old_values + jnp.clip(values - old_values, -config.clip_coef, config.clip_coef)
            value_loss = 0.5 * jnp.maximum(unclipped, jnp.square(clipped - returns)).mean()
        else:
            value_loss = 0.5 * jnp.square(values - returns).mean()
        loss = pg_loss - config.ent_coef * entropy + config.vf_coef * value_loss
        return loss, (pg_loss, value_loss, entropy, old_approx_kl, approx_kl, clipfrac)

    current_done = next_done

    @partial(jax.jit, donate_argnums=(0,))
    def update_ppo(train_state, storage, observation, final_done, update_rng):
        next_output = (
            agent.apply(train_state.params, observation, storage.next_candidates)
            if retrieval
            else agent.apply(train_state.params, observation)
        )
        next_value = next_output.value[:, 0]
        advantages = jnp.zeros_like(storage.rewards)
        following_values = jnp.concatenate((storage.values[1:], next_value[None, :]), axis=0)
        following_dones = jnp.concatenate((storage.dones[1:], final_done[None, :]), axis=0)

        def gae_step(index, carry):
            current_advantages, last_gae = carry
            step = config.num_steps - 1 - index
            next_nonterminal = 1.0 - following_dones[step]
            delta = (
                storage.rewards[step] + config.gamma * following_values[step] * next_nonterminal - storage.values[step]
            )
            last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
            return current_advantages.at[step].set(last_gae), last_gae

        advantages, _ = jax.lax.fori_loop(
            0, config.num_steps, gae_step, (advantages, jnp.zeros((config.num_envs,), dtype=jnp.float32))
        )
        returns = advantages + storage.values
        flat_obs = storage.obs.reshape((config.batch_size, *OBSERVATION_SHAPE))
        flat_candidates = storage.candidates.reshape((config.batch_size, candidate_count, config.memory_dim))
        flat_actions = storage.actions.reshape((config.batch_size,))
        flat_logprobs = storage.logprobs.reshape((config.batch_size,))
        flat_advantages = advantages.reshape((config.batch_size,))
        flat_returns = returns.reshape((config.batch_size,))
        flat_values = storage.values.reshape((config.batch_size,))

        def epoch_step(_epoch, carry):
            epoch_state, key, metric_sum = carry
            key, permutation_key = jax.random.split(key)
            permutation = jax.random.permutation(permutation_key, config.batch_size, independent=True)

            def minibatch_step(minibatch_index, inner_carry):
                minibatch_state, accumulated = inner_carry
                start = minibatch_index * config.minibatch_size
                indices = jax.lax.dynamic_slice_in_dim(permutation, start, config.minibatch_size)
                args = (
                    flat_obs[indices],
                    flat_candidates[indices],
                    flat_actions[indices],
                    flat_logprobs[indices],
                    flat_advantages[indices],
                    flat_returns[indices],
                    flat_values[indices],
                )
                bias = jnp.zeros((config.minibatch_size, candidate_count), dtype=jnp.float32)
                if config.retrieval_mode == "learned":
                    grad_fn = jax.value_and_grad(policy_loss, argnums=(0, 8), has_aux=True)
                    loss_and_aux, gradients = grad_fn(minibatch_state.params, *args, bias)
                    grads, score_grads = gradients
                else:
                    grad_fn = jax.value_and_grad(policy_loss, has_aux=True)
                    loss_and_aux, grads = grad_fn(minibatch_state.params, *args, bias)
                    score_grads = jnp.zeros_like(bias)
                loss, auxiliary = loss_and_aux
                grad_norm = optax.global_norm(grads)
                query_grad = (
                    optax.global_norm(grads["params"]["query"])
                    if config.retrieval_mode == "learned"
                    else jnp.asarray(0.0)
                )
                score_grad = jnp.linalg.norm(score_grads, axis=-1).mean()
                values = jnp.stack((loss, *auxiliary, grad_norm, query_grad, score_grad))
                return minibatch_state.apply_gradients(grads=grads), accumulated + values

            epoch_state, metric_sum = jax.lax.fori_loop(
                0, config.num_minibatches, minibatch_step, (epoch_state, metric_sum)
            )
            return epoch_state, key, metric_sum

        metric_count = config.update_epochs * config.num_minibatches
        train_state, update_rng, metric_sum = jax.lax.fori_loop(
            0,
            config.update_epochs,
            epoch_step,
            (train_state, update_rng, jnp.zeros((10,), dtype=jnp.float32)),
        )
        variance = jnp.var(flat_returns)
        explained_variance = jnp.where(variance > 0, 1.0 - jnp.var(flat_returns - flat_values) / variance, 0.0)
        return train_state, metric_sum / metric_count, explained_variance, update_rng

    if retrieval:

        @jax.jit
        def periodic_probe(train_state, storage, mem, key):
            observation = storage.obs[-1]
            candidates = storage.candidates[-1]
            output = agent.apply(train_state.params, observation, candidates)
            zero_context = jnp.zeros_like(output.retrieval.context)
            random_context = jnp.mean(candidates, axis=1)
            shuffled_context = jnp.roll(output.retrieval.context, 1, axis=0)
            zero_logits, zero_values, _, _ = agent.apply(
                train_state.params, observation, zero_context, method=agent.apply_retrieved_context
            )
            random_logits, _, _, _ = agent.apply(
                train_state.params, observation, random_context, method=agent.apply_retrieved_context
            )
            shuffled_logits, _, _, _ = agent.apply(
                train_state.params, observation, shuffled_context, method=agent.apply_retrieved_context
            )

            def kl(left, right):
                left_log = jax.nn.log_softmax(left)
                right_log = jax.nn.log_softmax(right)
                return jnp.mean(jnp.sum(jnp.exp(left_log) * (left_log - right_log), axis=-1))

            query, similarities, _ = agent.apply(
                train_state.params, observation, candidates, method=agent.retrieval_probe
            )
            sample = sample_batch(mem, key, 1, min(256, config.memory_capacity))
            embeddings = sample.embeddings[0]
            norms = jnp.linalg.norm(embeddings, axis=-1)
            normalized = embeddings / jnp.maximum(norms[:, None], 1e-8)
            paired = jnp.roll(normalized, 1, axis=0)
            pair_cosine = jnp.sum(normalized * paired, axis=-1)
            current_h = agent.apply(train_state.params, observation, method=agent.encode)[1]
            stored_h = storage.memory_embeddings[-1]
            drift = jnp.sum(
                stored_h
                / jnp.maximum(jnp.linalg.norm(stored_h, axis=-1, keepdims=True), 1e-8)
                * current_h
                / jnp.maximum(jnp.linalg.norm(current_h, axis=-1, keepdims=True), 1e-8),
                axis=-1,
            )
            return {
                "diagnostics/random_query_norm": jnp.linalg.norm(query, axis=-1).mean(),
                "diagnostics/random_similarity_mean": similarities.mean(),
                "diagnostics/random_similarity_std": similarities.std(),
                "interventions/policy_kl_memory_vs_zero": kl(output.logits, zero_logits),
                "interventions/value_abs_difference_memory_vs_zero": jnp.abs(output.value - zero_values).mean(),
                "interventions/policy_kl_memory_vs_random": kl(output.logits, random_logits),
                "interventions/policy_kl_memory_vs_shuffled": kl(output.logits, shuffled_logits),
                "memory/random_pair_cosine_mean": pair_cosine.mean(),
                "memory/random_pair_cosine_std": pair_cosine.std(),
                "memory/dimension_variance_mean": embeddings.var(axis=0).mean(),
                "memory/pairwise_embedding_distance": jnp.linalg.norm(
                    embeddings - jnp.roll(embeddings, 1, axis=0), axis=-1
                ).mean(),
                "memory/near_duplicate_fraction": (pair_cosine > 0.99).mean(),
                "memory/embedding_norm_before_normalization": norms.mean(),
                "drift/stored_current_cosine_mean": drift.mean(),
                "drift/stored_current_cosine_min": drift.min(),
            }, sample.key

    frame_ring = HostFrameRing.create(config.memory_capacity, (84, 84)) if retrieval else None
    global_step = 0
    completed_rollouts = 0
    resume_history: list[dict[str, Any]] = []
    resume_wandb_identity: dict[str, Any] = {}

    def training_payload() -> dict[str, Any]:
        payload = {
            "state": state,
            "action_key": action_key,
            "retrieval_key": retrieval_key,
            "update_key": update_key,
            "diagnostic_key": diagnostic_key,
            "global_step": jnp.asarray(global_step, dtype=jnp.int32),
            "completed_rollouts": jnp.asarray(completed_rollouts, dtype=jnp.int32),
            "next_episode_id": episode_state.next_episode_id,
        }
        if retrieval:
            payload["memory"] = memory
        return payload

    if config.resume_from is not None:
        bundle = restore_checkpoint(config.resume_from, training_payload())
        saved_config = bundle.metadata["config"]
        locked = ("retrieval_mode", "memory_capacity", "memory_dim", "retrieval_k", "num_envs", "num_steps")
        mismatches = [name for name in locked if saved_config.get(name) != config.to_dict().get(name)]
        if mismatches:
            raise ValueError(f"resume configuration changes locked fields: {mismatches}")
        restored = bundle.training
        state = restored["state"]
        action_key = restored["action_key"]
        retrieval_key = restored["retrieval_key"]
        update_key = restored["update_key"]
        diagnostic_key = restored["diagnostic_key"]
        global_step = int(restored["global_step"])
        completed_rollouts = int(restored["completed_rollouts"])
        base_episode_id = int(restored["next_episode_id"])
        if retrieval:
            memory = restored["memory"]
        episode_state = EpisodeState(
            statistics=initial_episode_statistics(config.num_envs),
            episode_ids=jnp.arange(base_episode_id, base_episode_id + config.num_envs, dtype=jnp.int32),
            previous_episode_ids=jnp.full((config.num_envs,), -1, dtype=jnp.int32),
            timesteps=jnp.zeros((config.num_envs,), dtype=jnp.int32),
            next_episode_id=jnp.asarray(base_episode_id + config.num_envs, dtype=jnp.int32),
        )
        resume_history = list(bundle.metadata.get("resume_history", []))
        resume_wandb_identity = dict(bundle.metadata.get("wandb_identity", {}))
        resume_history.append(
            {
                "source": str(Path(config.resume_from).resolve()),
                "resumed_at_unix": time.time(),
                "global_step": global_step,
                "environment_reset_discontinuity": True,
                "active_episode_statistics_reset": True,
            }
        )
        if frame_ring is not None and bundle.diagnostics is not None:
            frame_ring.frames[...] = bundle.diagnostics["frames"]
            frame_ring.insertion_ids[...] = bundle.diagnostics["insertion_ids"]
            frame_ring.valid[...] = bundle.diagnostics["valid"]

    timestamp = time.time_ns()
    run_name = f"{config.env_id}__{config.retrieval_mode}__seed{config.seed}__{timestamp}"
    run_dir = config.output_dir / run_name
    checkpoint_root = config.checkpoint_dir / run_name
    wandb_visible_name = str(resume_wandb_identity.get("run_name", run_name))
    logger = RunLogger(
        run_dir,
        config.to_dict(),
        wandb_visible_name,
        config.wandb_mode,
        config.wandb_project,
        config.wandb_entity,
        config.wandb_group,
        wandb_id=resume_wandb_identity.get("id"),
    )
    checkpoint_manager = (
        MemRLCheckpointManager(checkpoint_root, keep_periodic=config.retain_periodic_checkpoints)
        if config.checkpointing
        else None
    )
    provenance = _provenance()
    started_at = time.time()

    def checkpoint_metadata(frame_coverage: float) -> dict[str, Any]:
        return {
            "global_step": global_step,
            "completed_rollouts": completed_rollouts,
            "run_name": run_name,
            "config": config.to_dict(),
            "observation_shape": list(OBSERVATION_SHAPE),
            "action_dim": action_dim,
            "memory_layout": MEMORY_LAYOUT if retrieval else "observation_v1",
            "provenance": provenance,
            "wandb_identity": {
                "project": config.wandb_project,
                "entity": config.wandb_entity,
                "group": config.wandb_group,
                "run_name": wandb_visible_name,
                "id": logger.wandb_run.id if logger.wandb_run is not None else resume_wandb_identity.get("id"),
            },
            "resume_history": resume_history,
            "environment_state_restorable": False,
            "diagnostics_frame_coverage": frame_coverage,
        }

    def diagnostic_payload(memory_size: int) -> dict[str, np.ndarray] | None:
        if not retrieval or not config.save_memory_frames or frame_ring is None:
            return None
        occupied = np.zeros((config.memory_capacity,), dtype=np.bool_)
        occupied[: min(memory_size, config.memory_capacity)] = True
        return {
            "frames": frame_ring.frames,
            "insertion_ids": frame_ring.insertion_ids,
            "valid": frame_ring.valid,
            "occupied": occupied,
        }

    def current_frame_coverage() -> float:
        if frame_ring is None or memory_size == 0:
            return 0.0
        if memory_size >= config.memory_capacity:
            return float(frame_ring.valid.mean())
        return float(frame_ring.valid[:memory_size].mean())

    memory_size = int(memory.size) if retrieval and memory is not None else 0
    final_path = checkpoint_root / "final"
    timing_rows: list[dict[str, float | int]] = []
    cold_compilation_seconds: float | None = None
    try:
        for iteration in range(completed_rollouts + 1, config.num_iterations + 1):
            iteration_started = time.time()
            iteration_start_step = global_step
            storage = empty_storage()
            if retrieval:
                assert memory is not None and frame_ring is not None
                (
                    episode_state,
                    next_obs,
                    current_done,
                    storage,
                    action_key,
                    retrieval_key,
                    handle,
                    memory,
                    summary,
                    frame_batch,
                    frame_slots,
                    frame_insertions,
                ) = rollout_retrieval(
                    state.params,
                    episode_state,
                    next_obs,
                    current_done,
                    storage,
                    action_key,
                    retrieval_key,
                    handle,
                    memory,
                )
                host_summary, host_frames, host_slots, host_insertions, host_events, host_memory_size = jax.device_get(
                    (
                        summary,
                        frame_batch,
                        frame_slots,
                        frame_insertions,
                        (storage.event_mask, storage.event_returns, storage.event_lengths),
                        memory.size,
                    )
                )
                frame_ring.update(
                    host_frames.reshape((-1, 84, 84)), host_slots.reshape(-1), host_insertions.reshape(-1)
                )
                memory_size = int(host_memory_size)
            else:
                episode_state, next_obs, current_done, storage, action_key, handle = rollout_none(
                    state.params, episode_state, next_obs, current_done, storage, action_key, handle
                )
                host_events = jax.device_get((storage.event_mask, storage.event_returns, storage.event_lengths))
                host_summary = None

            global_step += config.batch_size
            completed_rollouts = iteration
            state, update_metrics, explained_variance, update_key = update_ppo(
                state, storage, next_obs, current_done, update_key
            )
            probe_metrics = {}
            if retrieval and iteration % config.diagnostics_interval == 0:
                assert memory is not None
                probe_metrics, diagnostic_key = periodic_probe(state, storage, memory, diagnostic_key)
            host_update, host_explained, host_probe, host_learning_rate = jax.device_get(
                (update_metrics, explained_variance, probe_metrics, schedule(state.step))
            )
            values = [_finite_float(value) for value in host_update]
            elapsed = max(time.time() - started_at, 1e-9)
            iteration_elapsed = max(time.time() - iteration_started, 1e-9)
            if cold_compilation_seconds is None:
                cold_compilation_seconds = iteration_elapsed
            timing_rows.append(
                {
                    "iteration_start_step": iteration_start_step,
                    "global_step": global_step,
                    "wall_time_seconds": iteration_elapsed,
                }
            )
            metrics: dict[str, Any] = {
                "charts/learning_rate": _finite_float(host_learning_rate),
                "charts/SPS": global_step / elapsed,
                "charts/SPS_update": config.batch_size / iteration_elapsed,
                "charts/iteration_wall_time_seconds": iteration_elapsed,
                "charts/projected_hours_per_10m_steps": (10_000_000 / (global_step / elapsed)) / 3600,
                "losses/loss": values[0],
                "losses/policy_loss": values[1],
                "losses/value_loss": values[2],
                "losses/entropy": values[3],
                "losses/old_approx_kl": values[4],
                "losses/approx_kl": values[5],
                "losses/clipfrac": values[6],
                "losses/explained_variance": _finite_float(host_explained),
                "gradients/global_norm": values[7],
                "gradients/query_network_norm": values[8],
                "gradients/mean_retrieval_score_norm": values[9],
                "memory/size": memory_size,
            }
            event_mask, event_returns, event_lengths = map(np.asarray, host_events)
            completed_positions = np.argwhere(event_mask)
            episode_returns = []
            episode_lengths = []
            for step_index, env_slot in completed_positions:
                raw_return = float(event_returns[step_index, env_slot])
                length = int(event_lengths[step_index, env_slot])
                completion_step = iteration_start_step + (int(step_index) + 1) * config.num_envs
                logger.log_episode(
                    env_slot=int(env_slot), completion_step=completion_step, raw_return=raw_return, length=length
                )
                episode_returns.append(raw_return)
                episode_lengths.append(length)
            if episode_returns:
                metrics.update(
                    {
                        "charts/episodic_return": float(np.mean(episode_returns)),
                        "charts/episodic_length": float(np.mean(episode_lengths)),
                        "charts/episodes_completed": len(episode_returns),
                    }
                )

            if retrieval and host_summary is not None:
                candidates_n = max(1, int(host_summary.candidate_count))
                env_n = max(1, int(host_summary.env_count))
                similarity_mean = float(host_summary.similarity_sum) / candidates_n
                similarity_variance = max(0.0, float(host_summary.similarity_sumsq) / candidates_n - similarity_mean**2)
                temporal_n = max(1, int(host_summary.temporal_count))
                metrics.update(
                    {
                        "retrieval/entropy": float(host_summary.entropy_sum) / env_n,
                        "retrieval/max_weight": float(host_summary.max_weight_sum) / env_n,
                        "retrieval/effective_num_memories": float(host_summary.effective_sum) / env_n,
                        "retrieval/mean_temporal_distance": float(host_summary.temporal_sum) / temporal_n,
                        "retrieval/same_episode_fraction": int(host_summary.same_episode_count) / candidates_n,
                        "retrieval/previous_episode_fraction": int(host_summary.previous_episode_count) / candidates_n,
                        "retrieval/mean_age": float(host_summary.age_sum) / candidates_n,
                        "retrieval/recent_under_500_fraction": int(host_summary.recent_count) / candidates_n,
                        "representations/observation_embedding_norm": float(host_summary.observation_norm_sum) / env_n,
                        "representations/retrieved_memory_norm": float(host_summary.context_norm_sum) / env_n,
                        "memory/observation_block_norm": float(host_summary.memory_observation_norm_sum) / candidates_n,
                        "memory/action_block_norm": float(host_summary.action_norm_sum) / candidates_n,
                        "memory/reward_block_norm": float(host_summary.reward_norm_sum) / candidates_n,
                        "representations/memory_to_observation_norm_ratio": float(host_summary.norm_ratio_sum) / env_n,
                        "retrieval/age_histogram": np.asarray(host_summary.age_histogram).astype(int).tolist(),
                        "retrieval/temperature": config.temperature,
                        "diagnostics/frame_coverage": current_frame_coverage(),
                    }
                )
                if config.retrieval_mode == "learned" and int(host_summary.candidate_count):
                    metrics.update(
                        {
                            "retrieval/max_similarity": float(host_summary.similarity_max),
                            "retrieval/min_similarity": float(host_summary.similarity_min),
                            "retrieval/mean_similarity": similarity_mean,
                            "retrieval/std_similarity": float(np.sqrt(similarity_variance)),
                        }
                    )
                if iteration % config.diagnostics_interval == 0:
                    metrics.update({key: _finite_float(value) for key, value in host_probe.items()})
                    rows = []
                    final_weights = np.asarray(host_summary.final_weights)
                    for env_slot in range(config.num_envs):
                        order = np.argsort(final_weights[env_slot])[-config.diagnostics_top_k :][::-1]
                        for rank, position in enumerate(order):
                            slot = int(host_summary.final_physical[env_slot, position])
                            insertion_id = int(host_summary.final_insertion_ids[env_slot, position])
                            if slot < 0:
                                continue
                            available = bool(frame_ring.available(np.asarray([slot]), np.asarray([insertion_id]))[0])
                            frame = frame_ring.frames[slot].copy() if available else None
                            image = (
                                logger.wandb.Image(frame) if logger.wandb is not None and frame is not None else frame
                            )
                            rows.append(
                                [
                                    global_step,
                                    env_slot,
                                    rank,
                                    slot,
                                    insertion_id,
                                    int(host_summary.final_episode_ids[env_slot, position]),
                                    int(host_summary.final_timesteps[env_slot, position]),
                                    float(host_summary.final_weights[env_slot, position]),
                                    float(host_summary.final_similarities[env_slot, position]),
                                    int(host_summary.final_actions[env_slot, position]),
                                    float(host_summary.final_rewards[env_slot, position]),
                                    available,
                                    image,
                                ]
                            )
                    logger.log_retrieval_table(
                        rows,
                        [
                            "step",
                            "env",
                            "rank",
                            "memory_slot",
                            "insertion_id",
                            "episode_id",
                            "timestep",
                            "weight",
                            "similarity",
                            "transition_action",
                            "transition_reward_symlog",
                            "frame_available",
                            "frame",
                        ],
                        global_step,
                    )

            if resume_history:
                metrics["resume/environment_reset_discontinuities"] = len(resume_history)
            logger.log(metrics, global_step)
            print(
                f"iteration={iteration}/{config.num_iterations} step={global_step} "
                f"mode={config.retrieval_mode} SPS={metrics['charts/SPS']:.0f}"
            )
            if (
                checkpoint_manager is not None
                and config.checkpoint_interval
                and iteration % config.checkpoint_interval == 0
            ):
                coverage = current_frame_coverage()
                checkpoint_manager.save(
                    f"step_{global_step}",
                    training_payload(),
                    checkpoint_metadata(coverage),
                    diagnostic_payload(memory_size),
                    periodic=True,
                )

        device_stats = jax.devices()[0].memory_stats() or {}
        timing_report = {
            "schema_version": 1,
            "mode": config.retrieval_mode,
            "seed": config.seed,
            "run_id": run_name,
            "process_id": os.getpid(),
            "benchmark_run": config.benchmark_run,
            "requested_total_timesteps": config.total_timesteps,
            "final_global_step": global_step,
            "batch_size": config.batch_size,
            "memory_capacity": config.memory_capacity,
            "memory_dim": config.memory_dim,
            "memory_layout": config.memory_layout,
            "finite": True,
            "oom": False,
            "cold_compilation_seconds": cold_compilation_seconds,
            "peak_device_bytes": device_stats.get("peak_bytes_in_use"),
            "peak_host_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "iterations": timing_rows,
            "host_frame_maintenance": retrieval,
            "checkpointing_enabled": config.checkpointing,
            "wandb_mode": config.wandb_mode,
        }
        (run_dir / "timing.json").write_text(json.dumps(timing_report, indent=2) + "\n")
        if checkpoint_manager is None:
            return run_dir
        coverage = current_frame_coverage()
        final_path = checkpoint_manager.save(
            "final", training_payload(), checkpoint_metadata(coverage), diagnostic_payload(memory_size)
        )
        checkpoint_manager.wait()
        return final_path
    finally:
        if checkpoint_manager is not None:
            checkpoint_manager.close()
        envs.close()
        logger.close()


def main() -> None:
    train(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
