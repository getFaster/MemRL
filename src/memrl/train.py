"""JAX PPO training loop with retrieval-memory controls."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import tyro

from memrl.config import TrainConfig


def _finite_float(value) -> float:
    import numpy as np

    result = float(np.asarray(value))
    if not np.isfinite(result):
        raise FloatingPointError(f"non-finite PPO update metric: {result}")
    return result


def train(config: TrainConfig) -> Path:
    config.validate()
    # Must be assigned before importing JAX. Six-GB cards need room for the display and CUDA runtime.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.xla_memory_fraction))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import flax.serialization
    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training.train_state import TrainState

    from memrl.envs import episode_end_mask, extract_final_episode_stats, make_env
    from memrl.logging import RunLogger
    from memrl.memory import EpisodicMemory
    from memrl.models import RetrievalAgent

    random.seed(config.seed)
    np.random.seed(config.seed)
    run_name = f"{config.env_id.replace('/', '-')}__{config.retrieval_mode}__seed{config.seed}__{int(time.time())}"
    run_dir = config.output_dir / run_name
    checkpoint_dir = config.checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(
        run_dir,
        config.to_dict(),
        run_name,
        config.wandb_mode,
        config.wandb_project,
        config.wandb_entity,
        config.wandb_group,
    )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(config.env_id, config.seed + i, i, config.capture_video, run_name, run_dir / "videos")
            for i in range(config.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError("only discrete action spaces are supported")
    observation_shape = tuple(envs.single_observation_space.shape)
    if observation_shape != (4, 84, 84):
        raise ValueError(f"expected CleanRL Atari observations (4,84,84), got {observation_shape}")

    agent = RetrievalAgent(
        action_dim=envs.single_action_space.n,
        retrieval_mode=config.retrieval_mode,
        temperature=config.temperature,
    )
    key = jax.random.PRNGKey(config.seed)
    init_key, action_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, *observation_shape), dtype=jnp.uint8)
    init_args = [dummy_obs]
    if config.retrieval_mode != "none":
        init_args.append(jnp.zeros((1, config.retrieval_k, config.memory_dim), dtype=jnp.float32))
    params = agent.init(init_key, *init_args)

    def schedule(count):
        if not config.anneal_lr:
            return config.learning_rate
        total_updates = max(1, config.num_iterations * config.update_epochs * config.num_minibatches)
        return config.learning_rate * (1.0 - count / total_updates)

    tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(schedule, eps=1e-5))
    state = TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

    @jax.jit
    def policy_apply(policy_params, obs, candidates):
        if config.retrieval_mode == "none":
            return agent.apply(policy_params, obs)
        return agent.apply(policy_params, obs, candidates)

    def policy_apply_with_bias(policy_params, obs, candidates, similarity_bias):
        if config.retrieval_mode == "none":
            return agent.apply(policy_params, obs)
        return agent.apply(policy_params, obs, candidates, similarity_bias=similarity_bias)

    @jax.jit
    def sample_actions(policy_params, obs, candidates, rng):
        output = policy_apply(policy_params, obs, candidates)
        rng, sample_key = jax.random.split(rng)
        action = jax.random.categorical(sample_key, output.logits, axis=-1)
        logprob = jax.nn.log_softmax(output.logits)[jnp.arange(action.shape[0]), action]
        return output, action, logprob, rng

    if config.retrieval_mode != "none":
        apply_external_context = jax.jit(
            lambda policy_params, observations, context: agent.apply(
                policy_params, observations, context, method=agent.apply_retrieved_context
            )
        )

    def loss_fn(
        policy_params, obs, candidates, actions, old_logprobs, advantages, returns, old_values, similarity_bias
    ):
        output = policy_apply_with_bias(policy_params, obs, candidates, similarity_bias)
        log_probs = jax.nn.log_softmax(output.logits)
        new_logprob = log_probs[jnp.arange(actions.shape[0]), actions]
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
        values = output.value.squeeze(-1)
        if config.clip_vloss:
            unclipped = jnp.square(values - returns)
            clipped_values = old_values + jnp.clip(values - old_values, -config.clip_coef, config.clip_coef)
            value_loss = 0.5 * jnp.maximum(unclipped, jnp.square(clipped_values - returns)).mean()
        else:
            value_loss = 0.5 * jnp.square(values - returns).mean()
        loss = pg_loss - config.ent_coef * entropy + config.vf_coef * value_loss
        return loss, (pg_loss, value_loss, entropy, old_approx_kl, approx_kl, clipfrac)

    @jax.jit
    def update_minibatch(train_state, obs, candidates, actions, old_logprobs, advantages, returns, old_values):
        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 8), has_aux=True)
        score_bias = jnp.zeros(candidates.shape[:2], dtype=candidates.dtype)
        (loss, auxiliary), (grads, score_grads) = grad_fn(
            train_state.params, obs, candidates, actions, old_logprobs, advantages, returns, old_values, score_bias
        )
        grad_norm = optax.global_norm(grads)
        if config.retrieval_mode == "learned":
            query_grad_norm = optax.global_norm(grads["params"]["query"])
        else:
            query_grad_norm = jnp.asarray(0.0)
        mean_score_grad_norm = jnp.linalg.norm(score_grads, axis=-1).mean()
        return train_state.apply_gradients(grads=grads), (
            loss,
            *auxiliary,
            grad_norm,
            query_grad_norm,
            mean_score_grad_norm,
        )

    memory = EpisodicMemory(
        capacity=config.memory_capacity,
        memory_dim=config.memory_dim,
        observation_shape=(84, 84) if config.retrieval_mode != "none" else None,
        seed=config.seed + 10_000,
    )
    candidate_count = config.retrieval_k if config.retrieval_mode != "none" else 1
    retrieval_rng = np.random.default_rng(config.seed + 20_000)
    update_rng = np.random.default_rng(config.seed + 30_000)
    diagnostic_rng = np.random.default_rng(config.seed + 40_000)
    episode_ids = np.arange(config.num_envs, dtype=np.int64)
    previous_episode_ids = np.full(config.num_envs, -1, dtype=np.int64)
    next_episode_id = config.num_envs
    episode_steps = np.zeros(config.num_envs, dtype=np.int64)

    next_obs, _ = envs.reset(seed=[config.seed + i for i in range(config.num_envs)])
    next_done = np.zeros(config.num_envs, dtype=np.float32)
    global_step = 0
    started_at = time.time()
    drift_observations: list[np.ndarray] = []
    drift_embeddings: list[np.ndarray] = []
    drift_insertion_ids: list[int] = []

    def candidate_batch():
        embeddings = np.zeros((config.num_envs, candidate_count, config.memory_dim), dtype=np.float32)
        indices = np.full((config.num_envs, candidate_count), -1, dtype=np.int64)
        candidate_episodes = np.full_like(indices, -1)
        candidate_steps = np.full_like(indices, -1)
        logical_ids = np.full_like(indices, -1)
        if config.retrieval_mode == "none" or len(memory) == 0:
            return embeddings, indices, logical_ids, candidate_episodes, candidate_steps
        for env_index in range(config.num_envs):
            sample = memory.sample_candidates(candidate_count, rng=retrieval_rng, include_frames=False)
            embeddings[env_index] = sample.embeddings
            indices[env_index] = sample.physical_indices
            logical_ids[env_index] = sample.logical_indices
            candidate_episodes[env_index] = sample.episode_ids
            candidate_steps[env_index] = sample.timesteps
        return embeddings, indices, logical_ids, candidate_episodes, candidate_steps

    def save_checkpoint(name: str) -> None:
        path = checkpoint_dir / f"{name}.msgpack"
        path.write_bytes(flax.serialization.to_bytes(state.params))
        (checkpoint_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "run_name": run_name,
                    "config": config.to_dict(),
                    "observation_shape": [int(value) for value in observation_shape],
                    "action_dim": int(envs.single_action_space.n),
                },
                indent=2,
            )
            + "\n"
        )
        memory.save(checkpoint_dir / "memory.npz", include_frames=config.save_memory_frames)
        if drift_observations:
            np.savez_compressed(
                checkpoint_dir / "drift_reference.npz",
                observations=np.asarray(drift_observations),
                stored_embeddings=np.asarray(drift_embeddings),
                insertion_ids=np.asarray(drift_insertion_ids),
            )

    try:
        for iteration in range(1, config.num_iterations + 1):
            iteration_started = time.time()
            obs_buf = np.empty((config.num_steps, config.num_envs, *observation_shape), dtype=np.uint8)
            candidates_buf = np.empty(
                (config.num_steps, config.num_envs, candidate_count, config.memory_dim), dtype=np.float32
            )
            indices_buf = np.empty((config.num_steps, config.num_envs, candidate_count), dtype=np.int64)
            candidate_episodes_buf = np.empty_like(indices_buf)
            candidate_steps_buf = np.empty_like(indices_buf)
            actions_buf = np.empty((config.num_steps, config.num_envs), dtype=np.int32)
            logprobs_buf = np.empty((config.num_steps, config.num_envs), dtype=np.float32)
            rewards_buf = np.empty((config.num_steps, config.num_envs), dtype=np.float32)
            dones_buf = np.empty((config.num_steps, config.num_envs), dtype=np.float32)
            values_buf = np.empty((config.num_steps, config.num_envs), dtype=np.float32)
            episode_events: list[tuple[float, int]] = []
            retrieval_rows: list[list] = []
            diagnostic_values = []
            intervention_values = []
            retrieval_ages = []

            for step in range(config.num_steps):
                global_step += config.num_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done
                candidate_embeddings, candidate_indices, candidate_logical_ids, candidate_episodes, candidate_steps = (
                    candidate_batch()
                )
                candidates_buf[step] = candidate_embeddings
                indices_buf[step] = candidate_indices
                candidate_episodes_buf[step] = candidate_episodes
                candidate_steps_buf[step] = candidate_steps

                output, action, logprob, action_key = sample_actions(
                    state.params, next_obs, candidate_embeddings, action_key
                )
                host_output, host_action, host_logprob = jax.device_get((output, action, logprob))
                actions_buf[step] = host_action
                logprobs_buf[step] = host_logprob
                values_buf[step] = np.asarray(host_output.value).squeeze(-1)

                # Insert only after selecting the action, preventing self-retrieval.
                h = np.asarray(host_output.memory_embedding, dtype=np.float32)
                if config.retrieval_mode != "none":
                    for env_index in range(config.num_envs):
                        if len(drift_observations) >= 64:
                            break
                        drift_observations.append(np.asarray(next_obs[env_index]).copy())
                        drift_embeddings.append(h[env_index].copy())
                        drift_insertion_ids.append(memory.total_added + env_index)

                if config.retrieval_mode != "none" and len(memory) > 0:
                    weights = np.asarray(host_output.retrieval.weights)
                    similarities = np.asarray(host_output.retrieval.similarities)
                    contexts = np.asarray(host_output.retrieval.context)
                    observation_embeddings = np.asarray(host_output.observation_embedding)
                    same_episode = candidate_episodes == episode_ids[:, None]
                    previous_episode = candidate_episodes == previous_episode_ids[:, None]
                    temporal = np.abs(candidate_steps - episode_steps[:, None])
                    retrieval_age = memory.total_added - candidate_logical_ids - 1
                    retrieval_ages.append(retrieval_age.reshape(-1))
                    mean_temporal = float(temporal[same_episode].mean()) if same_episode.any() else 0.0
                    z_norm = np.linalg.norm(observation_embeddings, axis=-1)
                    m_norm = np.linalg.norm(contexts, axis=-1)
                    diagnostic_values.append(
                        (
                            float(np.asarray(host_output.retrieval.entropy).mean()),
                            float(np.asarray(host_output.retrieval.max_weight).mean()),
                            float(np.exp(np.asarray(host_output.retrieval.entropy)).mean()),
                            mean_temporal,
                            float(similarities.max()),
                            float(similarities.min()),
                            float(similarities.mean()),
                            float(similarities.std()),
                            float(same_episode.mean()),
                            float(previous_episode.mean()),
                            float(retrieval_age.mean()),
                            float((retrieval_age < 500).mean()),
                            float(z_norm.mean()),
                            float(m_norm.mean()),
                            float((m_norm / np.maximum(z_norm, 1e-8)).mean()),
                        )
                    )
                    if step == config.num_steps - 1:
                        zero_context = np.zeros_like(contexts)
                        random_context = candidate_embeddings.mean(axis=1)
                        shuffled_context = np.roll(contexts, 1, axis=0) if config.num_envs > 1 else contexts
                        zero_logits, zero_values, _, _ = jax.device_get(
                            apply_external_context(state.params, next_obs, zero_context)
                        )
                        random_logits, _, _, _ = jax.device_get(
                            apply_external_context(state.params, next_obs, random_context)
                        )
                        shuffled_logits, _, _, _ = jax.device_get(
                            apply_external_context(state.params, next_obs, shuffled_context)
                        )
                        actual_logits = np.asarray(host_output.logits)

                        def categorical_kl(p_logits, q_logits):
                            p_log = np.asarray(jax.nn.log_softmax(p_logits, axis=-1))
                            q_log = np.asarray(jax.nn.log_softmax(q_logits, axis=-1))
                            value = float((np.exp(p_log) * (p_log - q_log)).sum(axis=-1).mean())
                            return max(0.0, value)

                        intervention_values.append(
                            (
                                categorical_kl(actual_logits, zero_logits),
                                float(np.abs(np.asarray(host_output.value) - np.asarray(zero_values)).mean()),
                                categorical_kl(actual_logits, random_logits),
                                categorical_kl(actual_logits, shuffled_logits),
                            )
                        )
                    if iteration % config.diagnostics_interval == 0 and step == config.num_steps - 1:
                        for env_index in range(config.num_envs):
                            top = np.argsort(weights[env_index])[-config.diagnostics_top_k :][::-1]
                            for rank, candidate_position in enumerate(top):
                                slot = candidate_indices[env_index, candidate_position]
                                if slot < 0:
                                    continue
                                frame = memory.frames[slot].copy() if memory.frames is not None else None
                                frame_path = (
                                    run_dir / "retrieval_frames" / f"step{global_step}_env{env_index}_rank{rank}.npy"
                                )
                                frame_path.parent.mkdir(parents=True, exist_ok=True)
                                np.save(frame_path, frame, allow_pickle=False)
                                image = logger.wandb.Image(frame) if logger.wandb is not None else str(frame_path)
                                retrieval_rows.append(
                                    [
                                        global_step,
                                        env_index,
                                        rank,
                                        int(slot),
                                        int(candidate_episodes[env_index, candidate_position]),
                                        int(candidate_steps[env_index, candidate_position]),
                                        float(weights[env_index, candidate_position]),
                                        float(similarities[env_index, candidate_position]),
                                        image,
                                    ]
                                )

                if config.retrieval_mode != "none":
                    memory.add(h, episode_ids, episode_steps, np.asarray(next_obs)[:, -1])

                next_obs, reward, terminated, truncated, infos = envs.step(host_action)
                next_done = np.logical_or(terminated, truncated).astype(np.float32)
                rewards_buf[step] = reward
                episode_events.extend(extract_final_episode_stats(infos))
                episode_steps += 1
                for env_index in np.flatnonzero(episode_end_mask(infos, config.num_envs)):
                    previous_episode_ids[env_index] = episode_ids[env_index]
                    episode_ids[env_index] = next_episode_id
                    next_episode_id += 1
                    episode_steps[env_index] = 0

            final_candidates, _, _, _, _ = candidate_batch()
            final_output = jax.device_get(policy_apply(state.params, next_obs, final_candidates))
            next_value = np.asarray(final_output.value).squeeze(-1)
            advantages = np.zeros_like(rewards_buf)
            last_gae = np.zeros(config.num_envs, dtype=np.float32)
            for step in reversed(range(config.num_steps)):
                if step == config.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    following_value = next_value
                else:
                    next_nonterminal = 1.0 - dones_buf[step + 1]
                    following_value = values_buf[step + 1]
                delta = rewards_buf[step] + config.gamma * following_value * next_nonterminal - values_buf[step]
                last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
                advantages[step] = last_gae
            returns = advantages + values_buf

            flat_obs = obs_buf.reshape(config.batch_size, *observation_shape)
            flat_candidates = candidates_buf.reshape(config.batch_size, candidate_count, config.memory_dim)
            flat_actions = actions_buf.reshape(config.batch_size)
            flat_logprobs = logprobs_buf.reshape(config.batch_size)
            flat_advantages = advantages.reshape(config.batch_size)
            flat_returns = returns.reshape(config.batch_size)
            flat_values = values_buf.reshape(config.batch_size)
            query_before = None
            if config.retrieval_mode == "learned":
                query_before = jax.tree.map(lambda value: np.asarray(value).copy(), state.params["params"]["query"])
            epoch_metrics = []
            for _epoch in range(config.update_epochs):
                permutation = update_rng.permutation(config.batch_size)
                epoch_kls = []
                for start in range(0, config.batch_size, config.minibatch_size):
                    mb = permutation[start : start + config.minibatch_size]
                    state, result = update_minibatch(
                        state,
                        flat_obs[mb],
                        flat_candidates[mb],
                        flat_actions[mb],
                        flat_logprobs[mb],
                        flat_advantages[mb],
                        flat_returns[mb],
                        flat_values[mb],
                    )
                    values = [_finite_float(x) for x in jax.device_get(result)]
                    epoch_metrics.append(values)
                    epoch_kls.append(values[5])
                if config.target_kl is not None and np.mean(epoch_kls) > config.target_kl:
                    break

            update_metrics = np.asarray(epoch_metrics).mean(axis=0)
            variance = np.var(flat_returns)
            explained_variance = 0.0 if variance == 0 else 1.0 - np.var(flat_returns - flat_values) / variance
            elapsed = max(time.time() - started_at, 1e-9)
            iteration_elapsed = max(time.time() - iteration_started, 1e-9)
            metrics = {
                "charts/learning_rate": schedule(state.step),
                "charts/SPS": global_step / elapsed,
                "charts/SPS_update": config.batch_size / iteration_elapsed,
                "charts/projected_hours_per_10m_steps": (10_000_000 / (global_step / elapsed)) / 3600,
                "losses/loss": update_metrics[0],
                "losses/policy_loss": update_metrics[1],
                "losses/value_loss": update_metrics[2],
                "losses/entropy": update_metrics[3],
                "losses/old_approx_kl": update_metrics[4],
                "losses/approx_kl": update_metrics[5],
                "losses/clipfrac": update_metrics[6],
                "losses/explained_variance": explained_variance,
                "gradients/global_norm": update_metrics[7],
                "gradients/query_network_norm": update_metrics[8],
                "gradients/mean_retrieval_score_norm": update_metrics[9],
                "memory/size": len(memory),
            }
            if query_before is not None:
                delta_leaves = jax.tree.leaves(
                    jax.tree.map(
                        lambda after, before: np.asarray(after) - before,
                        state.params["params"]["query"],
                        query_before,
                    )
                )
                metrics["retrieval/query_parameter_change"] = float(
                    np.sqrt(sum(float(np.square(leaf).sum()) for leaf in delta_leaves))
                )
                optimizer_updates = max(1, len(epoch_metrics))
                metrics["retrieval/query_parameter_change_per_optimizer_step"] = (
                    metrics["retrieval/query_parameter_change"] / optimizer_updates
                )
                # The memory projection is deliberately parameter-free; its trainable
                # similarity projection is the query network itself.
                metrics["gradients/similarity_projection_norm"] = update_metrics[8]
            if episode_events:
                metrics["charts/episodic_return"] = float(np.mean([x["return"] for x in episode_events]))
                metrics["charts/episodic_length"] = float(np.mean([x["length"] for x in episode_events]))
                metrics["charts/episodes_completed"] = len(episode_events)
            if diagnostic_values:
                d = np.asarray(diagnostic_values).mean(axis=0)
                keys = [
                    "retrieval/entropy",
                    "retrieval/max_weight",
                    "retrieval/effective_num_memories",
                    "retrieval/mean_temporal_distance",
                    "retrieval/max_similarity",
                    "retrieval/min_similarity",
                    "retrieval/mean_similarity",
                    "retrieval/std_similarity",
                    "retrieval/same_episode_fraction",
                    "retrieval/previous_episode_fraction",
                    "retrieval/mean_age",
                    "retrieval/recent_under_500_fraction",
                    "representations/observation_embedding_norm",
                    "representations/retrieved_memory_norm",
                    "representations/memory_to_observation_norm_ratio",
                ]
                metrics.update(dict(zip(keys, d, strict=True)))
                metrics["retrieval/temperature"] = config.temperature
            if intervention_values:
                intervention = np.asarray(intervention_values).mean(axis=0)
                metrics.update(
                    {
                        "interventions/policy_kl_memory_vs_zero": intervention[0],
                        "interventions/value_abs_difference_memory_vs_zero": intervention[1],
                        "interventions/policy_kl_memory_vs_random": intervention[2],
                        "interventions/policy_kl_memory_vs_shuffled": intervention[3],
                    }
                )

            if iteration % config.diagnostics_interval == 0 and len(memory) >= 2:
                sample_count = min(256, len(memory))
                sample = memory.sample_candidates(sample_count, rng=diagnostic_rng, replace=False, include_frames=False)
                embeddings = sample.embeddings
                norms = np.linalg.norm(embeddings, axis=-1)
                normalized = embeddings / np.maximum(norms[:, None], 1e-8)
                paired = np.roll(normalized, 1, axis=0)
                pair_cosine = np.sum(normalized * paired, axis=-1)
                pair_distance = np.linalg.norm(embeddings - np.roll(embeddings, 1, axis=0), axis=-1)
                metrics.update(
                    {
                        "memory/random_pair_cosine_mean": float(pair_cosine.mean()),
                        "memory/random_pair_cosine_std": float(pair_cosine.std()),
                        "memory/dimension_variance_mean": float(embeddings.var(axis=0).mean()),
                        "memory/pairwise_embedding_distance": float(pair_distance.mean()),
                        "memory/near_duplicate_fraction": float((pair_cosine > 0.99).mean()),
                        "memory/embedding_norm_before_normalization": float(norms.mean()),
                    }
                )
                if drift_observations:
                    _, current_h = agent.apply(state.params, np.asarray(drift_observations), method=agent.encode)
                    stored_h = np.asarray(drift_embeddings)
                    current_h = np.asarray(current_h)
                    drift_cosine = np.sum(
                        stored_h
                        / np.maximum(np.linalg.norm(stored_h, axis=-1, keepdims=True), 1e-8)
                        * current_h
                        / np.maximum(np.linalg.norm(current_h, axis=-1, keepdims=True), 1e-8),
                        axis=-1,
                    )
                    drift_age = memory.total_added - np.asarray(drift_insertion_ids) - 1
                    metrics["drift/stored_current_cosine_mean"] = float(drift_cosine.mean())
                    metrics["drift/stored_current_cosine_min"] = float(drift_cosine.min())
                    metrics["drift/diagnostic_mean_age"] = float(drift_age.mean())
                    if drift_cosine.std() > 0 and drift_age.std() > 0:
                        metrics["drift/age_cosine_correlation"] = float(np.corrcoef(drift_age, drift_cosine)[0, 1])
            logger.log(metrics, global_step)
            if logger.wandb is not None and retrieval_ages:
                logger.wandb.log(
                    {"retrieval/age_histogram": logger.wandb.Histogram(np.concatenate(retrieval_ages))},
                    step=global_step,
                )
            logger.log_retrieval_table(
                retrieval_rows,
                ["step", "env", "rank", "memory_slot", "episode_id", "timestep", "weight", "similarity", "frame"],
                global_step,
            )
            print(
                f"iteration={iteration}/{config.num_iterations} step={global_step} "
                f"mode={config.retrieval_mode} SPS={metrics['charts/SPS']:.0f}"
            )
            if config.checkpoint_interval and iteration % config.checkpoint_interval == 0:
                save_checkpoint(f"step_{global_step}")
        save_checkpoint("final")
        return checkpoint_dir / "final.msgpack"
    finally:
        envs.close()
        logger.close()


def main() -> None:
    train(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
