"""Flax policy networks for retrieval-augmented Atari PPO.

The convolutional encoder and policy/value heads mirror CleanRL's JAX Atari
PPO implementation.  Historical memory embeddings are always treated as
constants: PPO can train the query network in ``learned`` mode, but it cannot
backpropagate into entries that were written to episodic memory earlier.
"""

from __future__ import annotations

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from flax.linen.initializers import constant, orthogonal

RetrievalMode = Literal["none", "random", "learned"]
VALID_RETRIEVAL_MODES = ("none", "random", "learned")


def l2_normalize(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Normalize the final axis without producing NaNs for zero vectors."""

    squared_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    return x / jnp.sqrt(jnp.maximum(squared_norm, eps**2))


def symlog(x: jax.Array) -> jax.Array:
    """Compress raw reward magnitude while preserving its sign."""

    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def transition_embedding(
    observation_embeddings: jax.Array,
    actions: jax.Array,
    raw_rewards: jax.Array,
    action_dim: int,
) -> jax.Array:
    """Build detached (source observation, selected action, resulting reward) tuples."""

    observation_embeddings = jnp.asarray(observation_embeddings, dtype=jnp.float32)
    actions = jnp.asarray(actions)
    raw_rewards = jnp.asarray(raw_rewards, dtype=jnp.float32)
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    if observation_embeddings.ndim < 1 or observation_embeddings.shape[-1] != 512:
        raise ValueError("observation embeddings must have final dimension 512")
    if actions.shape != observation_embeddings.shape[:-1] or raw_rewards.shape != actions.shape:
        raise ValueError("actions and raw rewards must match observation embedding batch dimensions")
    values = jnp.concatenate(
        (observation_embeddings, jax.nn.one_hot(actions, action_dim), symlog(raw_rewards)[..., None]), axis=-1
    )
    return jax.lax.stop_gradient(values)


@struct.dataclass
class RetrievalOutput:
    """Retrieved context and diagnostics for one batch of observations."""

    context: jax.Array
    weights: jax.Array
    similarities: jax.Array
    entropy: jax.Array
    max_weight: jax.Array
    effective_num_memories: jax.Array
    candidate_episode_ids: jax.Array | None = None
    candidate_timesteps: jax.Array | None = None


@struct.dataclass
class AgentOutput:
    """Policy/value output plus features needed by collection and logging."""

    logits: jax.Array
    value: jax.Array
    observation_embedding: jax.Array
    memory_embedding: jax.Array
    query: jax.Array
    retrieval: RetrievalOutput


def _batched_candidates(query: jax.Array, candidates: jax.Array) -> jax.Array:
    candidates = jnp.asarray(candidates, dtype=jnp.float32)
    if candidates.ndim == 2:
        candidates = jnp.broadcast_to(candidates, (query.shape[0],) + candidates.shape)
    if candidates.ndim != 3:
        raise ValueError("candidate embeddings must have shape [K, D] or [B, K, D]")
    if candidates.shape[0] != query.shape[0]:
        raise ValueError("candidate and query batch dimensions must match")
    if candidates.shape[-1] != query.shape[-1]:
        raise ValueError("candidate and query embedding dimensions must match")
    if candidates.shape[1] == 0:
        raise ValueError("retrieval requires at least one candidate")
    return candidates


def retrieve_memories(
    query: jax.Array,
    candidate_embeddings: jax.Array,
    *,
    mode: Literal["random", "learned"],
    temperature: float = 0.1,
    similarity_bias: jax.Array | None = None,
    candidate_episode_ids: jax.Array | None = None,
    candidate_timesteps: jax.Array | None = None,
) -> RetrievalOutput:
    """Aggregate sampled memory candidates and expose retrieval diagnostics.

    Candidate embeddings are stopped at this boundary.  ``random`` uses a
    uniform mean and also stops the query for its diagnostic similarities, so
    every output is independent of query-network parameters in that control.
    ``learned`` uses temperature-scaled cosine similarities and softmax weights.
    Its optional ``similarity_bias`` is added to the scaled scores immediately
    before softmax, which lets callers measure PPO gradients with respect to
    each retrieval score without changing the forward pass when the bias is 0.
    """

    if mode not in ("random", "learned"):
        raise ValueError("retrieve_memories mode must be 'random' or 'learned'")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    query = jnp.asarray(query, dtype=jnp.float32)
    if query.ndim != 2:
        raise ValueError("query must have shape [B, D]")
    candidates = jax.lax.stop_gradient(_batched_candidates(query, candidate_embeddings))

    if mode == "learned":
        similarities = jnp.einsum("bd,bkd->bk", l2_normalize(query), l2_normalize(candidates))
        score_logits = similarities / temperature
        if similarity_bias is not None:
            similarity_bias = jnp.asarray(similarity_bias, dtype=score_logits.dtype)
            if similarity_bias.shape != score_logits.shape:
                raise ValueError(f"similarity bias must have shape {score_logits.shape}; got {similarity_bias.shape}")
            score_logits = score_logits + similarity_bias
        weights = jax.nn.softmax(score_logits, axis=-1)
    else:
        # The random control is deliberately a direct mean.  Query and cosine
        # work belongs to periodic probes, never ordinary policy inference.
        similarities = jnp.zeros(candidates.shape[:-1], dtype=candidates.dtype)
        weights = jnp.full_like(similarities, 1.0 / candidates.shape[1])

    context = jnp.einsum("bk,bkd->bd", weights, candidates)
    entropy = -jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-8)), axis=-1)
    return RetrievalOutput(
        context=context,
        weights=weights,
        similarities=similarities,
        entropy=entropy,
        max_weight=jnp.max(weights, axis=-1),
        effective_num_memories=jnp.exp(entropy),
        candidate_episode_ids=candidate_episode_ids,
        candidate_timesteps=candidate_timesteps,
    )


class AtariEncoder(nn.Module):
    """Three-convolution Atari encoder with GELU and a 512-D output."""

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = jnp.asarray(observations)
        if x.ndim != 4:
            raise ValueError("Atari observations must have rank 4")
        # CleanRL/envpool emits NCHW.  Accept NHWC too for easier evaluation with
        # Gymnasium wrappers while preserving the canonical NCHW behavior.
        if x.shape[1] in (1, 3, 4):
            x = jnp.transpose(x, (0, 2, 3, 1))
        elif x.shape[-1] not in (1, 3, 4):
            raise ValueError("expected channels in axis 1 (NCHW) or axis -1 (NHWC)")
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="conv1",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="conv2",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="conv3",
        )(x)
        x = nn.gelu(x, approximate=True)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            512,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="dense",
        )(x)
        return nn.gelu(x, approximate=True)


class QueryNetwork(nn.Module):
    """Observation-only query MLP with a configurable retrieval output width."""

    output_dim: int = 512

    @nn.compact
    def __call__(self, z: jax.Array) -> jax.Array:
        x = nn.Dense(
            512,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="dense1",
        )(z)
        x = nn.gelu(x, approximate=True)
        return nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="dense2",
        )(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="output",
        )(x)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="output",
        )(x)


class RetrievalAgent(nn.Module):
    """Atari actor-critic with a retrieval mode fixed at module construction."""

    action_dim: int
    retrieval_mode: RetrievalMode = "none"
    temperature: float = 0.1

    def setup(self) -> None:
        if self.retrieval_mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode: {self.retrieval_mode!r}")
        self.encoder = AtariEncoder(name="encoder")
        self.query_network = QueryNetwork(output_dim=512 + self.action_dim + 1, name="query")
        self.actor = Actor(self.action_dim, name="actor")
        self.critic = Critic(name="critic")

    def encode(self, observations: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return trainable observation features and detached-storage features.

        Memory uses the raw encoder features without projection or normalization.
        Device FIFO insertion and retrieval stop gradients through stored features;
        the observation features remain trainable.
        """

        z = self.encoder(observations)
        return z, z

    def apply_retrieved_context(
        self, observations: jax.Array, context: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Apply actor/critic to an externally aggregated evaluation context."""

        if self.retrieval_mode == "none":
            raise ValueError("external context is only valid for a retrieval policy")
        z, h = self.encode(observations)
        context_shape = (z.shape[0], 512 + self.action_dim + 1)
        query = self.query_network(z) if self.retrieval_mode == "learned" else jnp.zeros(context_shape, dtype=z.dtype)
        if context.shape != context_shape:
            raise ValueError(f"retrieval context must have shape {context_shape}; got {context.shape}")
        policy_input = jnp.concatenate((z, jax.lax.stop_gradient(context)), axis=-1)
        return self.actor(policy_input), self.critic(policy_input), query, h

    def retrieval_probe(
        self, observations: jax.Array, candidate_embeddings: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Run query/cosine diagnostics outside ordinary random inference."""

        if self.retrieval_mode == "none":
            raise ValueError("retrieval probes require a retrieval policy")
        z, _ = self.encode(observations)
        query = self.query_network(z)
        candidates = jax.lax.stop_gradient(_batched_candidates(query, candidate_embeddings))
        similarities = jnp.einsum("bd,bkd->bk", l2_normalize(query), l2_normalize(candidates))
        return query, similarities, z

    def __call__(
        self,
        observations: jax.Array,
        candidate_embeddings: jax.Array | None = None,
        candidate_episode_ids: jax.Array | None = None,
        candidate_timesteps: jax.Array | None = None,
        similarity_bias: jax.Array | None = None,
    ) -> AgentOutput:
        z, h = self.encode(observations)
        batch_size = z.shape[0]

        if self.retrieval_mode == "none":
            query = jnp.zeros_like(z)
            retrieval = RetrievalOutput(
                context=jnp.zeros_like(z),
                weights=jnp.zeros((batch_size, 0), dtype=z.dtype),
                similarities=jnp.zeros((batch_size, 0), dtype=z.dtype),
                entropy=jnp.zeros((batch_size,), dtype=z.dtype),
                max_weight=jnp.zeros((batch_size,), dtype=z.dtype),
                effective_num_memories=jnp.zeros((batch_size,), dtype=z.dtype),
            )
            policy_input = z
        else:
            if candidate_embeddings is None:
                raise ValueError(f"{self.retrieval_mode} mode requires candidate embeddings")
            # Random retains query parameters for architectural comparability,
            # but its ordinary inference path does not execute the MLP.
            if self.retrieval_mode == "random":
                if self.is_initializing():
                    self.query_network(z)
                query = jnp.zeros((batch_size, 512 + self.action_dim + 1), dtype=z.dtype)
            else:
                query = self.query_network(z)
            retrieval = retrieve_memories(
                query,
                candidate_embeddings,
                mode=self.retrieval_mode,
                temperature=self.temperature,
                similarity_bias=similarity_bias,
                candidate_episode_ids=candidate_episode_ids,
                candidate_timesteps=candidate_timesteps,
            )
            policy_input = jnp.concatenate((z, retrieval.context), axis=-1)

        return AgentOutput(
            logits=self.actor(policy_input),
            value=self.critic(policy_input),
            observation_embedding=z,
            memory_embedding=h,
            query=query,
            retrieval=retrieval,
        )


# Familiar CleanRL names for call sites that only need the baseline components.
Network = AtariEncoder
Agent = RetrievalAgent
