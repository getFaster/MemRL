"""Device-resident episodic memory and retrieval operations.

Training memory is a pure JAX value. Sampling and insertion can therefore run
inside the compiled rollout without copying embeddings or metadata through the
host. Diagnostic frames deliberately live in a separate NumPy ring: they are
optional evidence, not part of the state required to continue training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

Array = jax.Array


@struct.dataclass
class DeviceMemoryState:
    """Immutable JAX FIFO state carried by the compiled rollout."""

    embeddings: Array
    episode_ids: Array
    timesteps: Array
    insertion_ids: Array
    size: Array
    next_index: Array
    total_insertions: Array

    @property
    def capacity(self) -> int:
        return self.embeddings.shape[0]

    @property
    def memory_dim(self) -> int:
        return self.embeddings.shape[1]


class MemorySampleBatch(NamedTuple):
    """Fixed-shape independent samples for all rollout environments."""

    embeddings: Array
    physical_indices: Array
    insertion_ids: Array
    episode_ids: Array
    timesteps: Array
    valid: Array
    key: Array


class MemoryInsertResult(NamedTuple):
    """Physical write locations and the updated immutable memory state."""

    physical_indices: Array
    state: DeviceMemoryState


def create_device_memory(capacity: int, dim: int) -> DeviceMemoryState:
    """Allocate an empty device memory.

    Callers must not invoke this function in ``none`` mode; absence of a state
    is how that mode avoids the roughly 100 MiB embedding allocation.
    """

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    return DeviceMemoryState(
        embeddings=jnp.zeros((capacity, dim), dtype=jnp.float32),
        episode_ids=jnp.full((capacity,), -1, dtype=jnp.int32),
        timesteps=jnp.full((capacity,), -1, dtype=jnp.int32),
        insertion_ids=jnp.full((capacity,), -1, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
        next_index=jnp.asarray(0, dtype=jnp.int32),
        total_insertions=jnp.asarray(0, dtype=jnp.int32),
    )


def _floyd_offsets(key: Array, num_envs: int, k: int, size: Array) -> Array:
    """Draw independent uniform K-subsets from ``range(size)``.

    Floyd's algorithm needs exactly K draws and K slots per environment. Its
    work and temporary storage are independent of memory capacity.
    """

    # Column i is uniform on [0, size - k + i], the Floyd range for that step.
    upper_bounds = size - k + jnp.arange(k, dtype=jnp.int32) + 1
    draws = jax.random.randint(key, (num_envs, k), 0, upper_bounds, dtype=jnp.int32)
    selected = jnp.full((num_envs, k), -1, dtype=jnp.int32)

    def add_one(i: int, values: Array) -> Array:
        draw = draws[:, i]
        collision = jnp.any(values == draw[:, None], axis=1)
        replacement = size - k + i
        chosen = jnp.where(collision, replacement, draw)
        return values.at[:, i].set(chosen)

    return jax.lax.fori_loop(0, k, add_one, selected)


def sample_batch(state: DeviceMemoryState, key: Array, num_envs: int, k: int) -> MemorySampleBatch:
    """Sample candidates independently for each environment.

    For ``0 < size < k`` sampling is with replacement. At ``size >= k`` a
    Floyd sampler produces a uniform subset without replacement. Empty memory
    returns invalid zero candidates. The key advances in all cases so restore
    and replay have one stable PRNG transition per rollout step.
    """

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > state.capacity:
        raise ValueError("k cannot exceed memory capacity")

    next_key, sample_key = jax.random.split(key)
    index_shape = (num_envs, k)

    def empty(_: None) -> tuple[Array, Array]:
        return jnp.zeros(index_shape, dtype=jnp.int32), jnp.zeros(index_shape, dtype=jnp.bool_)

    def nonempty(_: None) -> tuple[Array, Array]:
        def warmup(_: None) -> Array:
            return jax.random.randint(sample_key, index_shape, 0, state.size, dtype=jnp.int32)

        def full(_: None) -> Array:
            return _floyd_offsets(sample_key, num_envs, k, state.size)

        offsets = jax.lax.cond(state.size < k, warmup, full, operand=None)
        oldest = jnp.where(state.size == state.capacity, state.next_index, 0)
        physical = (oldest + offsets) % state.capacity
        return physical, jnp.ones(index_shape, dtype=jnp.bool_)

    physical, valid = jax.lax.cond(state.size == 0, empty, nonempty, operand=None)
    # Slot zero is a safe gather for empty memory; validity marks it meaningless.
    embeddings = jnp.where(valid[..., None], state.embeddings[physical], 0.0)
    invalid_metadata = jnp.full(index_shape, -1, dtype=jnp.int32)
    insertion_ids = jnp.where(valid, state.insertion_ids[physical], invalid_metadata)
    episode_ids = jnp.where(valid, state.episode_ids[physical], invalid_metadata)
    timesteps = jnp.where(valid, state.timesteps[physical], invalid_metadata)
    return MemorySampleBatch(
        embeddings=jax.lax.stop_gradient(embeddings),
        physical_indices=jnp.where(valid, physical, invalid_metadata),
        insertion_ids=insertion_ids,
        episode_ids=episode_ids,
        timesteps=timesteps,
        valid=valid,
        key=next_key,
    )


def insert_batch(
    state: DeviceMemoryState,
    embeddings: Array,
    episode_ids: Array,
    timesteps: Array,
    valid: Array | None = None,
) -> MemoryInsertResult:
    """Insert valid environment rows in env-slot order.

    Invalid rows consume no FIFO slots and return physical index ``-1``.
    Mask compaction touches only the environment batch, never the full FIFO.
    """

    values = jnp.asarray(embeddings, dtype=jnp.float32)
    episodes = jnp.asarray(episode_ids, dtype=jnp.int32)
    steps = jnp.asarray(timesteps, dtype=jnp.int32)
    if values.ndim != 2 or values.shape[1] != state.memory_dim:
        raise ValueError(f"embeddings must have shape [N, {state.memory_dim}]")
    count = values.shape[0]
    if count <= 0:
        raise ValueError("insert batch must not be empty")
    if count > state.capacity:
        raise ValueError("insert batch cannot exceed memory capacity")
    if episodes.shape != (count,) or steps.shape != (count,):
        raise ValueError("episode_ids and timesteps must have shape [N]")

    if valid is None:
        offsets = jnp.arange(count, dtype=jnp.int32)
        inserted_count = count
        physical = (state.next_index + offsets) % state.capacity
        write_indices = physical
    else:
        mask = jnp.asarray(valid, dtype=jnp.bool_)
        if mask.shape != (count,):
            raise ValueError("valid must have shape [N]")
        offsets = jnp.cumsum(mask, dtype=jnp.int32) - 1
        inserted_count = jnp.sum(mask, dtype=jnp.int32)
        physical = jnp.where(mask, (state.next_index + offsets) % state.capacity, -1)
        # Drop masked rows rather than writing a sentinel into a real FIFO slot.
        write_indices = jnp.where(mask, physical, state.capacity)
    insertion_ids = state.total_insertions + offsets
    next_state = state.replace(
        embeddings=state.embeddings.at[write_indices].set(jax.lax.stop_gradient(values), mode="drop"),
        episode_ids=state.episode_ids.at[write_indices].set(episodes, mode="drop"),
        timesteps=state.timesteps.at[write_indices].set(steps, mode="drop"),
        insertion_ids=state.insertion_ids.at[write_indices].set(insertion_ids, mode="drop"),
        size=jnp.minimum(state.capacity, state.size + inserted_count),
        next_index=(state.next_index + inserted_count) % state.capacity,
        total_insertions=state.total_insertions + inserted_count,
    )
    return MemoryInsertResult(physical_indices=physical, state=next_state)


@dataclass
class HostFrameRing:
    """Optional host-only diagnostic frames keyed by device-memory slots."""

    frames: np.ndarray
    insertion_ids: np.ndarray
    valid: np.ndarray

    @classmethod
    def create(cls, capacity: int, frame_shape: tuple[int, ...]) -> HostFrameRing:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not frame_shape or any(dimension <= 0 for dimension in frame_shape):
            raise ValueError("frame_shape must contain positive dimensions")
        return cls(
            frames=np.zeros((capacity, *frame_shape), dtype=np.uint8),
            insertion_ids=np.full((capacity,), -1, dtype=np.int64),
            valid=np.zeros((capacity,), dtype=np.bool_),
        )

    @property
    def capacity(self) -> int:
        return self.frames.shape[0]

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return self.frames.shape[1:]

    @property
    def coverage(self) -> float:
        return float(self.valid.mean())

    def update(
        self,
        frames: np.ndarray | Array,
        physical_indices: np.ndarray | Array,
        insertion_ids: np.ndarray | Array,
    ) -> None:
        """Apply one host transfer worth of frames to recorded physical slots."""

        frame_rows = np.asarray(frames, dtype=np.uint8)
        slots = np.asarray(physical_indices, dtype=np.int64)
        logical = np.asarray(insertion_ids, dtype=np.int64)
        if frame_rows.ndim != self.frames.ndim or frame_rows.shape[1:] != self.frame_shape:
            raise ValueError(f"frames must have shape [N, {', '.join(map(str, self.frame_shape))}]")
        count = frame_rows.shape[0]
        if slots.shape != (count,) or logical.shape != (count,):
            raise ValueError("physical_indices and insertion_ids must have shape [N]")
        if np.any(slots < -1) or np.any(slots >= self.capacity):
            raise IndexError("physical frame slot is outside ring capacity")
        keep = (slots >= 0) & (logical >= 0)
        slots = slots[keep]
        self.frames[slots] = frame_rows[keep]
        self.insertion_ids[slots] = logical[keep]
        self.valid[slots] = True

    def invalidate_all(self) -> None:
        """Mark optional frames unavailable after a frame-less restore."""

        self.valid.fill(False)
        self.insertion_ids.fill(-1)

    def available(self, physical_indices: np.ndarray | Array, insertion_ids: np.ndarray | Array) -> np.ndarray:
        """Return whether slots still contain the requested logical memories."""

        slots = np.asarray(physical_indices, dtype=np.int64)
        logical = np.asarray(insertion_ids, dtype=np.int64)
        if slots.shape != logical.shape:
            raise ValueError("physical_indices and insertion_ids must have matching shapes")
        safe_slots = np.clip(slots, 0, self.capacity - 1)
        return (
            (slots >= 0)
            & (logical >= 0)
            & (slots < self.capacity)
            & self.valid[safe_slots]
            & (self.insertion_ids[safe_slots] == logical)
        )

    def coverage_for(self, physical_indices: np.ndarray | Array, insertion_ids: np.ndarray | Array) -> float:
        """Return exact frame coverage for a set of currently resident memories."""

        availability = self.available(physical_indices, insertion_ids)
        return float(availability.mean()) if availability.size else 1.0


class RetrievalResult(NamedTuple):
    context: Array
    weights: Array
    similarities: Array


class EvaluationRetrievalResult(NamedTuple):
    context: Array
    probabilities: Array
    similarities: Array
    sampled_indices: Array


def l2_normalize(x: Array, axis: int = -1, eps: float = 1e-8) -> Array:
    """Normalize vectors while leaving all-zero vectors finite."""

    x = jnp.asarray(x)
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / jnp.maximum(norm, jnp.asarray(eps, dtype=x.dtype))


normalize = l2_normalize


def cosine_similarities(query: Array, candidates: Array) -> Array:
    """Cosine scores for ``[..., D]`` queries and ``[..., K, D]`` candidates."""

    query = l2_normalize(jnp.asarray(query))
    candidates = l2_normalize(jnp.asarray(candidates))
    if candidates.shape[-1] != query.shape[-1]:
        raise ValueError(f"incompatible query {query.shape} and candidates {candidates.shape}")
    if candidates.ndim == 2:
        return jnp.einsum("...d,kd->...k", query, candidates)
    if candidates.ndim != query.ndim + 1 or candidates.shape[:-2] != query.shape[:-1]:
        raise ValueError(f"incompatible query {query.shape} and candidates {candidates.shape}")
    return jnp.einsum("...d,...kd->...k", query, candidates)


def learned_retrieval(query: Array, candidates: Array, temperature: float = 0.1) -> RetrievalResult:
    """Softmax-weighted retrieval with gradients only through the query."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    historical = jax.lax.stop_gradient(jnp.asarray(candidates))
    if historical.shape[-2] == 0:
        raise ValueError("at least one candidate is required")
    similarities = cosine_similarities(query, historical)
    weights = jax.nn.softmax(similarities / temperature, axis=-1)
    context = (
        jnp.einsum("...k,kd->...d", weights, historical)
        if historical.ndim == 2
        else jnp.einsum("...k,...kd->...d", weights, historical)
    )
    return RetrievalResult(context, weights, similarities)


weighted_retrieval = learned_retrieval


def random_retrieval(candidates: Array) -> RetrievalResult:
    """Unweighted candidate mean for the random-memory control."""

    historical = jax.lax.stop_gradient(jnp.asarray(candidates))
    count = historical.shape[-2]
    if count == 0:
        raise ValueError("at least one candidate is required")
    weights = jnp.full(historical.shape[:-1], 1.0 / count, dtype=historical.dtype)
    similarities = jnp.zeros_like(weights)
    return RetrievalResult(jnp.mean(historical, axis=-2), weights, similarities)


def retrieve(mode: str, query: Array, candidates: Array, temperature: float = 0.1) -> RetrievalResult:
    """Dispatch the three experiment modes with a fixed context shape."""

    if mode == "learned":
        return learned_retrieval(query, candidates, temperature)
    if mode == "random":
        return random_retrieval(candidates)
    if mode == "none":
        historical = jax.lax.stop_gradient(jnp.asarray(candidates))
        prefix = historical.shape[:-2]
        context = jnp.zeros((*prefix, historical.shape[-1]), dtype=historical.dtype)
        diagnostics = jnp.zeros(historical.shape[:-1], dtype=historical.dtype)
        return RetrievalResult(context, diagnostics, diagnostics)
    raise ValueError(f"unknown retrieval mode: {mode!r}")


def evaluation_retrieval(
    key: Array,
    query: Array,
    memories: Array,
    count: int,
    temperature: float = 0.1,
) -> EvaluationRetrievalResult:
    """Score the whole memory, sample K indices, then average sampled values."""

    if count <= 0:
        raise ValueError("count must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    historical = jax.lax.stop_gradient(jnp.asarray(memories))
    if historical.shape[-2] == 0:
        raise ValueError("at least one memory is required")
    similarities = cosine_similarities(query, historical)
    probabilities = jax.nn.softmax(similarities / temperature, axis=-1)
    flat_probabilities = probabilities.reshape((-1, probabilities.shape[-1]))
    keys = jax.random.split(key, flat_probabilities.shape[0])

    def sample_one(sample_key: Array, probs: Array) -> Array:
        return jax.random.choice(sample_key, probs.shape[0], shape=(count,), replace=True, p=probs)

    flat_indices = jax.vmap(sample_one)(keys, flat_probabilities)
    sampled_indices = flat_indices.reshape((*probabilities.shape[:-1], count))
    broadcast_memories = jnp.broadcast_to(
        historical,
        (*probabilities.shape[:-1], historical.shape[-2], historical.shape[-1]),
    )
    flat_memories = broadcast_memories.reshape((-1, historical.shape[-2], historical.shape[-1]))
    flat_context = jax.vmap(lambda rows, indices: jnp.mean(rows[indices], axis=0))(flat_memories, flat_indices)
    context = flat_context.reshape((*probabilities.shape[:-1], historical.shape[-1]))
    return EvaluationRetrievalResult(context, probabilities, similarities, sampled_indices)


exact_evaluation_retrieval = evaluation_retrieval
