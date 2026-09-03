"""Episodic memory storage and pure JAX retrieval operations.

The storage class deliberately owns NumPy arrays.  This keeps historical
embeddings outside the JAX computation graph; the retrieval functions also
apply ``stop_gradient`` so callers cannot accidentally differentiate through a
candidate array after converting it to a JAX array.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@dataclass(frozen=True)
class MemorySample:
    """A detached selection from :class:`EpisodicMemory`.

    ``physical_indices`` are ring-buffer slots. ``logical_indices`` are
    monotonically increasing insertion IDs and therefore remain meaningful
    after the ring wraps.
    """

    embeddings: np.ndarray
    physical_indices: np.ndarray
    logical_indices: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    frames: np.ndarray | None

    def as_dict(self) -> dict[str, np.ndarray | None]:
        return {
            "embeddings": self.embeddings,
            "physical_indices": self.physical_indices,
            "logical_indices": self.logical_indices,
            "episode_ids": self.episode_ids,
            "timesteps": self.timesteps,
            "frames": self.frames,
        }


class RetrievalResult(NamedTuple):
    context: Array
    weights: Array
    similarities: Array


class EvaluationRetrievalResult(NamedTuple):
    context: Array
    probabilities: Array
    similarities: Array
    sampled_indices: Array


class EpisodicMemory:
    """Fixed-capacity FIFO ring buffer for detached per-step memories."""

    def __init__(
        self,
        capacity: int,
        memory_dim: int,
        observation_shape: Sequence[int] | None = None,
        *,
        frame_shape: Sequence[int] | None = None,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if memory_dim <= 0:
            raise ValueError("memory_dim must be positive")
        if observation_shape is not None and frame_shape is not None:
            if tuple(observation_shape) != tuple(frame_shape):
                raise ValueError("observation_shape and frame_shape disagree")

        shape = observation_shape if observation_shape is not None else frame_shape
        self.capacity = int(capacity)
        self.memory_dim = int(memory_dim)
        self.observation_shape = tuple(int(x) for x in shape) if shape is not None else None
        if self.observation_shape is not None and any(x <= 0 for x in self.observation_shape):
            raise ValueError("observation dimensions must be positive")

        self.embeddings = np.empty((self.capacity, self.memory_dim), dtype=np.float32)
        self.episode_ids = np.empty(self.capacity, dtype=np.int64)
        self.timesteps = np.empty(self.capacity, dtype=np.int64)
        self.insertion_ids = np.empty(self.capacity, dtype=np.int64)
        self.frames = (
            np.empty((self.capacity, *self.observation_shape), dtype=np.uint8)
            if self.observation_shape is not None
            else None
        )
        self._size = 0
        self._next_index = 0
        self._total_added = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    @property
    def next_index(self) -> int:
        """Physical slot that the next insertion will overwrite."""

        return self._next_index

    @property
    def total_added(self) -> int:
        return self._total_added

    def clear(self) -> None:
        self._size = 0
        self._next_index = 0
        self._total_added = 0

    def add(
        self,
        embedding: np.ndarray | Array,
        episode_id: int | np.ndarray,
        timestep: int | np.ndarray,
        frame: np.ndarray | Array | None = None,
    ) -> np.ndarray | np.int64:
        """Append one step or a batch of steps and return their insertion IDs.

        Inputs are copied to fixed NumPy dtypes. A batch must have a single
        leading dimension, with corresponding episode/timestep/frame rows.
        """

        values = np.asarray(embedding, dtype=np.float32)
        scalar = values.ndim == 1
        if scalar:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.memory_dim:
            raise ValueError(
                f"embedding must have shape ({self.memory_dim},) or (N, {self.memory_dim}); got {values.shape}"
            )
        count = values.shape[0]
        episodes = self._metadata_rows(episode_id, count, "episode_id")
        steps = self._metadata_rows(timestep, count, "timestep")
        frame_rows = self._frame_rows(frame, count)
        logical = np.arange(self._total_added, self._total_added + count, dtype=np.int64)

        # If the batch itself exceeds capacity, only its newest portion can be
        # resident. Account for every insertion while writing the retained tail
        # in chronological order.
        if count >= self.capacity:
            keep = slice(count - self.capacity, count)
            physical = (self._next_index + np.arange(count, dtype=np.int64)) % self.capacity
            physical = physical[keep]
            self.embeddings[physical] = values[keep]
            self.episode_ids[physical] = episodes[keep]
            self.timesteps[physical] = steps[keep]
            self.insertion_ids[physical] = logical[keep]
            if self.frames is not None:
                assert frame_rows is not None
                self.frames[physical] = frame_rows[keep]
            self._size = self.capacity
            self._next_index = int((self._next_index + count) % self.capacity)
        else:
            physical = (self._next_index + np.arange(count)) % self.capacity
            self.embeddings[physical] = values
            self.episode_ids[physical] = episodes
            self.timesteps[physical] = steps
            self.insertion_ids[physical] = logical
            if self.frames is not None:
                assert frame_rows is not None
                self.frames[physical] = frame_rows
            self._next_index = int((self._next_index + count) % self.capacity)
            self._size = min(self.capacity, self._size + count)

        self._total_added += count
        return logical[0] if scalar else logical

    # A familiar replay-buffer spelling, useful at call sites.
    append = add

    def sample_candidates(
        self,
        count: int,
        *,
        rng: np.random.Generator | int | None = None,
        replace: bool | None = None,
        include_frames: bool = False,
    ) -> MemorySample:
        """Uniformly sample candidates from the resident memory.

        Sampling automatically uses replacement during memory warm-up so the
        returned candidate dimension remains fixed.
        """

        if count <= 0:
            raise ValueError("count must be positive")
        if not self._size:
            raise ValueError("cannot sample an empty memory")
        generator = (
            self._rng if rng is None else (rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng))
        )
        use_replacement = self._size < count if replace is None else replace
        if not use_replacement and count > self._size:
            raise ValueError("cannot sample more resident memories without replacement")
        resident_offset = generator.choice(self._size, size=count, replace=use_replacement)
        physical = self.chronological_physical_indices()[resident_offset]
        return self._selection(physical, include_frames=include_frames)

    sample = sample_candidates

    def all(self, *, include_frames: bool = False) -> MemorySample:
        """Return all resident entries, oldest to newest."""

        return self._selection(self.chronological_physical_indices(), include_frames=include_frames)

    get_all = all

    def get_by_physical_indices(
        self, physical_indices: np.ndarray | Sequence[int], *, include_frames: bool = True
    ) -> MemorySample:
        """Fetch selected ring slots for periodic retrieval diagnostics."""

        physical = np.asarray(physical_indices, dtype=np.int64)
        if physical.ndim != 1:
            raise ValueError("physical_indices must be one-dimensional")
        if np.any(physical < 0) or np.any(physical >= self.capacity):
            raise IndexError("physical index is outside memory capacity")
        resident = self.chronological_physical_indices()
        if not np.all(np.isin(physical, resident)):
            raise IndexError("physical index does not refer to a resident memory")
        return self._selection(physical, include_frames=include_frames)

    def chronological_physical_indices(self) -> np.ndarray:
        if self._size == 0:
            return np.empty(0, dtype=np.int64)
        oldest = self._next_index if self._size == self.capacity else 0
        return (oldest + np.arange(self._size, dtype=np.int64)) % self.capacity

    def _selection(self, physical: np.ndarray, *, include_frames: bool) -> MemorySample:
        physical = np.asarray(physical, dtype=np.int64)
        return MemorySample(
            embeddings=self.embeddings[physical].copy(),
            physical_indices=physical.copy(),
            logical_indices=self.insertion_ids[physical].copy(),
            episode_ids=self.episode_ids[physical].copy(),
            timesteps=self.timesteps[physical].copy(),
            frames=(self.frames[physical].copy() if include_frames and self.frames is not None else None),
        )

    @staticmethod
    def _metadata_rows(value: int | np.ndarray, count: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.int64)
        if array.ndim == 0:
            return np.full(count, array.item(), dtype=np.int64)
        if array.shape != (count,):
            raise ValueError(f"{name} must be scalar or shape ({count},); got {array.shape}")
        return array.copy()

    def _frame_rows(self, frame: np.ndarray | Array | None, count: int) -> np.ndarray | None:
        if self.frames is None:
            if frame is not None:
                raise ValueError("frame supplied to a memory configured without observation_shape")
            return None
        if frame is None:
            raise ValueError("frame is required when observation_shape is configured")
        array = np.asarray(frame, dtype=np.uint8)
        expected_single = self.observation_shape
        expected_batch = (count, *self.observation_shape)
        if count == 1 and array.shape == expected_single:
            return array[None, ...].copy()
        if array.shape != expected_batch:
            raise ValueError(f"frame must have shape {expected_single} or {expected_batch}; got {array.shape}")
        return array.copy()

    @staticmethod
    def _checkpoint_paths(path: str | Path) -> tuple[Path, Path]:
        path = Path(path)
        if path.suffix == ".npz":
            path.parent.mkdir(parents=True, exist_ok=True)
            return path, path.with_name(f"{path.stem}.frames.npy")
        path.mkdir(parents=True, exist_ok=True)
        return path / "memory.npz", path / "frames.npy"

    def save(self, path: str | Path, *, include_frames: bool = True) -> None:
        """Save occupied ring slots; frames use a separate uncompressed NPY."""

        metadata_path, frames_path = self._checkpoint_paths(path)
        occupied = self.chronological_physical_indices()
        np.savez_compressed(
            metadata_path,
            capacity=np.asarray(self.capacity, dtype=np.int64),
            memory_dim=np.asarray(self.memory_dim, dtype=np.int64),
            observation_shape=np.asarray(self.observation_shape or (), dtype=np.int64),
            size=np.asarray(self._size, dtype=np.int64),
            next_index=np.asarray(self._next_index, dtype=np.int64),
            total_added=np.asarray(self._total_added, dtype=np.int64),
            physical_indices=occupied,
            embeddings=self.embeddings[occupied],
            episode_ids=self.episode_ids[occupied],
            timesteps=self.timesteps[occupied],
            insertion_ids=self.insertion_ids[occupied],
            frames_saved=np.asarray(bool(include_frames and self.frames is not None)),
        )
        if include_frames and self.frames is not None:
            np.save(frames_path, self.frames[occupied], allow_pickle=False)

    save_checkpoint = save

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0) -> EpisodicMemory:
        metadata_path, frames_path = cls._checkpoint_paths(path)
        with np.load(metadata_path, allow_pickle=False) as data:
            observation_shape = tuple(int(x) for x in data["observation_shape"])
            memory = cls(
                capacity=int(data["capacity"]),
                memory_dim=int(data["memory_dim"]),
                observation_shape=observation_shape or None,
                seed=seed,
            )
            physical = data["physical_indices"].astype(np.int64, copy=False)
            memory.embeddings[physical] = data["embeddings"]
            memory.episode_ids[physical] = data["episode_ids"]
            memory.timesteps[physical] = data["timesteps"]
            memory.insertion_ids[physical] = data["insertion_ids"]
            memory._size = int(data["size"])
            memory._next_index = int(data["next_index"])
            memory._total_added = int(data["total_added"])
            frames_saved = bool(data["frames_saved"])
        if frames_saved:
            if not frames_path.exists():
                raise FileNotFoundError(f"frame checkpoint is missing: {frames_path}")
            assert memory.frames is not None
            memory.frames[physical] = np.load(frames_path, allow_pickle=False)
        elif memory.frames is not None and physical.size:
            memory.frames[physical] = 0
        return memory

    load_checkpoint = load


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


def learned_retrieval(
    query: Array,
    candidates: Array,
    temperature: float = 0.1,
) -> RetrievalResult:
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


def retrieve(
    mode: str,
    query: Array,
    candidates: Array,
    temperature: float = 0.1,
) -> RetrievalResult:
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
    """Score the whole memory, sample indices, then average sampled values.

    This implements final evaluation retrieval without candidate subsampling.
    Batched queries and memories are supported; their leading dimensions must
    match. Sampling is with replacement, as specified by categorical draws.
    """

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
