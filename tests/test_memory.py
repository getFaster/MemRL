from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from memrl.memory import (
    EpisodicMemory,
    evaluation_retrieval,
    l2_normalize,
    learned_retrieval,
    random_retrieval,
)


def test_l2_normalize_has_unit_norm_and_finite_zero() -> None:
    values = jnp.asarray([[3.0, 4.0], [0.0, 0.0]])
    normalized = l2_normalize(values)
    np.testing.assert_allclose(np.asarray(normalized[0]), [0.6, 0.8], atol=1e-6)
    np.testing.assert_array_equal(np.asarray(normalized[1]), [0.0, 0.0])


def test_learned_weights_sum_to_one_and_context_shape() -> None:
    query = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    candidates = jnp.asarray([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], [[1.0, 1.0], [0.0, 2.0], [1.0, 0.0]]])
    result = learned_retrieval(query, candidates)
    assert result.context.shape == (2, 2)
    assert result.weights.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(result.weights.sum(axis=-1)), 1.0, atol=1e-6)


def test_historical_candidates_are_stopped_but_query_has_gradient() -> None:
    query = jnp.asarray([0.7, -0.2, 0.5])
    candidates = jnp.asarray([[0.1, 0.3, 0.5], [0.9, -0.3, 0.2], [-0.4, 0.8, 0.1]])

    candidate_gradient = jax.grad(lambda stored: jnp.sum(learned_retrieval(query, stored).context))(candidates)
    query_gradient = jax.grad(lambda current: jnp.sum(learned_retrieval(current, candidates).context))(query)

    np.testing.assert_array_equal(np.asarray(candidate_gradient), np.zeros((3, 3)))
    assert np.linalg.norm(np.asarray(query_gradient)) > 0


def test_fifo_ring_sampling_and_diagnostics() -> None:
    memory = EpisodicMemory(capacity=3, memory_dim=2, observation_shape=(2, 2), seed=4)
    for step in range(5):
        memory.add(
            np.asarray([step, step + 0.5], dtype=np.float64),
            episode_id=10 + step // 3,
            timestep=step,
            frame=np.full((2, 2), step, dtype=np.uint8),
        )

    resident = memory.all(include_frames=True)
    np.testing.assert_array_equal(resident.logical_indices, [2, 3, 4])
    np.testing.assert_array_equal(resident.timesteps, [2, 3, 4])
    np.testing.assert_array_equal(resident.frames[:, 0, 0], [2, 3, 4])
    assert resident.embeddings.dtype == np.float32
    assert resident.episode_ids.dtype == np.int64
    assert resident.frames.dtype == np.uint8

    sample = memory.sample_candidates(8)
    assert sample.embeddings.shape == (8, 2)
    assert set(sample.logical_indices).issubset({2, 3, 4})
    assert sample.frames is None

    diagnostic = memory.get_by_physical_indices(sample.physical_indices[:2])
    assert diagnostic.frames.shape == (2, 2, 2)


def test_checkpoint_round_trip_preserves_ring_state(tmp_path) -> None:
    memory = EpisodicMemory(capacity=3, memory_dim=2, observation_shape=(1,))
    memory.add(
        np.arange(10, dtype=np.float32).reshape(5, 2),
        episode_id=np.asarray([1, 1, 1, 2, 2]),
        timestep=np.arange(5),
        frame=np.arange(5, dtype=np.uint8).reshape(5, 1),
    )
    memory.save(tmp_path)
    restored = EpisodicMemory.load(tmp_path)

    assert restored.next_index == memory.next_index
    assert restored.total_added == memory.total_added
    original = memory.all(include_frames=True)
    loaded = restored.all(include_frames=True)
    for field in ("embeddings", "physical_indices", "logical_indices", "episode_ids", "timesteps", "frames"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(original, field))


def test_exact_evaluation_scores_whole_memory_and_returns_sample_indices() -> None:
    memories = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = evaluation_retrieval(jax.random.PRNGKey(0), jnp.asarray([1.0, 0.0]), memories, count=7)
    assert result.context.shape == (2,)
    assert result.probabilities.shape == (3,)
    assert result.sampled_indices.shape == (7,)
    np.testing.assert_allclose(np.asarray(result.probabilities.sum()), 1.0, atol=1e-6)
    assert np.all((np.asarray(result.sampled_indices) >= 0) & (np.asarray(result.sampled_indices) < 3))


def test_exact_evaluation_supports_batched_queries_over_shared_memory() -> None:
    memories = jnp.eye(3)
    queries = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = evaluation_retrieval(jax.random.PRNGKey(1), queries, memories, count=4)
    assert result.context.shape == (2, 3)
    assert result.probabilities.shape == (2, 3)
    assert result.sampled_indices.shape == (2, 4)


def test_random_retrieval_is_unweighted_mean() -> None:
    candidates = jnp.asarray([[1.0, 2.0], [3.0, 6.0]])
    result = random_retrieval(candidates)
    np.testing.assert_allclose(np.asarray(result.context), [2.0, 4.0])
    np.testing.assert_allclose(np.asarray(result.weights), [0.5, 0.5])
