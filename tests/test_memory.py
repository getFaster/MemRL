from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from memrl.memory import (
    HostFrameRing,
    _floyd_offsets,
    create_device_memory,
    evaluation_retrieval,
    insert_batch,
    l2_normalize,
    learned_retrieval,
    random_retrieval,
    sample_batch,
)


def _insert_range(state, start: int, count: int):
    ids = jnp.arange(start, start + count, dtype=jnp.int32)
    embeddings = jnp.stack((ids, ids + 100), axis=-1).astype(jnp.float32)
    return insert_batch(state, embeddings, ids // 3, ids).state


def _primitive_names(closed_jaxpr) -> tuple[str, ...]:
    return tuple(equation.primitive.name for equation in closed_jaxpr.jaxpr.eqns)


def test_empty_memory_returns_invalid_zeros_and_advances_key() -> None:
    state = create_device_memory(8, 2)
    key = jax.random.PRNGKey(4)
    sample = jax.jit(lambda memory, rng: sample_batch(memory, rng, 3, 4))(state, key)
    np.testing.assert_array_equal(sample.embeddings, np.zeros((3, 4, 2), dtype=np.float32))
    np.testing.assert_array_equal(sample.physical_indices, -np.ones((3, 4), dtype=np.int32))
    assert not np.asarray(sample.valid).any()
    assert not np.array_equal(np.asarray(sample.key), np.asarray(key))


def test_batched_insert_wraps_in_env_order_and_preserves_metadata() -> None:
    state = create_device_memory(5, 2)
    first = insert_batch(
        state,
        jnp.asarray([[0, 100], [1, 101], [2, 102]], dtype=jnp.float32),
        jnp.asarray([7, 7, 7]),
        jnp.asarray([0, 1, 2]),
    )
    np.testing.assert_array_equal(first.physical_indices, [0, 1, 2])
    second = insert_batch(
        first.state,
        jnp.asarray([[3, 103], [4, 104], [5, 105]], dtype=jnp.float32),
        jnp.asarray([8, 8, 8]),
        jnp.asarray([3, 4, 5]),
    )
    np.testing.assert_array_equal(second.physical_indices, [3, 4, 0])
    assert int(second.state.size) == 5
    assert int(second.state.next_index) == 1
    assert int(second.state.total_insertions) == 6
    chronological = (int(second.state.next_index) + np.arange(5)) % 5
    np.testing.assert_array_equal(np.asarray(second.state.insertion_ids)[chronological], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.asarray(second.state.timesteps)[chronological], [1, 2, 3, 4, 5])


def test_warmup_samples_with_replacement() -> None:
    state = _insert_range(create_device_memory(8, 2), 0, 3)
    sample = sample_batch(state, jax.random.PRNGKey(12), num_envs=64, k=5)
    assert np.asarray(sample.valid).all()
    assert set(np.asarray(sample.insertion_ids).reshape(-1)).issubset({0, 1, 2})
    assert any(len(set(row)) < 5 for row in np.asarray(sample.insertion_ids))


def test_post_warmup_samples_unique_uniform_subsets() -> None:
    state = _insert_range(create_device_memory(7, 2), 0, 7)
    sample = sample_batch(state, jax.random.PRNGKey(22), num_envs=4096, k=3)
    rows = np.asarray(sample.insertion_ids)
    assert all(len(set(row)) == 3 for row in rows)
    counts = np.bincount(rows.reshape(-1), minlength=7)
    expected = rows.shape[0] * rows.shape[1] / 7
    # The bound is deliberately loose enough to avoid a flaky statistical test
    # while still detecting a materially biased subset implementation.
    assert np.max(np.abs(counts - expected)) < expected * 0.08


def test_sampling_replays_deterministically_and_envs_are_independent() -> None:
    state = _insert_range(create_device_memory(12, 2), 0, 10)
    key = jax.random.PRNGKey(33)
    first = sample_batch(state, key, num_envs=32, k=4)
    replay = sample_batch(state, key, num_envs=32, k=4)
    for field in first._fields:
        np.testing.assert_array_equal(getattr(first, field), getattr(replay, field))
    assert np.unique(np.asarray(first.insertion_ids), axis=0).shape[0] > 1


def test_sampler_structure_is_independent_of_memory_capacity() -> None:
    source = inspect.getsource(_floyd_offsets)
    for forbidden in ("permutation", "gumbel", "random.choice", "arange(size)", "arange(capacity)"):
        assert forbidden not in source

    def traced_primitives(capacity: int) -> tuple[str, ...]:
        state = _insert_range(create_device_memory(capacity, 2), 0, 10)
        traced = jax.make_jaxpr(lambda memory, key: sample_batch(memory, key, 4, 4))(state, jax.random.PRNGKey(1))
        return _primitive_names(traced)

    assert traced_primitives(17) == traced_primitives(100_003)


def test_sampling_before_insert_prevents_self_retrieval_and_snapshot_survives_overwrite() -> None:
    state = _insert_range(create_device_memory(4, 2), 0, 4)
    snapshot = sample_batch(state, jax.random.PRNGKey(44), num_envs=1, k=4)
    before_embeddings = np.asarray(snapshot.embeddings).copy()
    before_ids = np.asarray(snapshot.insertion_ids).copy()
    assert set(before_ids[0]) == {0, 1, 2, 3}

    inserted = insert_batch(
        state,
        jnp.asarray([[10, 110], [11, 111]], dtype=jnp.float32),
        jnp.asarray([4, 4]),
        jnp.asarray([10, 11]),
    )
    assert np.asarray(snapshot.insertion_ids).max() < int(inserted.state.total_insertions)
    np.testing.assert_array_equal(snapshot.embeddings, before_embeddings)
    np.testing.assert_array_equal(snapshot.insertion_ids, before_ids)
    assert {4, 5}.issubset(set(np.asarray(inserted.state.insertion_ids)))


def test_sampled_embeddings_are_detached() -> None:
    state = _insert_range(create_device_memory(4, 2), 0, 4)

    def loss(embeddings):
        updated = state.replace(embeddings=embeddings)
        return sample_batch(updated, jax.random.PRNGKey(1), 1, 4).embeddings.sum()

    gradient = jax.grad(loss)(state.embeddings)
    np.testing.assert_array_equal(gradient, np.zeros_like(state.embeddings))


def test_host_frame_ring_tracks_logical_validity_and_coverage() -> None:
    ring = HostFrameRing.create(5, (2, 2))
    frames = np.stack((np.full((2, 2), 8), np.full((2, 2), 9))).astype(np.uint8)
    ring.update(frames, physical_indices=np.asarray([3, 4]), insertion_ids=np.asarray([8, 9]))
    assert ring.coverage == 0.4
    assert ring.coverage_for([3, 4], [8, 9]) == 1.0
    np.testing.assert_array_equal(ring.available([3, 4], [8, 9]), [True, True])

    ring.update(np.full((1, 2, 2), 10, dtype=np.uint8), [3], [10])
    np.testing.assert_array_equal(ring.available([3, 3, 4, -1], [8, 10, 9, -1]), [False, True, True, False])
    assert ring.coverage_for([3, 4], [8, 9]) == 0.5
    ring.invalidate_all()
    assert ring.coverage == 0.0
    assert not ring.available([3, 4], [10, 9]).any()


def test_l2_normalize_has_unit_norm_and_finite_zero() -> None:
    values = jnp.asarray([[3.0, 4.0], [0.0, 0.0]])
    normalized = l2_normalize(values)
    np.testing.assert_allclose(np.asarray(normalized[0]), [0.6, 0.8], atol=1e-6)
    np.testing.assert_array_equal(np.asarray(normalized[1]), [0.0, 0.0])


def test_historical_candidates_are_stopped_but_query_has_gradient() -> None:
    query = jnp.asarray([0.7, -0.2, 0.5])
    candidates = jnp.asarray([[0.1, 0.3, 0.5], [0.9, -0.3, 0.2], [-0.4, 0.8, 0.1]])
    candidate_gradient = jax.grad(lambda stored: jnp.sum(learned_retrieval(query, stored).context))(candidates)
    query_gradient = jax.grad(lambda current: jnp.sum(learned_retrieval(current, candidates).context))(query)
    np.testing.assert_array_equal(np.asarray(candidate_gradient), np.zeros((3, 3)))
    assert np.linalg.norm(np.asarray(query_gradient)) > 0


def test_exact_evaluation_scores_whole_memory_and_returns_sample_indices() -> None:
    memories = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = evaluation_retrieval(jax.random.PRNGKey(0), jnp.asarray([1.0, 0.0]), memories, count=7)
    assert result.context.shape == (2,)
    assert result.probabilities.shape == (3,)
    assert result.sampled_indices.shape == (7,)
    np.testing.assert_allclose(np.asarray(result.probabilities.sum()), 1.0, atol=1e-6)


def test_random_retrieval_is_unweighted_mean() -> None:
    candidates = jnp.asarray([[1.0, 2.0], [3.0, 6.0]])
    result = random_retrieval(candidates)
    np.testing.assert_allclose(np.asarray(result.context), [2.0, 4.0])
    np.testing.assert_allclose(np.asarray(result.weights), [0.5, 0.5])
