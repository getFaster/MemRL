from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memrl.models import RetrievalAgent, deterministic_memory_projection, retrieve_memories


def _observations(batch_size: int = 2) -> jax.Array:
    values = jnp.arange(batch_size * 4 * 84 * 84, dtype=jnp.uint32) % 256
    return values.astype(jnp.uint8).reshape(batch_size, 4, 84, 84)


def _candidates(batch_size: int = 2, count: int = 5) -> jax.Array:
    values = jnp.arange(batch_size * count * 256, dtype=jnp.float32)
    return jnp.sin(values / 31.0).reshape(batch_size, count, 256)


def test_deterministic_projection_shape_and_values() -> None:
    z = jnp.arange(1024, dtype=jnp.float32).reshape(2, 512)
    projected = deterministic_memory_projection(z)
    assert projected.shape == (2, 256)
    np.testing.assert_allclose(projected[0, 0], (z[0, 0] + z[0, 1]) / np.sqrt(2))


def test_retrieval_aggregation_and_metadata() -> None:
    query = jnp.ones((2, 256), dtype=jnp.float32)
    candidates = _candidates()
    episode_ids = jnp.arange(10).reshape(2, 5)
    timesteps = episode_ids + 100
    output = retrieve_memories(
        query,
        candidates,
        mode="learned",
        candidate_episode_ids=episode_ids,
        candidate_timesteps=timesteps,
    )
    assert output.context.shape == (2, 256)
    assert output.weights.shape == (2, 5)
    assert output.similarities.shape == (2, 5)
    np.testing.assert_allclose(output.weights.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(output.candidate_episode_ids, episode_ids)
    np.testing.assert_array_equal(output.candidate_timesteps, timesteps)


def test_similarity_bias_changes_learned_retrieval_and_has_gradient() -> None:
    query = jnp.linspace(-1.0, 1.0, 256).reshape(1, 256)
    candidates = _candidates(batch_size=1)
    zero_bias = jnp.zeros((1, 5), dtype=jnp.float32)
    preferred_bias = zero_bias.at[0, 2].set(4.0)

    unbiased = retrieve_memories(query, candidates, mode="learned", similarity_bias=zero_bias)
    biased = retrieve_memories(query, candidates, mode="learned", similarity_bias=preferred_bias)
    assert biased.weights[0, 2] > unbiased.weights[0, 2]
    assert not np.allclose(biased.context, unbiased.context)

    def loss(bias):
        context = retrieve_memories(query, candidates, mode="learned", similarity_bias=bias).context
        return jnp.square(context).sum()

    gradient = jax.grad(loss)(zero_bias)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0.0


def test_random_retrieval_ignores_similarity_bias() -> None:
    query = jnp.linspace(-1.0, 1.0, 256).reshape(1, 256)
    candidates = _candidates(batch_size=1)
    bias = jnp.arange(5, dtype=jnp.float32).reshape(1, 5)
    unbiased = retrieve_memories(query, candidates, mode="random")
    biased = retrieve_memories(query, candidates, mode="random", similarity_bias=bias)
    np.testing.assert_array_equal(biased.weights, unbiased.weights)
    np.testing.assert_array_equal(biased.context, unbiased.context)

    gradient = jax.grad(
        lambda value: retrieve_memories(query, candidates, mode="random", similarity_bias=value).context.sum()
    )(bias)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(bias))


@pytest.mark.parametrize("mode", ["none", "random", "learned"])
def test_agent_mode_switching_and_shapes(mode: str) -> None:
    agent = RetrievalAgent(action_dim=6, retrieval_mode=mode)
    observations = _observations()
    candidates = _candidates() if mode != "none" else None
    similarity_bias = jnp.zeros((2, 5), dtype=jnp.float32) if mode != "none" else None
    variables = agent.init(jax.random.PRNGKey(0), observations, candidates, similarity_bias=similarity_bias)
    output = agent.apply(variables, observations, candidates, similarity_bias=similarity_bias)

    assert output.logits.shape == (2, 6)
    assert output.value.shape == (2, 1)
    assert output.observation_embedding.shape == (2, 512)
    assert output.memory_embedding.shape == (2, 256)
    assert output.retrieval.context.shape == (2, 256)
    if mode == "none":
        assert "query" not in variables["params"]
        assert output.retrieval.weights.shape == (2, 0)
    else:
        assert "query" in variables["params"]
        assert output.retrieval.weights.shape == (2, 5)
        assert variables["params"]["actor"]["output"]["kernel"].shape[0] == 768


def test_encode_method_accepts_channels_last() -> None:
    agent = RetrievalAgent(action_dim=3)
    observations = jnp.transpose(_observations(batch_size=1), (0, 2, 3, 1))
    variables = agent.init(jax.random.PRNGKey(1), observations)
    z, h = agent.apply(variables, observations, method=agent.encode)
    assert z.shape == (1, 512)
    assert h.shape == (1, 256)


def _query_gradient_norm(mode: str) -> float:
    agent = RetrievalAgent(action_dim=4, retrieval_mode=mode)
    observations = _observations(batch_size=1)
    candidates = _candidates(batch_size=1)
    variables = agent.init(jax.random.PRNGKey(2), observations, candidates)

    def loss(params):
        output = agent.apply({"params": params}, observations, candidates)
        return jnp.square(output.logits).sum() + jnp.square(output.value).sum()

    grads = jax.grad(loss)(variables["params"])["query"]
    return float(sum(jnp.vdot(leaf, leaf) for leaf in jax.tree_util.tree_leaves(grads)))


def test_query_receives_gradient_only_in_learned_mode() -> None:
    assert _query_gradient_norm("learned") > 0.0
    assert _query_gradient_norm("random") == 0.0


def test_historical_candidate_embeddings_are_detached() -> None:
    query = jnp.linspace(-1.0, 1.0, 256).reshape(1, 256)
    candidates = _candidates(batch_size=1)

    def loss(candidate_values):
        return retrieve_memories(query, candidate_values, mode="learned").context

    gradients = jax.grad(lambda values: jnp.square(loss(values)).sum())(candidates)
    np.testing.assert_array_equal(gradients, jnp.zeros_like(candidates))
