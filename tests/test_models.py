from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memrl.models import (
    AtariEncoder,
    QueryNetwork,
    RetrievalAgent,
    l2_normalize,
    retrieve_memories,
    symlog,
    transition_embedding,
)


def _observations(batch_size: int = 2) -> jax.Array:
    values = jnp.arange(batch_size * 4 * 84 * 84, dtype=jnp.uint32) % 256
    return values.astype(jnp.uint8).reshape(batch_size, 4, 84, 84)


def _candidates(batch_size: int = 2, count: int = 5, width: int = 512) -> jax.Array:
    values = jnp.arange(batch_size * count * width, dtype=jnp.float32)
    return jnp.sin(values / 31.0).reshape(batch_size, count, width)


def test_retrieval_aggregation_and_metadata() -> None:
    query = jnp.ones((2, 512), dtype=jnp.float32)
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
    assert output.context.shape == (2, 512)
    assert output.weights.shape == (2, 5)
    assert output.similarities.shape == (2, 5)
    np.testing.assert_allclose(output.weights.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(output.candidate_episode_ids, episode_ids)
    np.testing.assert_array_equal(output.candidate_timesteps, timesteps)


def test_similarity_bias_changes_learned_retrieval_and_has_gradient() -> None:
    query = jnp.linspace(-1.0, 1.0, 512).reshape(1, 512)
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
    query = jnp.linspace(-1.0, 1.0, 512).reshape(1, 512)
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


def test_random_policy_logits_do_not_depend_on_query_parameters() -> None:
    agent = RetrievalAgent(action_dim=4, retrieval_mode="random")
    observations = _observations(batch_size=1)
    candidates = _candidates(batch_size=1, width=517)
    variables = agent.init(jax.random.PRNGKey(7), observations, candidates)
    baseline = agent.apply(variables, observations, candidates).logits
    perturbed = copy.deepcopy(variables)
    perturbed["params"]["query"] = jax.tree.map(lambda value: value + 1000.0, variables["params"]["query"])
    changed = agent.apply(perturbed, observations, candidates).logits
    np.testing.assert_array_equal(changed, baseline)


def test_random_query_cosine_path_is_explicit_probe_only() -> None:
    agent = RetrievalAgent(action_dim=4, retrieval_mode="random")
    observations = _observations(batch_size=1)
    candidates = _candidates(batch_size=1, width=517)
    variables = agent.init(jax.random.PRNGKey(8), observations, candidates)
    output = agent.apply(variables, observations, candidates)
    np.testing.assert_array_equal(output.query, jnp.zeros((1, 517)))
    np.testing.assert_array_equal(output.retrieval.similarities, jnp.zeros((1, 5)))
    query, similarities, _ = agent.apply(variables, observations, candidates, method=agent.retrieval_probe)
    assert jnp.linalg.norm(query) > 0
    assert jnp.linalg.norm(similarities) > 0


@pytest.mark.parametrize("mode", ["none", "random", "learned"])
def test_agent_mode_switching_and_shapes(mode: str) -> None:
    agent = RetrievalAgent(action_dim=6, retrieval_mode=mode)
    observations = _observations()
    candidates = _candidates(width=519) if mode != "none" else None
    similarity_bias = jnp.zeros((2, 5), dtype=jnp.float32) if mode != "none" else None
    variables = agent.init(jax.random.PRNGKey(0), observations, candidates, similarity_bias=similarity_bias)
    output = agent.apply(variables, observations, candidates, similarity_bias=similarity_bias)

    for leaf in jax.tree_util.tree_leaves(output):
        assert np.isfinite(np.asarray(leaf)).all()
    assert output.logits.shape == (2, 6)
    assert output.value.shape == (2, 1)
    assert output.observation_embedding.shape == (2, 512)
    assert output.memory_embedding.shape == (2, 512)
    np.testing.assert_array_equal(output.memory_embedding, output.observation_embedding)
    assert output.query.shape == (2, 512 if mode == "none" else 519)
    expected_width = 512 if mode == "none" else 1031
    for head in ("actor", "critic"):
        assert variables["params"][head]["output"]["kernel"].shape[0] == expected_width
    assert output.retrieval.context.shape == (2, 512 if mode == "none" else 519)
    if mode == "none":
        assert "query" not in variables["params"]
        assert output.retrieval.weights.shape == (2, 0)
    else:
        assert "query" in variables["params"]
        for layer in ("dense1", "dense2"):
            assert variables["params"]["query"][layer]["kernel"].shape == (512, 512 if layer == "dense1" else 519)
        assert output.retrieval.weights.shape == (2, 5)
        assert variables["params"]["actor"]["output"]["kernel"].shape[0] == 1031


def test_encode_method_accepts_channels_last() -> None:
    agent = RetrievalAgent(action_dim=3)
    observations = jnp.transpose(_observations(batch_size=1), (0, 2, 3, 1))
    variables = agent.init(jax.random.PRNGKey(1), observations)
    z, h = agent.apply(variables, observations, method=agent.encode)
    assert z.shape == (1, 512)
    assert h.shape == (1, 512)


def _query_gradient_norm(mode: str) -> float:
    agent = RetrievalAgent(action_dim=4, retrieval_mode=mode)
    observations = _observations(batch_size=1)
    candidates = _candidates(batch_size=1, width=517)
    variables = agent.init(jax.random.PRNGKey(2), observations, candidates)

    def loss(params):
        output = agent.apply({"params": params}, observations, candidates)
        return jnp.square(output.logits).sum() + jnp.square(output.value).sum()

    all_grads = jax.grad(loss)(variables["params"])
    for leaf in jax.tree_util.tree_leaves(all_grads):
        assert np.isfinite(np.asarray(leaf)).all()
    grads = all_grads["query"]
    return float(sum(jnp.vdot(leaf, leaf) for leaf in jax.tree_util.tree_leaves(grads)))


def test_query_receives_gradient_only_in_learned_mode() -> None:
    assert _query_gradient_norm("learned") > 0.0
    assert _query_gradient_norm("random") == 0.0


def test_historical_candidate_embeddings_are_detached() -> None:
    query = jnp.linspace(-1.0, 1.0, 512).reshape(1, 512)
    candidates = _candidates(batch_size=1)

    def loss(candidate_values):
        return retrieve_memories(query, candidate_values, mode="learned").context

    gradients = jax.grad(lambda values: jnp.square(loss(values)).sum())(candidates)
    np.testing.assert_array_equal(gradients, jnp.zeros_like(candidates))


@pytest.mark.parametrize("mode", ["random", "learned"])
def test_external_context_matches_policy_and_rejects_old_width(mode):
    agent = RetrievalAgent(action_dim=4, retrieval_mode=mode)
    observations = _observations(batch_size=1)
    candidates = _candidates(batch_size=1, width=517)
    variables = agent.init(jax.random.PRNGKey(9), observations, candidates)
    output = agent.apply(variables, observations, candidates)
    logits, value, query, h = agent.apply(
        variables, observations, output.retrieval.context, method=agent.apply_retrieved_context
    )
    for actual, expected in (
        (logits, output.logits),
        (value, output.value),
        (query, output.query),
        (h, output.memory_embedding),
    ):
        np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="retrieval context must have shape"):
        agent.apply(variables, observations, jnp.zeros((1, 256)), method=agent.apply_retrieved_context)


def test_encoder_applies_approximate_gelu_at_all_four_layers():
    encoder = AtariEncoder()
    observations = jnp.full((1, 4, 84, 84), 255, dtype=jnp.uint8)
    variables = encoder.init(jax.random.PRNGKey(20), observations)
    # A single unit-weight path isolates the four activation functions.
    for layer in variables["params"].values():
        kernel = jnp.zeros_like(layer["kernel"])
        layer["kernel"] = kernel.at[(0,) * kernel.ndim].set(1.0)
        layer["bias"] = jnp.zeros_like(layer["bias"])
    expected = jnp.asarray(1.0)
    for _ in range(4):
        expected = jax.nn.gelu(expected, approximate=True)
    output = encoder.apply(variables, observations)
    np.testing.assert_allclose(output[0, 0], expected, rtol=1e-6)
    np.testing.assert_array_equal(output[0, 1:], jnp.zeros(511))


def test_query_uses_approximate_gelu_and_linear_output():
    query = QueryNetwork()
    inputs = jnp.linspace(-3.0, 3.0, 512).reshape(1, 512)
    variables = query.init(jax.random.PRNGKey(21), inputs)
    for layer in variables["params"].values():
        layer["kernel"] = jnp.eye(512)
        layer["bias"] = jnp.zeros(512)
    np.testing.assert_allclose(
        query.apply(variables, inputs), jax.nn.gelu(inputs, approximate=True), rtol=1e-6, atol=1e-7
    )


def test_unit_orthogonal_initialization_preserves_head_gains():
    agent = RetrievalAgent(action_dim=6, retrieval_mode="learned")
    params = agent.init(jax.random.PRNGKey(22), _observations(1), _candidates(1, width=519))["params"]
    for module_name, layers in params.items():
        gain = 0.01 if module_name == "actor" else 1.0
        for layer in layers.values():
            kernel = np.asarray(layer["kernel"]).reshape(-1, layer["kernel"].shape[-1])
            np.testing.assert_allclose(
                kernel.T @ kernel if kernel.shape[0] >= kernel.shape[1] else kernel @ kernel.T,
                gain**2 * np.eye(min(kernel.shape)),
                atol=2e-6,
            )
            np.testing.assert_array_equal(layer["bias"], np.zeros_like(layer["bias"]))


def test_transition_embedding_alignment_and_detachment():
    features = jnp.arange(3 * 512, dtype=jnp.float32).reshape(3, 512)
    actions = jnp.array([2, 0, 1])
    rewards = jnp.array([-9.0, 0.0, 99.0])
    result = jax.jit(lambda z, r: transition_embedding(z, actions, r, 3))(features, rewards)
    np.testing.assert_array_equal(result[:, :512], features)
    np.testing.assert_array_equal(result[:, 512:515], np.eye(3)[actions])
    np.testing.assert_allclose(result[:, -1], [-np.log(10), 0, np.log(100)], rtol=1e-6)
    gradients = jax.grad(lambda z, r: transition_embedding(z, actions, r, 3).sum(), argnums=(0, 1))(features, rewards)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))
    np.testing.assert_array_equal(symlog(jnp.zeros(3)), jnp.zeros(3))


def test_zero_query_normalization_has_finite_gradient():
    query = jnp.zeros((1, 517))
    candidates = _candidates(1, width=517)
    gradient = jax.grad(lambda q: retrieve_memories(q, candidates, mode="learned").context.sum())(query)
    assert np.isfinite(gradient).all()
    assert jnp.linalg.norm(gradient) > 0
    np.testing.assert_array_equal(l2_normalize(query), query)
