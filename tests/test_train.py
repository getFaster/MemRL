import ast
import inspect

import pytest

from memrl.train import _finite_float, train


def test_nonfinite_update_metrics_fail_fast():
    assert _finite_float(1.25) == 1.25
    with pytest.raises(FloatingPointError):
        _finite_float(float("nan"))
    with pytest.raises(FloatingPointError):
        _finite_float(float("inf"))


def test_compiled_rollout_and_update_regions_have_no_host_callbacks():
    tree = ast.parse(inspect.getsource(train))
    compiled_names = {"rollout_retrieval", "rollout_none", "update_ppo"}
    compiled = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in compiled_names
    }
    assert compiled.keys() == compiled_names

    forbidden = {"callback", "device_get", "io_callback", "pure_callback"}
    for name, function in compiled.items():
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert calls.isdisjoint(forbidden), f"{name} contains a host callback: {calls & forbidden}"


@pytest.mark.parametrize("mode", ["none", "random", "learned"])
def test_memory_dimension_config(mode):
    from memrl.config import TrainConfig

    config = TrainConfig(retrieval_mode=mode)
    assert config.memory_dim is None
    config.validate()
    assert config.resolve_memory_dim(3) == (512 if mode == "none" else 516)
    config.memory_dim = 256
    if mode == "none":
        assert config.resolve_memory_dim(3) == 256
    else:
        with pytest.raises(ValueError, match="memory_dim=516"):
            config.resolve_memory_dim(3)


def test_transition_recording_preserves_source_alignment_and_skips_reset():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from memrl.memory import create_device_memory, sample_batch
    from memrl.train import record_transition

    memory = create_device_memory(8, 516)
    features = jnp.broadcast_to(jnp.arange(3, dtype=jnp.float32)[:, None], (3, 512))
    actions = jnp.array([2, 1, 0])
    info = {
        "reward": jnp.array([100.0, -7.0, -20.0]),
        "elapsed_step": jnp.array([5, 0, 9]),
        "terminated": jnp.array([True, False, False]),
    }
    episodes, steps = jnp.array([11, 12, 13]), jnp.array([4, 0, 8])
    before = sample_batch(memory, jax.random.PRNGKey(3), 3, 2)
    assert not np.asarray(before.valid).any()
    result = jax.jit(record_transition, static_argnums=6)(memory, features, actions, info, episodes, steps, 3)
    np.testing.assert_array_equal(result.physical_indices, [0, -1, 1])
    np.testing.assert_array_equal(result.state.episode_ids[:2], [11, 13])
    np.testing.assert_array_equal(result.state.timesteps[:2], [4, 8])
    np.testing.assert_array_equal(result.state.embeddings[:2, :512], features[jnp.array([0, 2])])
    np.testing.assert_array_equal(result.state.embeddings[:2, 512:-1], [[0, 0, 1], [1, 0, 0]])
    np.testing.assert_allclose(result.state.embeddings[:2, -1], [np.log1p(100), -np.log1p(20)])
    # Real terminal transitions are retained; discarded-action resets are not.
    assert int(result.state.size) == 2
    next_sample = sample_batch(result.state, before.key, 1, 2)
    assert set(np.asarray(next_sample.episode_ids[0])) == {11, 13}
    compiled = str(jax.make_jaxpr(lambda m: record_transition(m, features, actions, info, episodes, steps, 3))(memory))
    assert "callback" not in compiled


def test_rollout_stores_completed_tuple_after_action_and_step_before_episode_advance():
    tree = ast.parse(inspect.getsource(train))
    rollout = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "rollout_retrieval"
    )
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(rollout)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"sample_action", "step_env", "record_transition", "advance_episodes"}
    }
    assert calls["sample_action"] < calls["step_env"] < calls["record_transition"] < calls["advance_episodes"]
    assert "rewards=storage.rewards.at[step].set(reward)" in inspect.getsource(train)
