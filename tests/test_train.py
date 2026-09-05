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
    assert config.memory_dim == 512
    config.validate()
    config.memory_dim = 256
    if mode == "none":
        config.validate()
    else:
        with pytest.raises(ValueError, match="memory_dim=512"):
            config.validate()
