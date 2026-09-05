import pytest

from memrl.config import TrainConfig


@pytest.mark.parametrize("mode,expected", [("none", 512), ("random", 519), ("learned", 519)])
def test_memory_width_resolution(mode, expected):
    config = TrainConfig(retrieval_mode=mode)
    config.validate()
    assert config.memory_dim is None
    assert config.resolve_memory_dim(6) == expected
    config.validate()
    assert config.to_dict()["memory_dim"] == expected
    assert config.memory_layout == ("observation_v1" if mode == "none" else "transition_obs_action_symlog_v1")
    assert config.resolve_memory_dim(6) == expected


def test_explicit_observation_only_width_rejected_for_retrieval():
    with pytest.raises(ValueError, match="requires memory_dim=519"):
        TrainConfig(retrieval_mode="learned", memory_dim=512).resolve_memory_dim(6)
    assert TrainConfig(retrieval_mode="learned", memory_dim=519).resolve_memory_dim(6) == 519


def test_invalid_memory_width_and_action_count():
    with pytest.raises(ValueError, match="positive"):
        TrainConfig(memory_dim=0).validate()
    with pytest.raises(ValueError, match="action_dim"):
        TrainConfig().resolve_memory_dim(0)


def test_baseline_retains_unused_explicit_width():
    assert TrainConfig(retrieval_mode="none", memory_dim=256).resolve_memory_dim(6) == 256
