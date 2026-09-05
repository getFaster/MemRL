from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from memrl.envs import (
    OBSERVATION_SHAPE,
    envpool_config,
    initial_episode_statistics,
    make_envpool,
    normalize_env_id,
    update_episode_statistics,
    validate_envpool_contract,
    validate_envpool_step,
)


class FakeEnvPool:
    def __init__(self, observation_shape=OBSERVATION_SHAPE, action_count=4) -> None:
        self.action_space = SimpleNamespace(n=action_count)
        self.observation_space = SimpleNamespace(shape=observation_shape)

    def xla(self):
        return object(), object(), object(), object()


def test_frostbite_aliases_normalize_to_envpool_task():
    for alias in (
        "Frostbite-v5",
        "ALE/Frostbite-v5",
        "FrostbiteNoFrameskip-v4",
        "ALE/FrostbiteNoFrameskip-v4",
    ):
        assert normalize_env_id(alias) == "Frostbite-v5"
    assert normalize_env_id("Pong-v5") == "Pong-v5"


def test_locked_envpool_configuration_is_explicit():
    config = envpool_config("ALE/Frostbite-v5", num_envs=3, seed=11)
    assert config == {
        "task_id": "Frostbite-v5",
        "env_type": "gym",
        "num_envs": 3,
        "batch_size": 3,
        "seed": [11, 12, 13],
        "frame_skip": 4,
        "stack_num": 4,
        "noop_max": 30,
        "use_fire_reset": True,
        "reward_clip": True,
        "episodic_life": True,
        "img_height": 84,
        "img_width": 84,
        "gray_scale": True,
        "use_inter_area_resize": True,
        "repeat_action_probability": 0.0,
    }


def test_explicit_seed_sequence_must_match_num_envs():
    assert envpool_config("Frostbite-v5", num_envs=2, seed=[101, 303])["seed"] == [101, 303]
    with pytest.raises(ValueError, match="exactly 2"):
        envpool_config("Frostbite-v5", num_envs=2, seed=[101])


def test_make_envpool_calls_native_factory_and_adds_vector_contract(monkeypatch):
    captured = {}
    fake_env = FakeEnvPool()

    def fake_make(task_id, **kwargs):
        captured.update(task_id=task_id, **kwargs)
        return fake_env

    monkeypatch.setattr("memrl.envs.import_module", lambda name: SimpleNamespace(make=fake_make))
    result = make_envpool("FrostbiteNoFrameskip-v4", num_envs=2, seed=7)

    assert result is fake_env
    assert captured["task_id"] == "Frostbite-v5"
    assert captured["seed"] == [7, 8]
    assert captured["env_type"] == "gym"
    assert fake_env.num_envs == 2
    assert fake_env.single_action_space is fake_env.action_space
    assert fake_env.single_observation_space is fake_env.observation_space
    assert fake_env.is_vector_env is True


def test_validate_envpool_contract_rejects_wrong_observations():
    env = FakeEnvPool(observation_shape=(84, 84, 4))
    env.num_envs = 2
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    with pytest.raises(ValueError, match="expected EnvPool observations"):
        validate_envpool_contract(env, num_envs=2)


def test_validate_step_requires_raw_reward_and_real_game_boundaries():
    observation = np.zeros((2, *OBSERVATION_SHAPE), dtype=np.uint8)
    reward = np.asarray([1.0, -1.0], dtype=np.float32)
    terminated = np.asarray([False, True])
    truncated = np.asarray([False, False])
    info = {
        "reward": np.asarray([5.0, -3.0], dtype=np.float32),
        "terminated": np.asarray([False, False]),
    }
    validate_envpool_step(observation, reward, terminated, truncated, info, num_envs=2)
    with pytest.raises(KeyError, match="reward"):
        validate_envpool_step(
            observation,
            reward,
            terminated,
            truncated,
            {key: value for key, value in info.items() if key != "reward"},
            num_envs=2,
        )


def test_episode_accounting_uses_raw_reward_and_ignores_life_loss_done():
    statistics = initial_episode_statistics(2)
    update = jax.jit(update_episode_statistics)

    statistics, events = update(
        statistics,
        {
            "reward": np.asarray([5.0, 2.5]),
            "terminated": np.asarray([False, False]),
        },
        np.asarray([False, False]),
    )
    np.testing.assert_array_equal(np.asarray(events.mask), [False, False])
    np.testing.assert_allclose(np.asarray(statistics.returns), [5.0, 2.5])

    # Slot zero can have an episodic-life `done` outside this helper; with no
    # real termination in info, its real-game accumulator must continue.
    statistics, events = update(
        statistics,
        {
            "reward": np.asarray([7.0, -4.0]),
            "terminated": np.asarray([False, True]),
        },
        np.asarray([False, False]),
    )
    np.testing.assert_array_equal(np.asarray(events.mask), [False, True])
    np.testing.assert_allclose(np.asarray(events.returns), [12.0, -1.5])
    np.testing.assert_array_equal(np.asarray(events.lengths), [2, 2])
    np.testing.assert_allclose(np.asarray(statistics.returns), [12.0, 0.0])
    np.testing.assert_array_equal(np.asarray(statistics.lengths), [2, 0])

    statistics, events = update(
        statistics,
        {
            "reward": np.asarray([1.0, 3.0]),
            "terminated": np.asarray([False, False]),
        },
        np.asarray([True, False]),
    )
    np.testing.assert_array_equal(np.asarray(events.mask), [True, False])
    np.testing.assert_allclose(np.asarray(events.returns), [13.0, 3.0])
    np.testing.assert_allclose(np.asarray(statistics.returns), [0.0, 3.0])


def test_real_envpool_observation_action_and_info_contract_if_available():
    pytest.importorskip("envpool")
    env = make_envpool("Frostbite-v5", num_envs=2, seed=[17, 23])
    try:
        observation, _ = env.reset()
        actions = np.zeros((2,), dtype=np.int32)
        next_observation, reward, terminated, truncated, info = env.step(actions)
        assert np.shape(observation) == (2, *OBSERVATION_SHAPE)
        validate_envpool_step(next_observation, reward, terminated, truncated, info, num_envs=2)
        handle, recv, send, step = env.xla()
        assert handle is not None
        assert all(callable(function) for function in (recv, send, step))
    finally:
        env.close()
