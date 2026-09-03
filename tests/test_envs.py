from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from memrl.envs import (
    ChannelFirstObservation,
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
    episode_end_mask,
    extract_final_episode_stats,
    make_env,
)


class MockAtari(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.ale = SimpleNamespace(lives=lambda: self._lives)
        self._lives = 3
        self.steps = 0

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "UP", "DOWN"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._lives = 3
        self.steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        frame = np.full(self.observation_space.shape, self.steps, dtype=np.uint8)
        return frame, 2.5, False, False, {}


def test_cleanrl_gameplay_wrappers_without_rom():
    env = MockAtari()
    env = NoopResetEnv(env, noop_max=1)
    env.override_num_noops = 1
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = ClipRewardEnv(env)

    observation, _ = env.reset(seed=7)
    observation, reward, terminated, truncated, _ = env.step(3)
    assert observation.shape == (210, 160, 3)
    assert reward == 1.0
    assert not terminated and not truncated


def test_channel_first_accepts_channel_last_stack():
    env = MockAtari()
    env.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), dtype=np.uint8)
    wrapped = ChannelFirstObservation(env)
    result = wrapped.observation(np.zeros((84, 84, 4), dtype=np.uint8))
    assert result.shape == (4, 84, 84)
    assert wrapped.observation_space.shape == (4, 84, 84)


def test_full_preprocessing_has_channel_first_shape(monkeypatch):
    monkeypatch.setattr("memrl.envs._make_atari", lambda env_id, render_mode: MockAtari())
    env = make_env("ALE/Frostbite-v5", seed=3, idx=0, capture_video=False, run_name="test")()
    observation, _ = env.reset(seed=3)
    assert observation.shape == (4, 84, 84)
    assert observation.dtype == np.uint8
    observation, reward, *_ = env.step(0)
    assert observation.shape == (4, 84, 84)
    assert reward == 1.0


def test_warp_frame_preserves_constant_luma():
    env = WarpFrame(MockAtari())
    rgb = np.empty((210, 160, 3), dtype=np.uint8)
    rgb[...] = (100, 150, 200)
    observation = env.observation(rgb)
    assert observation.shape == (84, 84)
    assert np.unique(observation).tolist() == [141]


def test_extract_episode_stats_from_vector_final_info():
    infos = {
        "final_info": np.asarray(
            [{"episode": {"r": np.asarray(42.5), "l": np.asarray(123), "t": np.asarray(1.5)}}, None],
            dtype=object,
        ),
        "_final_info": np.asarray([True, False]),
    }
    assert extract_final_episode_stats(infos) == [{"return": 42.5, "length": 123, "time": 1.5}]


def test_extract_episode_stats_from_masked_vector_episode():
    infos = {
        "episode": {
            "r": np.asarray([10.0, 20.0]),
            "l": np.asarray([5, 8]),
            "t": np.asarray([0.1, 0.2]),
        },
        "_episode": np.asarray([False, True]),
    }
    assert extract_final_episode_stats(infos) == [{"return": 20.0, "length": 8, "time": 0.2}]
    np.testing.assert_array_equal(episode_end_mask(infos, 2), [False, True])


def test_episode_end_mask_ignores_life_loss_final_info():
    infos = {
        "final_info": np.asarray([{"lives": 2}, {"episode": {"r": 10.0, "l": 5}}], dtype=object),
        "_final_info": np.asarray([True, True]),
    }
    np.testing.assert_array_equal(episode_end_mask(infos, 2), [False, True])


def test_current_same_step_nested_final_info_layout():
    life_loss = {
        "final_info": {"lives": np.asarray([2]), "_lives": np.asarray([True])},
        "_final_info": np.asarray([True]),
    }
    assert extract_final_episode_stats(life_loss) == []
    np.testing.assert_array_equal(episode_end_mask(life_loss, 1), [False])

    game_over = {
        "final_info": {
            "episode": {
                "r": np.asarray([130.0]),
                "l": np.asarray([900]),
                "t": np.asarray([7.5]),
            },
            "_episode": np.asarray([True]),
        },
        "_final_info": np.asarray([True]),
    }
    assert extract_final_episode_stats(game_over) == [{"return": 130.0, "length": 900, "time": 7.5}]
    np.testing.assert_array_equal(episode_end_mask(game_over, 1), [True])
