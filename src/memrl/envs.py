"""Gymnasium Atari environments with the preprocessing used by CleanRL PPO.

The wrappers in this module intentionally mirror ``cleanrl/ppo_atari.py``:
episode statistics are recorded before reward clipping and episodic-life
termination, actions are repeated four times, and observations are returned as
four grayscale 84 x 84 frames in channel-first order.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import ale_py

    gym.register_envs(ale_py)
except ImportError:  # Lets wrapper-only unit tests run without the Atari extra.
    ale_py = None


class NoopResetEnv(gym.Wrapper):
    """Take a random number of no-op actions after a real reset."""

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        if noop_max < 1:
            raise ValueError("noop_max must be positive")
        if self.unwrapped.get_action_meanings()[0] != "NOOP":
            raise ValueError("Atari action zero must be NOOP")
        self.noop_max = noop_max
        self.override_num_noops: int | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        noops = self.override_num_noops
        if noops is None:
            noops = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))
        if noops < 1:
            raise ValueError("override_num_noops must be positive")

        for _ in range(noops):
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                # Do not pass the original seed repeatedly: doing so would make
                # every retry select exactly the same no-op count and trajectory.
                retry_kwargs = {key: value for key, value in kwargs.items() if key != "seed"}
                observation, info = self.env.reset(**retry_kwargs)
        return observation, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat an action, sum rewards, and max-pool the final two frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        if skip < 1:
            raise ValueError("skip must be positive")
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("MaxAndSkipEnv requires a Box observation space")
        self._skip = skip
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._obs_buffer.fill(0)
        return self.env.reset(**kwargs)

    def step(self, action: int):
        total_reward = 0.0
        frames: list[np.ndarray] = []
        info: dict[str, Any] = {}
        terminated = truncated = False
        for _ in range(self._skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            frames.append(observation)
            if len(frames) > 2:
                frames.pop(0)
            total_reward += float(reward)
            if terminated or truncated:
                break

        # Using the frames actually observed also handles an episode ending on
        # the first or second repeated action without retaining stale pixels.
        if len(frames) == 1:
            max_frame = frames[0]
        else:
            self._obs_buffer[0] = frames[-2]
            self._obs_buffer[1] = frames[-1]
            max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Expose a lost life as terminal while resetting only on game over."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = int(self.unwrapped.ale.lives())
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if self.was_real_done:
            observation, info = self.env.reset(**kwargs)
        else:
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
        self.lives = int(self.unwrapped.ale.lives())
        return observation, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE and the second action after reset to start an Atari game."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        meanings = self.unwrapped.get_action_meanings()
        if len(meanings) < 3 or meanings[1] != "FIRE":
            raise ValueError("FireResetEnv requires FIRE as action one and at least three actions")

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        observation, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)
        observation, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)
        return observation, info


class ClipRewardEnv(gym.RewardWrapper):
    """Map rewards to their sign, as in the CleanRL Atari baseline."""

    def reward(self, reward: SupportsFloat) -> float:
        return float(np.sign(float(reward)))


def _area_weights(input_size: int, output_size: int) -> np.ndarray:
    """Build pixel-overlap weights equivalent to area downsampling."""

    scale = input_size / output_size
    weights = np.zeros((output_size, input_size), dtype=np.float32)
    for output_index in range(output_size):
        start = output_index * scale
        end = (output_index + 1) * scale
        first = int(np.floor(start))
        last = min(int(np.ceil(end)), input_size)
        for input_index in range(first, last):
            overlap = max(0.0, min(end, input_index + 1) - max(start, input_index))
            weights[output_index, input_index] = overlap / scale
    return weights


class WarpFrame(gym.ObservationWrapper):
    """Area-resize RGB frames and convert them to 84 x 84 grayscale.

    CleanRL normally composes Gymnasium's resize and grayscale wrappers. The
    former imports OpenCV at step time. This equivalent NumPy implementation
    keeps the experiment portable without making OpenCV a runtime dependency.
    """

    def __init__(self, env: gym.Env, height: int = 84, width: int = 84) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("WarpFrame requires a Box observation space")
        shape = env.observation_space.shape
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"expected an RGB observation, got {shape}")
        self._height_weights = _area_weights(shape[0], height)
        self._width_weights = _area_weights(shape[1], width)
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        rgb = np.asarray(observation, dtype=np.float32)
        # Luma coefficients match the RGB-to-gray conversion used by the
        # standard Atari wrapper. Area filtering is linear, so doing it after
        # grayscale conversion yields the same value before integer rounding.
        grayscale = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        resized = self._height_weights @ grayscale @ self._width_weights.T
        return np.clip(np.rint(resized), 0, 255).astype(np.uint8)


class ChannelFirstObservation(gym.ObservationWrapper):
    """Guarantee the PPO input contract ``(4, 84, 84)``."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("ChannelFirstObservation requires a Box observation space")
        shape = env.observation_space.shape
        if shape == (4, 84, 84):
            self._layout = "channel_first"
        elif shape == (84, 84, 4):
            self._layout = "channel_last"
        elif shape == (4, 84, 84, 1):
            self._layout = "trailing_singleton"
        else:
            raise ValueError(f"unexpected stacked Atari observation shape: {shape}")
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation, dtype=np.uint8)
        if self._layout == "channel_last":
            return np.moveaxis(array, -1, 0)
        if self._layout == "trailing_singleton":
            return array[..., 0]
        return array


def _frame_stack(env: gym.Env, stack_size: int) -> gym.Env:
    """Use the spelling/signature available in the installed Gymnasium."""

    wrapper = getattr(gym.wrappers, "FrameStackObservation", None)
    if wrapper is not None:
        return wrapper(env, stack_size=stack_size)
    return gym.wrappers.FrameStack(env, stack_size)


def _frostbite_candidates(env_id: str) -> tuple[str, ...]:
    if env_id == "FrostbiteNoFrameskip-v4":
        return env_id, "ALE/Frostbite-v5"
    if env_id == "Frostbite-v5":
        return env_id, "ALE/Frostbite-v5", "FrostbiteNoFrameskip-v4"
    if env_id == "ALE/Frostbite-v5":
        return env_id, "FrostbiteNoFrameskip-v4"
    return (env_id,)


def _make_atari(env_id: str, render_mode: str | None) -> gym.Env:
    """Create either the legacy or namespaced Frostbite registration."""

    registration_errors = tuple(
        error
        for error in (
            getattr(gym.error, "NameNotFound", None),
            getattr(gym.error, "NamespaceNotFound", None),
            getattr(gym.error, "VersionNotFound", None),
        )
        if error is not None
    )
    last_error: Exception | None = None
    for candidate in _frostbite_candidates(env_id):
        try:
            return gym.make(
                candidate,
                render_mode=render_mode,
                frameskip=1,
                repeat_action_probability=0.0,
            )
        except registration_errors as error:
            last_error = error
    assert last_error is not None
    raise last_error


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    video_dir: str | Path = "videos",
):
    """Return a thunk suitable for :class:`gym.vector.SyncVectorEnv`."""

    def thunk() -> gym.Env:
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = _make_atari(env_id, render_mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, str(Path(video_dir) / run_name))
        # Keep this inside all gameplay wrappers so the reported episode return
        # is the raw game score and life loss does not end a logged episode.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = WarpFrame(env)
        env = _frame_stack(env, 4)
        env = ChannelFirstObservation(env)
        env.action_space.seed(seed)
        return env

    return thunk


def _episode_dicts(infos: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    final_infos = infos.get("final_info")
    if isinstance(final_infos, Mapping):
        # Gymnasium 1.x SAME_STEP vectorization batches nested final info into
        # a dict of arrays rather than an object array of per-env dictionaries.
        final_episode = final_infos.get("episode")
        if isinstance(final_episode, Mapping):
            mask = final_infos.get("_episode", infos.get("_final_info"))
            returns = np.asarray(final_episode.get("r", []))
            for index in range(len(returns)):
                if mask is not None and not bool(np.asarray(mask)[index]):
                    continue
                yield {key: np.asarray(value)[index] for key, value in final_episode.items()}
    elif final_infos is not None:
        mask = infos.get("_final_info")
        for index, final_info in enumerate(final_infos):
            if mask is not None and not bool(mask[index]):
                continue
            if isinstance(final_info, Mapping) and isinstance(final_info.get("episode"), Mapping):
                yield final_info["episode"]

    episode = infos.get("episode")
    if isinstance(episode, Mapping):
        mask = infos.get("_episode")
        returns = np.asarray(episode.get("r", []))
        if returns.ndim == 0:
            yield episode
        else:
            for index in range(len(returns)):
                if mask is not None and not bool(mask[index]):
                    continue
                yield {key: np.asarray(value)[index] for key, value in episode.items()}


def extract_final_episode_stats(infos: Mapping[str, Any]) -> list[dict[str, float | int]]:
    """Normalize Gymnasium single/vector episode info into serializable records.

    This supports both current same-step vector autoreset ``final_info`` and
    older/direct ``episode`` layouts. Missing wall-clock time is represented by
    zero so callers can always log the same keys.
    """

    records: list[dict[str, float | int]] = []
    for episode in _episode_dicts(infos):
        try:
            episode_return = float(np.asarray(episode["r"]).item())
            episode_length = int(np.asarray(episode["l"]).item())
        except (KeyError, TypeError, ValueError):
            continue
        elapsed = float(np.asarray(episode.get("t", 0.0)).item())
        records.append({"return": episode_return, "length": episode_length, "time": elapsed})
    return records


def episode_end_mask(infos: Mapping[str, Any], num_envs: int) -> np.ndarray:
    """Return the vector slots that completed a real game, excluding life loss."""

    result = np.zeros(num_envs, dtype=bool)
    final_infos = infos.get("final_info")
    if isinstance(final_infos, Mapping):
        final_episode = final_infos.get("episode")
        if isinstance(final_episode, Mapping):
            mask = final_infos.get("_episode", infos.get("_final_info"))
            if mask is None:
                count = min(num_envs, len(np.asarray(final_episode.get("r", []))))
                result[:count] = True
            else:
                mask_array = np.asarray(mask, dtype=bool)
                result[: min(num_envs, len(mask_array))] |= mask_array[:num_envs]
    elif final_infos is not None:
        final_mask = infos.get("_final_info")
        for index, final_info in enumerate(final_infos):
            if index >= num_envs or (final_mask is not None and not bool(final_mask[index])):
                continue
            result[index] = isinstance(final_info, Mapping) and isinstance(final_info.get("episode"), Mapping)
    direct_mask = infos.get("_episode")
    if direct_mask is not None:
        direct_mask = np.asarray(direct_mask, dtype=bool)
        result[: min(num_envs, len(direct_mask))] |= direct_mask[:num_envs]
    elif isinstance(infos.get("episode"), Mapping) and num_envs == 1:
        result[0] = True
    return result


__all__ = [
    "ChannelFirstObservation",
    "ClipRewardEnv",
    "EpisodicLifeEnv",
    "FireResetEnv",
    "MaxAndSkipEnv",
    "NoopResetEnv",
    "WarpFrame",
    "extract_final_episode_stats",
    "episode_end_mask",
    "make_env",
]
