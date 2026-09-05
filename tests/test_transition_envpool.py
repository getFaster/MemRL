"""Real CPU EnvPool transition timing across a collection boundary."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memrl.envs import make_envpool
from memrl.memory import create_device_memory
from memrl.train import record_transition


def test_real_terminal_is_stored_and_next_segment_reset_is_skipped() -> None:
    pytest.importorskip("envpool")
    env = make_envpool(num_envs=1, seed=901)
    rng = np.random.default_rng(901)
    try:
        action_dim = env.action_space.n
        # Exercise the real record path without allocating/training a CNN or
        # requiring a GPU; feature values identify the source observation.
        with jax.default_device(jax.devices("cpu")[0]):
            record = jax.jit(partial(record_transition, action_dim=action_dim))
            memory = create_device_memory(8, 512 + action_dim + 1)
            observation, _ = env.reset()
            episode, timestep, inserted = 0, 0, 0
            saw_nonzero_raw_reward = False

            def collect_step():
                nonlocal memory, observation, episode, timestep, inserted, saw_nonzero_raw_reward
                source_features = np.full((1, 512), observation.mean() / 255.0, dtype=np.float32)
                source_episode, source_timestep = episode, timestep
                action = rng.integers(0, action_dim, size=1, dtype=np.int32)
                next_observation, clipped_reward, _, truncated, info = env.step(action)
                raw_reward = float(info["reward"][0])
                saw_nonzero_raw_reward |= raw_reward != 0
                before = memory
                result = record(
                    memory,
                    source_features,
                    action,
                    {"reward": info["reward"], "elapsed_step": info["elapsed_step"]},
                    jnp.asarray([source_episode]),
                    jnp.asarray([source_timestep]),
                )
                memory = result.state
                valid = int(info["elapsed_step"][0]) > 0
                if valid:
                    slot = int(result.physical_indices[0])
                    stored = np.asarray(memory.embeddings[slot])
                    np.testing.assert_array_equal(stored[:512], source_features[0])
                    np.testing.assert_array_equal(stored[512:-1], np.eye(action_dim)[action[0]])
                    np.testing.assert_allclose(stored[-1], np.sign(raw_reward) * np.log1p(abs(raw_reward)), atol=1e-6)
                    assert int(memory.episode_ids[slot]) == source_episode
                    assert int(memory.timesteps[slot]) == source_timestep
                    assert int(memory.insertion_ids[slot]) == inserted
                    np.testing.assert_allclose(clipped_reward[0], np.sign(raw_reward))
                    inserted += 1
                else:
                    assert int(result.physical_indices[0]) == -1
                    for old, new in zip(
                        jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(memory), strict=True
                    ):
                        np.testing.assert_array_equal(old, new)
                assert int(memory.total_insertions) == inserted
                real_done = bool(info["terminated"][0] or truncated[0])
                episode += int(real_done)
                timestep = 0 if real_done else timestep + int(valid)
                observation = next_observation
                return valid, real_done

            # End segment one exactly at a real game boundary. Random actions
            # with this fixed seed terminate in ~350 calls on EnvPool 1.2.5.
            for _ in range(3000):
                valid, terminal = collect_step()
                if terminal:
                    assert valid, "actual terminal transition must be inserted"
                    break
            else:
                pytest.fail("no real Frostbite termination within 3000 deterministic random actions")
            assert saw_nonzero_raw_reward
            terminal_insertions = inserted
            terminal_memory = memory

            # Segment two resumes from the terminal observation. Its first
            # action is discarded by EnvPool's reset, then real actions resume.
            assert collect_step() == (False, False)
            assert inserted == terminal_insertions
            assert int(memory.total_insertions) == int(terminal_memory.total_insertions)
            assert collect_step() == (True, False)
            assert inserted == terminal_insertions + 1
            newest = (int(memory.next_index) - 1) % memory.capacity
            assert int(memory.episode_ids[newest]) == 1
            assert int(memory.timesteps[newest]) == 0
    finally:
        env.close()
