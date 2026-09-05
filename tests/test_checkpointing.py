from __future__ import annotations

import numpy as np
import pytest

from memrl.checkpointing import MEMORY_LAYOUT, MemRLCheckpointManager, restore_checkpoint


def _diagnostics() -> dict[str, np.ndarray]:
    return {
        "frames": np.arange(12, dtype=np.uint8).reshape(3, 2, 2),
        "insertion_ids": np.asarray([7, 8, -1]),
        "valid": np.asarray([True, True, False]),
        "occupied": np.asarray([True, True, False]),
    }


@pytest.mark.orbax
def test_orbax_round_trip_with_and_without_frames(tmp_path) -> None:
    manager = MemRLCheckpointManager(tmp_path, keep_periodic=2)
    framed = manager.save("step_1", {"weights": np.arange(4)}, {"global_step": 1}, _diagnostics(), periodic=True)
    bare = manager.save("final", {"weights": np.arange(4) + 1}, {"global_step": 2})
    manager.close()

    framed_bundle = restore_checkpoint(framed)
    np.testing.assert_array_equal(framed_bundle.training["weights"], np.arange(4))
    assert framed_bundle.frame_coverage == 1.0
    assert restore_checkpoint(bare).diagnostics is None


@pytest.mark.orbax
def test_periodic_retention_preserves_final(tmp_path) -> None:
    manager = MemRLCheckpointManager(tmp_path, keep_periodic=2)
    for step in (1, 2, 3):
        manager.save(f"step_{step}", {"step": np.asarray(step)}, {"global_step": step}, periodic=True)
    manager.save("final", {"step": np.asarray(4)}, {"global_step": 4})
    manager.close()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["final", "step_2", "step_3"]


def test_legacy_checkpoint_is_rejected(tmp_path) -> None:
    legacy = tmp_path / "final.msgpack"
    legacy.write_bytes(b"old")
    with pytest.raises(ValueError, match="legacy"):
        restore_checkpoint(legacy)


@pytest.mark.orbax
@pytest.mark.parametrize("mode", ["random", "learned"])
@pytest.mark.parametrize("abstract", [None, {"missing_tensor": np.zeros((1,))}])
@pytest.mark.parametrize("dimension", [256, 512])
def test_observation_only_retrieval_rejected_before_tensor_restore(tmp_path, mode, abstract, monkeypatch, dimension):
    import orbax.checkpoint as ocp

    with MemRLCheckpointManager(tmp_path) as manager:
        checkpoint = manager.save(
            "old",
            {"weights": np.ones((dimension,))},
            {"config": {"retrieval_mode": mode, "memory_dim": dimension}},
        )

    def forbidden_restore(*args, **kwargs):
        pytest.fail("tensor restore ran before dimension validation")

    monkeypatch.setattr(ocp.PyTreeCheckpointHandler, "restore", forbidden_restore)
    with pytest.raises(ValueError, match="expected transition tuple memory layout"):
        restore_checkpoint(checkpoint, abstract)


@pytest.mark.orbax
@pytest.mark.parametrize("mode,dimension", [("random", 519), ("learned", 519), ("none", 256)])
def test_supported_architecture_checkpoint_round_trip(tmp_path, mode, dimension):
    weights = np.arange(2 * dimension, dtype=np.float32).reshape(2, dimension)
    with MemRLCheckpointManager(tmp_path) as manager:
        checkpoint = manager.save(
            "final",
            {"weights": weights},
            {
                "config": {"retrieval_mode": mode, "memory_dim": dimension},
                "action_dim": 6,
                "memory_layout": MEMORY_LAYOUT,
            },
        )
    bundle = restore_checkpoint(checkpoint)
    np.testing.assert_array_equal(bundle.training["weights"], weights)


@pytest.mark.orbax
@pytest.mark.parametrize("mode", ["none", "random", "learned"])
def test_schema2_only_preserves_baseline(tmp_path, mode, monkeypatch):
    import orbax.checkpoint as ocp

    import memrl.checkpointing as checkpointing

    with monkeypatch.context() as old_schema:
        old_schema.setattr(checkpointing, "SCHEMA_VERSION", 2)
        with MemRLCheckpointManager(tmp_path) as manager:
            checkpoint = manager.save(
                "old",
                {"weights": np.ones((2,))},
                {"config": {"retrieval_mode": mode, "memory_dim": 512}},
            )
    if mode == "none":
        np.testing.assert_array_equal(restore_checkpoint(checkpoint).training["weights"], np.ones((2,)))
    else:

        def forbidden_restore(*args, **kwargs):
            pytest.fail("tensor restore ran before schema validation")

        monkeypatch.setattr(ocp.PyTreeCheckpointHandler, "restore", forbidden_restore)
        with pytest.raises(ValueError, match="expected transition tuple memory layout"):
            restore_checkpoint(checkpoint)


@pytest.mark.orbax
@pytest.mark.parametrize(
    "action_dim,dimension,message",
    [
        (None, 519, "action_dim"),
        (0, 519, "action_dim"),
        (True, 519, "action_dim"),
        (6, 512, "expected memory_dim=519"),
    ],
)
def test_tuple_metadata_validated_before_tensor_restore(tmp_path, monkeypatch, action_dim, dimension, message):
    import orbax.checkpoint as ocp

    with MemRLCheckpointManager(tmp_path) as manager:
        checkpoint = manager.save(
            "invalid",
            {"weights": np.ones((2,))},
            {
                "config": {"retrieval_mode": "learned", "memory_dim": dimension},
                "action_dim": action_dim,
                "memory_layout": MEMORY_LAYOUT,
            },
        )

    def forbidden_restore(*args, **kwargs):
        pytest.fail("tensor restore ran before tuple validation")

    monkeypatch.setattr(ocp.PyTreeCheckpointHandler, "restore", forbidden_restore)
    with pytest.raises(ValueError, match=message):
        restore_checkpoint(checkpoint)
