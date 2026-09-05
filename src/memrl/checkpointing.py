"""Orbax directory checkpoints for resumable MemRL training."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
FORMAT_NAME = "memrl-orbax"


@dataclass(frozen=True)
class CheckpointBundle:
    training: Any
    metadata: dict[str, Any]
    diagnostics: dict[str, np.ndarray] | None

    @property
    def frame_coverage(self) -> float:
        if self.diagnostics is None:
            return 0.0
        valid = np.asarray(self.diagnostics["valid"], dtype=bool)
        occupied = np.asarray(self.diagnostics.get("occupied", np.ones_like(valid)), dtype=bool)
        return float(valid[occupied].mean()) if occupied.any() else 0.0


def _orbax():
    import orbax.checkpoint as ocp

    return ocp


def _handler():
    ocp = _orbax()
    return ocp.CompositeCheckpointHandler(
        training=ocp.PyTreeCheckpointHandler(),
        metadata=ocp.JsonCheckpointHandler(),
        diagnostics=ocp.PyTreeCheckpointHandler(),
    )


def _validate_checkpoint_directory(path: Path) -> None:
    if path.is_file() or path.suffix in {".msgpack", ".npz"}:
        raise ValueError("legacy MemRL checkpoint files are unsupported; no migration is provided")
    required = {"training", "metadata", "diagnostics", "_CHECKPOINT_METADATA"}
    if not path.is_dir() or not required.issubset(child.name for child in path.iterdir()):
        raise ValueError(f"not a {FORMAT_NAME} checkpoint directory: {path}")


class MemRLCheckpointManager:
    """Own one asynchronous Orbax writer and periodic retention policy."""

    def __init__(self, root: Path, *, keep_periodic: int = 2) -> None:
        if keep_periodic < 1:
            raise ValueError("keep_periodic must be positive")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_periodic = keep_periodic
        ocp = _orbax()
        self._checkpointer = ocp.AsyncCheckpointer(_handler())

    def _destination(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("checkpoint name must be one path component")
        return self.root / name

    def save(
        self,
        name: str,
        training: Any,
        metadata: Mapping[str, Any],
        diagnostics: Mapping[str, Any] | None = None,
        *,
        periodic: bool = False,
    ) -> Path:
        """Start a coherent composite checkpoint and return its directory."""

        destination = self._destination(name)
        if destination.exists():
            raise FileExistsError(destination)
        full_metadata = {
            **dict(metadata),
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_NAME,
            "frames_saved": diagnostics is not None,
        }
        if diagnostics is None:
            diagnostic_item: dict[str, Any] = {
                # Orbax refuses zero-size arrays. This one-element sentinel is
                # ignored because metadata.frames_saved is false.
                "frames": np.zeros((1,), dtype=np.uint8),
                "insertion_ids": np.asarray([-1], dtype=np.int64),
                "valid": np.asarray([False], dtype=np.bool_),
                "occupied": np.asarray([False], dtype=np.bool_),
            }
        else:
            diagnostic_item = {key: np.asarray(value) for key, value in diagnostics.items()}
            required = {"frames", "insertion_ids", "valid", "occupied"}
            missing = required - diagnostic_item.keys()
            if missing:
                raise ValueError(f"diagnostic checkpoint is missing: {sorted(missing)}")
        ocp = _orbax()
        self._checkpointer.save(
            destination,
            args=ocp.args.Composite(
                training=ocp.args.PyTreeSave(training),
                metadata=ocp.args.JsonSave(full_metadata),
                diagnostics=ocp.args.PyTreeSave(diagnostic_item),
            ),
        )
        if periodic:
            # Rotation is destructive, so it happens only after the new
            # checkpoint is durably finalized.
            self.wait()
            self._rotate_periodic()
        return destination

    def wait(self) -> None:
        self._checkpointer.wait_until_finished()

    def _rotate_periodic(self) -> None:
        periodic = []
        for path in self.root.glob("step_*"):
            try:
                step = int(path.name.removeprefix("step_"))
            except ValueError:
                continue
            if path.is_dir():
                periodic.append((step, path))
        for _, path in sorted(periodic)[: -self.keep_periodic]:
            shutil.rmtree(path)

    def close(self) -> None:
        self.wait()
        self._checkpointer.close()

    def __enter__(self) -> MemRLCheckpointManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def restore_checkpoint(path: Path, training_abstract: Any | None = None) -> CheckpointBundle:
    """Restore one rewritten checkpoint, rejecting legacy formats."""

    checkpoint_path = Path(path).resolve()
    _validate_checkpoint_directory(checkpoint_path)
    ocp = _orbax()
    metadata_reader = ocp.Checkpointer(ocp.JsonCheckpointHandler())
    try:
        metadata = dict(metadata_reader.restore(checkpoint_path / "metadata", args=ocp.args.JsonRestore()))
    finally:
        metadata_reader.close()
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("format") != FORMAT_NAME:
        raise ValueError("unsupported MemRL checkpoint schema")
    saved_config = metadata.get("config", {})
    if saved_config.get("retrieval_mode") in {"random", "learned"} and saved_config.get("memory_dim") != 512:
        raise ValueError(
            "incompatible retrieval checkpoint: expected memory_dim=512, "
            f"got {saved_config.get('memory_dim')}; start a fresh run (no migration is provided)"
        )
    checkpointer = ocp.Checkpointer(_handler())
    try:
        restored = checkpointer.restore(
            checkpoint_path,
            args=ocp.args.Composite(
                training=ocp.args.PyTreeRestore(item=training_abstract),
                metadata=ocp.args.JsonRestore(),
                diagnostics=ocp.args.PyTreeRestore(),
            ),
        )
    finally:
        checkpointer.close()
    diagnostics = None
    if metadata.get("frames_saved"):
        diagnostics = {key: np.asarray(value) for key, value in restored.diagnostics.items()}
    return CheckpointBundle(training=restored.training, metadata=metadata, diagnostics=diagnostics)
