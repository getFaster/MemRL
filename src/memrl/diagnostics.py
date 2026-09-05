"""Small host-side helpers for retrieval diagnostics and frame availability."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

AGE_BIN_EDGES: tuple[int, ...] = (
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
)


@dataclass
class AgeHistogram:
    """Mergeable exact fixed-bin counts; no raw ages need to be retained."""

    counts: np.ndarray = field(default_factory=lambda: np.zeros(len(AGE_BIN_EDGES) - 1, dtype=np.int64))

    def update(self, ages: Iterable[int] | np.ndarray) -> None:
        values = np.asarray(ages, dtype=np.int64).reshape(-1)
        if values.size and (np.any(values < AGE_BIN_EDGES[0]) or np.any(values > AGE_BIN_EDGES[-1])):
            raise ValueError("retrieval ages fall outside the fixed histogram range")
        additions, _ = np.histogram(values, bins=np.asarray(AGE_BIN_EDGES, dtype=np.int64))
        self.counts += additions

    def merge(self, other: AgeHistogram) -> None:
        self.counts += other.counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": list(AGE_BIN_EDGES),
            "counts": self.counts.tolist(),
            "total": int(self.counts.sum()),
        }


def fixed_age_histogram(ages: Iterable[int] | np.ndarray) -> dict[str, Any]:
    histogram = AgeHistogram()
    histogram.update(ages)
    return histogram.to_dict()


def frame_coverage(valid: np.ndarray, occupied: np.ndarray | None = None) -> float:
    """Return valid-frame coverage over occupied slots, with empty coverage defined as 0."""

    valid_array = np.asarray(valid, dtype=bool)
    selected = valid_array if occupied is None else valid_array[np.asarray(occupied, dtype=bool)]
    return float(selected.mean()) if selected.size else 0.0


def table_frame(frames: np.ndarray | None, valid: np.ndarray, physical_slot: int) -> np.ndarray | None:
    """Return a table image only when the host ring has a valid frame for the slot."""

    validity = np.asarray(valid, dtype=bool)
    if physical_slot < 0 or physical_slot >= validity.size:
        raise IndexError(f"physical slot {physical_slot} is outside frame validity ring")
    if frames is None or not validity[physical_slot]:
        return None
    return np.asarray(frames[physical_slot]).copy()
