"""Descriptive episode-level analysis for rewritten-stack MemRL runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

MODES = ("none", "random", "learned")
CONTRASTS = (("learned", "random"), ("random", "none"), ("learned", "none"))
EPISODE_FIELDS = {"mode", "seed", "env_slot", "completion_step", "raw_return", "length"}


@dataclass
class AnalyzeConfig:
    inputs: tuple[Path, ...]
    evaluation_summaries: tuple[Path, ...] = ()
    output: Path = Path("analysis.json")
    total_steps: int = 10_000_000
    grid_points: int = 101
    trailing_episodes: int = 100


def load_episode_records(paths: tuple[Path, ...] | list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for input_path in paths:
        path = input_path / "episodes.jsonl" if input_path.is_dir() else input_path
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = EPISODE_FIELDS - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing episode fields: {sorted(missing)}")
            if record["mode"] not in MODES:
                raise ValueError(f"{path}:{line_number} has invalid mode {record['mode']!r}")
            if not np.isfinite(float(record["raw_return"])):
                raise ValueError(f"{path}:{line_number} has non-finite raw_return")
            records.append(record)
    return records


def trailing_return_curve(records: list[dict[str, Any]], grid: np.ndarray, trailing_episodes: int = 100) -> np.ndarray:
    ordered = sorted(records, key=lambda item: (int(item["completion_step"]), int(item["env_slot"])))
    if not ordered:
        raise ValueError("cannot construct a curve without completed episodes")
    steps = np.asarray([int(item["completion_step"]) for item in ordered], dtype=np.int64)
    returns = np.asarray([float(item["raw_return"]) for item in ordered], dtype=np.float64)
    curve = np.full(grid.shape, np.nan, dtype=np.float64)
    for index, step in enumerate(grid):
        completed = int(np.searchsorted(steps, step, side="right"))
        if completed:
            curve[index] = returns[max(0, completed - trailing_episodes) : completed].mean()
    populated = np.flatnonzero(np.isfinite(curve))
    if not populated.size:
        raise ValueError("no episodes complete on or before the final analysis grid step")
    curve[: populated[0]] = curve[populated[0]]
    return curve


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _curve_summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def analyze_records(
    records: list[dict[str, Any]],
    *,
    total_steps: int = 10_000_000,
    grid_points: int = 101,
    trailing_episodes: int = 100,
    evaluation_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if total_steps <= 0 or grid_points < 2 or trailing_episodes < 1:
        raise ValueError("analysis dimensions must be positive and grid_points must be at least two")
    grid = np.linspace(0, total_steps, grid_points, dtype=np.int64)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["mode"]), int(record["seed"])), []).append(record)

    seeds_by_mode = {mode: {seed for candidate_mode, seed in grouped if candidate_mode == mode} for mode in MODES}
    common_seeds = sorted(set.intersection(*(seeds_by_mode[mode] for mode in MODES)))
    if not common_seeds:
        raise ValueError("analysis requires at least one seed completed in all three modes")

    curves: dict[tuple[str, int], np.ndarray] = {}
    aucs: dict[tuple[str, int], float] = {}
    runs = []
    for mode in MODES:
        for seed in common_seeds:
            curve = trailing_return_curve(grouped[(mode, seed)], grid, trailing_episodes)
            auc = float(np.trapezoid(curve, grid) / total_steps)
            curves[(mode, seed)] = curve
            aucs[(mode, seed)] = auc
            runs.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "grid_returns": curve.tolist(),
                    "auc_return_units": auc,
                    "final_window_return": float(curve[-1]),
                    "episodes": len(grouped[(mode, seed)]),
                }
            )

    contrasts: dict[str, Any] = {}
    for left, right in CONTRASTS:
        difference_curves = np.stack([curves[(left, seed)] - curves[(right, seed)] for seed in common_seeds])
        auc_differences = np.asarray([aucs[(left, seed)] - aucs[(right, seed)] for seed in common_seeds])
        final_differences = difference_curves[:, -1]
        contrasts[f"{left}-{right}"] = {
            "per_seed": [
                {
                    "seed": seed,
                    "difference_curve": difference_curves[index].tolist(),
                    "auc_difference": float(auc_differences[index]),
                    "final_window_difference": float(final_differences[index]),
                }
                for index, seed in enumerate(common_seeds)
            ],
            "difference_curve_summary": _curve_summary(difference_curves),
            "auc_difference_summary": _summary(auc_differences),
            "final_window_difference_summary": _summary(final_differences),
        }

    result = {
        "schema_version": 1,
        "grid_steps": grid.tolist(),
        "trailing_episodes": trailing_episodes,
        "seeds": common_seeds,
        "runs": runs,
        "contrasts": contrasts,
    }
    if evaluation_summaries:
        result["final_checkpoint_evaluation"] = evaluation_summaries
    return result


def main() -> None:
    config = tyro.cli(AnalyzeConfig)
    result = analyze_records(
        load_episode_records(config.inputs),
        total_steps=config.total_steps,
        grid_points=config.grid_points,
        trailing_episodes=config.trailing_episodes,
        evaluation_summaries=[json.loads(path.read_text()) for path in config.evaluation_summaries],
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
