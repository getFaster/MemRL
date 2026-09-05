from __future__ import annotations

import json

import numpy as np

from memrl.analyze import analyze_records, load_episode_records, trailing_return_curve


def _records():
    records = []
    offsets = {"none": 0.0, "random": 10.0, "learned": 30.0}
    for mode, offset in offsets.items():
        for seed in (1, 2):
            records.extend(
                [
                    {
                        "mode": mode,
                        "seed": seed,
                        "env_slot": 0,
                        "completion_step": 50,
                        "raw_return": offset + seed,
                        "length": 10,
                    },
                    {
                        "mode": mode,
                        "seed": seed,
                        "env_slot": 1,
                        "completion_step": 150,
                        "raw_return": offset + seed + 2,
                        "length": 20,
                    },
                ]
            )
    return records


def test_trailing_curve_backfills_leading_grid_and_uses_latest_window():
    records = [record for record in _records() if record["mode"] == "none" and record["seed"] == 1]
    curve = trailing_return_curve(records, np.asarray([0, 100, 200]), trailing_episodes=1)
    np.testing.assert_allclose(curve, [1.0, 1.0, 3.0])


def test_analysis_emits_normalized_auc_and_all_paired_descriptive_contrasts():
    result = analyze_records(_records(), total_steps=200, grid_points=3, trailing_episodes=100)
    assert result["seeds"] == [1, 2]
    assert set(result["contrasts"]) == {"learned-random", "random-none", "learned-none"}
    none_seed_one = next(row for row in result["runs"] if row["mode"] == "none" and row["seed"] == 1)
    np.testing.assert_allclose(none_seed_one["grid_returns"], [1.0, 1.0, 2.0])
    assert none_seed_one["auc_return_units"] == 1.25
    contrast = result["contrasts"]["learned-random"]
    assert contrast["auc_difference_summary"] == {"mean": 20.0, "median": 20.0, "min": 20.0, "max": 20.0}
    assert len(contrast["per_seed"]) == 2
    serialized = json.dumps(result).lower()
    assert "p_value" not in serialized
    assert "superiority" not in serialized


def test_load_episode_records_accepts_run_directories(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(_records()[0]) + "\n")
    assert load_episode_records([tmp_path]) == [_records()[0]]
