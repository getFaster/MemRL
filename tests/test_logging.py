from __future__ import annotations

import json

import pytest

from memrl.logging import MetricMetadata, RunLogger


def _logger(tmp_path):
    return RunLogger(
        tmp_path,
        {"retrieval_mode": "learned", "seed": 7},
        "test-run",
        "disabled",
        "test-project",
        None,
        None,
    )


def test_logger_writes_metric_metadata_and_episode_rows(tmp_path):
    logger = _logger(tmp_path)
    logger.register_metric_metadata("probe/custom", MetricMetadata("periodic_probe", "every 10 rollouts"))
    logger.log({"retrieval/recent_under_500_fraction": 0.75, "probe/custom": 2.0}, 1_024)
    logger.log_episode(env_slot=3, completion_step=1_024, raw_return=42.5, length=123)
    logger.close()

    metadata = json.loads((tmp_path / "metric_metadata.json").read_text())
    assert metadata["metrics"]["retrieval/recent_under_500_fraction"]["fidelity"] == "exact_rollout_reduction"
    assert metadata["metrics"]["probe/custom"] == {
        "cadence": "every 10 rollouts",
        "description": "",
        "fidelity": "periodic_probe",
    }
    episode = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert episode == {
        "mode": "learned",
        "seed": 7,
        "env_slot": 3,
        "completion_step": 1_024,
        "raw_return": 42.5,
        "length": 123,
    }


def test_logger_rejects_nonfinite_episode_return(tmp_path):
    logger = _logger(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        logger.log_episode(env_slot=0, completion_step=1, raw_return=float("nan"), length=1)
    logger.close()
