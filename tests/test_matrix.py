from __future__ import annotations

import json

import pytest

from memrl.matrix import MatrixConfig, build_commands, run_commands, validate_config


def _passing_report():
    return {
        "concurrency_canary": {
            "learned_processes": 2,
            "full_host_frames": True,
            "frame_complete_checkpoint": True,
            "combined_peak_vram_bytes": 4 * 1024**3,
            "min_mem_available_bytes": 3 * 1024**3,
            "swap_growth_bytes": 128 * 1024**2,
        }
    }


def test_parallel_two_requires_a_passing_resource_report(tmp_path):
    with pytest.raises(ValueError, match="resource-report"):
        validate_config(MatrixConfig(max_parallel=2))

    report = tmp_path / "resources.json"
    report.write_text(json.dumps(_passing_report()))
    validate_config(MatrixConfig(max_parallel=2, resource_report=report))

    failed = _passing_report()
    failed["concurrency_canary"]["swap_growth_bytes"] = 300 * 1024**2
    report.write_text(json.dumps(failed))
    with pytest.raises(ValueError, match="swap growth"):
        validate_config(MatrixConfig(max_parallel=2, resource_report=report))


def test_timing_matrix_cannot_run_concurrently(tmp_path):
    report = tmp_path / "resources.json"
    report.write_text(json.dumps(_passing_report()))
    with pytest.raises(ValueError, match="exclusively"):
        validate_config(MatrixConfig(max_parallel=2, resource_report=report, timing_run=True))


def test_commands_are_counterbalanced_across_seed_order():
    commands = build_commands(MatrixConfig(seeds=(1, 2, 3)))
    mode_index = commands[0].index("--retrieval-mode") + 1
    assert [command[mode_index] for command in commands[:3]] == ["none", "random", "learned"]
    assert [command[mode_index] for command in commands[3:6]] == ["random", "learned", "none"]
    assert [command[mode_index] for command in commands[6:9]] == ["learned", "none", "random"]


def test_process_orchestration_never_exceeds_bound(monkeypatch):
    active = 0
    maximum = 0

    class FakeProcess:
        def __init__(self, command):
            nonlocal active, maximum
            self.command = command
            active += 1
            maximum = max(maximum, active)

        def wait(self):
            nonlocal active
            active -= 1
            return 0

        def terminate(self):
            return None

    monkeypatch.setattr("memrl.matrix.subprocess.Popen", FakeProcess)
    run_commands([["job", str(index)] for index in range(5)], max_parallel=2)
    assert maximum == 2
