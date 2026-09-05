from __future__ import annotations

from memrl.benchmark import analyze_timing_records


def _records():
    records = []
    multipliers = {"none": 1.0, "random": 1.8, "learned": 2.5}
    for seed in (901, 902, 903):
        for mode, multiplier in multipliers.items():
            records.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "run_id": f"{mode}-{seed}",
                    "process_id": seed * 10 + len(mode),
                    "benchmark_run": True,
                    "requested_total_timesteps": 200_000,
                    "final_global_step": 199_680,
                    "batch_size": 1_024,
                    "memory_capacity": 100_000,
                    "iterations": [
                        {"iteration_start_step": 99_328, "global_step": 100_352, "wall_time_seconds": 99.0},
                        {"iteration_start_step": 100_352, "global_step": 101_376, "wall_time_seconds": multiplier},
                        {"iteration_start_step": 101_376, "global_step": 102_400, "wall_time_seconds": multiplier},
                    ],
                    "finite": True,
                    "oom": False,
                    "checkpointing_enabled": False,
                    "wandb_mode": "disabled",
                    "host_frame_maintenance": mode != "none",
                    "cold_compilation_seconds": 12.0,
                    "peak_device_bytes": 100,
                    "peak_host_bytes": 200,
                }
            )
    return records


def test_benchmark_excludes_warmup_and_applies_paired_median_gates():
    result = analyze_timing_records(_records())
    assert result["overall_passed"] is True
    assert result["gates"]["random/none"]["median_ratio"] == 1.8
    assert result["gates"]["learned/none"]["median_ratio"] == 2.5
    assert len(result["gates"]["random/none"]["paired_ratios"]) == 3
    assert all(run["steady_samples"] == 2 for run in result["runs"])
    assert result["resource_summary_non_gate"]["peak_device_bytes"] == 100


def test_benchmark_fails_gate_on_oom_or_missing_replicate():
    records = _records()
    records[0]["oom"] = True
    result = analyze_timing_records(records)
    assert result["all_runs_finite_and_no_oom"] is False
    assert result["overall_passed"] is False

    missing = [record for record in _records() if not (record["mode"] == "learned" and record["seed"] == 903)]
    result = analyze_timing_records(missing)
    assert result["gates"]["learned/none"]["complete"] is False
    assert result["overall_passed"] is False


def test_benchmark_rejects_unverified_run_conditions():
    records = _records()
    records[0].pop("benchmark_run")
    records[1]["checkpointing_enabled"] = True
    records[2]["wandb_mode"] = "online"
    result = analyze_timing_records(records)
    assert result["all_runs_finite_and_no_oom"] is False
    assert result["overall_passed"] is False
    assert any(run["condition_failures"] for run in result["runs"])
