"""Compute the retrieval performance gate from process-local timing reports."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

MODES = ("none", "random", "learned")
THRESHOLDS = {"random": 2.0, "learned": 3.0}
LOCKED_TOTAL_TIMESTEPS = 200_000
LOCKED_BATCH_SIZE = 1_024
LOCKED_MEMORY_CAPACITY = 100_000


@dataclass
class BenchmarkConfig:
    inputs: tuple[Path, ...] = ()
    output: Path = Path("benchmark-report.json")
    exclude_through_step: int = 100_000
    expected_seeds: tuple[int, ...] = (901, 902, 903)
    run_gate: bool = False
    total_timesteps: int = 200_000
    run_root: Path = Path("runs/benchmark")


def load_timing_records(paths: tuple[Path, ...] | list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            records.extend(payload)
        elif "runs" in payload:
            records.extend(payload["runs"])
        else:
            records.append(payload)
    return records


def _steady_times(record: dict[str, Any], exclude_through_step: int) -> np.ndarray:
    batch_size = int(record.get("batch_size", 0))
    values = [
        item["wall_time_seconds"]
        for item in record.get("iterations", [])
        if int(item.get("iteration_start_step", int(item["global_step"]) - batch_size)) >= exclude_through_step
    ]
    return np.asarray(values, dtype=np.float64)


def _condition_failures(record: dict[str, Any], mode: str) -> list[str]:
    failures = []
    if record.get("benchmark_run") is not True:
        failures.append("not tagged as a benchmark subprocess")
    if int(record.get("requested_total_timesteps", -1)) != LOCKED_TOTAL_TIMESTEPS:
        failures.append("wrong requested step budget")
    if int(record.get("batch_size", -1)) != LOCKED_BATCH_SIZE:
        failures.append("wrong rollout batch size")
    if int(record.get("memory_capacity", -1)) != LOCKED_MEMORY_CAPACITY:
        failures.append("wrong memory capacity")
    minimum_completed = (LOCKED_TOTAL_TIMESTEPS // LOCKED_BATCH_SIZE) * LOCKED_BATCH_SIZE
    if int(record.get("final_global_step", -1)) < minimum_completed:
        failures.append("incomplete step budget")
    if record.get("wandb_mode") != "disabled":
        failures.append("W&B was not disabled")
    if record.get("checkpointing_enabled") is not False:
        failures.append("checkpointing was not disabled")
    if record.get("host_frame_maintenance") is not (mode != "none"):
        failures.append("host-frame maintenance setting is invalid")
    if not record.get("run_id") or not isinstance(record.get("process_id"), int):
        failures.append("missing fresh-process identity")
    return failures


def analyze_timing_records(
    records: list[dict[str, Any]],
    *,
    exclude_through_step: int = 100_000,
    expected_seeds: tuple[int, ...] = (901, 902, 903),
) -> dict[str, Any]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    run_results = []
    all_runs_healthy = True
    for record in records:
        mode = str(record["mode"])
        seed = int(record["seed"])
        if mode not in MODES:
            raise ValueError(f"unknown benchmark mode: {mode}")
        key = (mode, seed)
        if key in indexed:
            raise ValueError(f"duplicate benchmark record for mode={mode} seed={seed}")
        indexed[key] = record
        times = _steady_times(record, exclude_through_step)
        condition_failures = _condition_failures(record, mode)
        healthy = record.get("finite") is True and record.get("oom") is False and not condition_failures
        healthy = healthy and bool(times.size) and bool(np.isfinite(times).all()) and bool(np.all(times > 0))
        all_runs_healthy = all_runs_healthy and healthy
        run_results.append(
            {
                "mode": mode,
                "seed": seed,
                "steady_samples": int(times.size),
                "median_iteration_seconds": float(np.median(times)) if times.size else None,
                "finite": bool(record.get("finite", True)) and bool(np.isfinite(times).all()),
                "oom": bool(record.get("oom", False)),
                "condition_failures": condition_failures,
                "cold_compilation_seconds": record.get("cold_compilation_seconds"),
                "peak_device_bytes": record.get("peak_device_bytes"),
                "peak_host_bytes": record.get("peak_host_bytes"),
            }
        )

    gates: dict[str, Any] = {}
    for mode, threshold in THRESHOLDS.items():
        ratios = []
        paired = []
        for seed in expected_seeds:
            baseline = indexed.get(("none", seed))
            candidate = indexed.get((mode, seed))
            if baseline is None or candidate is None:
                continue
            baseline_times = _steady_times(baseline, exclude_through_step)
            candidate_times = _steady_times(candidate, exclude_through_step)
            if not baseline_times.size or not candidate_times.size:
                continue
            ratio = float(np.median(candidate_times) / np.median(baseline_times))
            ratios.append(ratio)
            paired.append({"seed": seed, "ratio": ratio})
        median_ratio = float(np.median(ratios)) if ratios else None
        complete = len(ratios) == len(expected_seeds)
        passed = complete and all_runs_healthy and median_ratio is not None and math.isfinite(median_ratio)
        passed = bool(passed and median_ratio <= threshold)
        gates[f"{mode}/none"] = {
            "threshold": threshold,
            "paired_ratios": paired,
            "median_ratio": median_ratio,
            "complete": complete,
            "passed": passed,
        }

    numeric_compilation = [
        float(row["cold_compilation_seconds"]) for row in run_results if row["cold_compilation_seconds"] is not None
    ]
    numeric_device = [int(row["peak_device_bytes"]) for row in run_results if row["peak_device_bytes"] is not None]
    numeric_host = [int(row["peak_host_bytes"]) for row in run_results if row["peak_host_bytes"] is not None]
    return {
        "schema_version": 1,
        "exclude_through_step": exclude_through_step,
        "expected_seeds": list(expected_seeds),
        "runs": sorted(run_results, key=lambda row: (row["seed"], MODES.index(row["mode"]))),
        "gates": gates,
        "all_runs_finite_and_no_oom": all_runs_healthy,
        "overall_passed": all(gate["passed"] for gate in gates.values()),
        "resource_summary_non_gate": {
            "cold_compilation_seconds": numeric_compilation,
            "peak_device_bytes": max(numeric_device) if numeric_device else None,
            "peak_host_bytes": max(numeric_host) if numeric_host else None,
        },
    }


def launch_gate(config: BenchmarkConfig) -> tuple[Path, ...]:
    """Run the nine counterbalanced benchmark processes exclusively."""

    session_root = config.run_root / f"gate-{int(time.time())}"
    timing_files: list[Path] = []
    modes = ("none", "random", "learned")
    for seed_index, seed in enumerate(config.expected_seeds):
        for offset in range(len(modes)):
            mode = modes[(seed_index + offset) % len(modes)]
            output_dir = session_root / f"seed{seed}" / mode
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "memrl.train",
                    "--retrieval-mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--total-timesteps",
                    str(config.total_timesteps),
                    "--wandb-mode",
                    "disabled",
                    "--no-checkpointing",
                    "--benchmark-run",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
            )
            matches = sorted(output_dir.glob("*/timing.json"), key=lambda path: path.stat().st_mtime)
            if not matches:
                raise RuntimeError(f"benchmark process did not emit timing.json for mode={mode} seed={seed}")
            timing_files.append(matches[-1])
    return tuple(timing_files)


def main() -> None:
    config = tyro.cli(BenchmarkConfig)
    inputs = launch_gate(config) if config.run_gate else config.inputs
    if not inputs:
        raise ValueError("provide --inputs or use --run-gate")
    result = analyze_timing_records(
        load_timing_records(inputs),
        exclude_through_step=config.exclude_through_step,
        expected_seeds=config.expected_seeds,
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
