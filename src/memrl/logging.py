from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

MetricFidelity = Literal["exact_rollout_reduction", "periodic_probe", "buffered_summary", "run_scalar"]


@dataclass(frozen=True)
class MetricMetadata:
    fidelity: MetricFidelity
    cadence: str
    description: str = ""


@dataclass(frozen=True)
class EpisodeRecord:
    mode: Literal["none", "random", "learned"]
    seed: int
    env_slot: int
    completion_step: int
    raw_return: float
    length: int

    def validate(self) -> None:
        if self.mode not in {"none", "random", "learned"}:
            raise ValueError(f"invalid retrieval mode: {self.mode}")
        if self.seed < 0 or self.env_slot < 0 or self.completion_step < 0 or self.length < 0:
            raise ValueError("episode integer fields must be non-negative")
        if not math.isfinite(self.raw_return):
            raise ValueError("episode raw_return must be finite")


DEFAULT_METRIC_METADATA: dict[str, MetricMetadata] = {
    "retrieval/entropy": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "retrieval/max_weight": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "retrieval/effective_num_memories": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "retrieval/mean_temporal_distance": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "retrieval/max_similarity": MetricMetadata("exact_rollout_reduction", "every learned retrieval rollout"),
    "retrieval/min_similarity": MetricMetadata("exact_rollout_reduction", "every learned retrieval rollout"),
    "retrieval/mean_similarity": MetricMetadata("exact_rollout_reduction", "every learned retrieval rollout"),
    "retrieval/std_similarity": MetricMetadata("exact_rollout_reduction", "every learned retrieval rollout"),
    "retrieval/same_episode_fraction": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "retrieval/previous_episode_fraction": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "retrieval/mean_age": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "retrieval/recent_under_500_fraction": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "representations/observation_embedding_norm": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "representations/retrieved_memory_norm": MetricMetadata("exact_rollout_reduction", "every retrieval rollout"),
    "representations/memory_to_observation_norm_ratio": MetricMetadata(
        "exact_rollout_reduction", "every retrieval rollout"
    ),
    "retrieval/age_histogram": MetricMetadata("buffered_summary", "every retrieval rollout"),
    "diagnostics/frame_coverage": MetricMetadata("buffered_summary", "every diagnostics emission"),
    "diagnostics/random_query_norm": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "diagnostics/random_similarity_mean": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "diagnostics/random_similarity_std": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "interventions/policy_kl_memory_vs_zero": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "interventions/value_abs_difference_memory_vs_zero": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "interventions/policy_kl_memory_vs_random": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "interventions/policy_kl_memory_vs_shuffled": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "memory/random_pair_cosine_mean": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "memory/random_pair_cosine_std": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "memory/dimension_variance_mean": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "memory/pairwise_embedding_distance": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "memory/near_duplicate_fraction": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "memory/embedding_norm_before_normalization": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "drift/stored_current_cosine_mean": MetricMetadata("periodic_probe", "every diagnostics_interval"),
    "drift/stored_current_cosine_min": MetricMetadata("periodic_probe", "every diagnostics_interval"),
}


class RunLogger:
    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        run_name: str,
        mode: str,
        project: str,
        entity: str | None,
        group: str | None,
        metric_metadata: dict[str, MetricMetadata] | None = None,
        wandb_id: str | None = None,
    ):
        self.run_dir = run_dir
        self.config = config
        self.retrieval_mode = str(config.get("retrieval_mode", "none"))
        self.seed = int(config.get("seed", 0))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        self.metrics_file = (run_dir / "metrics.jsonl").open("a", buffering=1)
        self.episodes_file = (run_dir / "episodes.jsonl").open("a", buffering=1)
        self.metric_metadata = dict(DEFAULT_METRIC_METADATA)
        if metric_metadata:
            self.metric_metadata.update(metric_metadata)
        self._write_metric_metadata()
        self.wandb = None
        self.wandb_run = None
        if mode != "disabled":
            import wandb

            self.wandb = wandb
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                group=group,
                name=run_name,
                config=config,
                mode=mode,
                save_code=True,
                dir=str(run_dir),
                id=wandb_id,
                resume="allow" if wandb_id is not None else None,
            )

    def _write_metric_metadata(self) -> None:
        payload = {
            "schema_version": 1,
            "metrics": {key: asdict(value) for key, value in sorted(self.metric_metadata.items())},
        }
        (self.run_dir / "metric_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def register_metric_metadata(self, name: str, metadata: MetricMetadata) -> None:
        self.metric_metadata[name] = metadata
        self._write_metric_metadata()

    def log(self, metrics: dict[str, Any], step: int) -> None:
        serializable = {key: float(value) if hasattr(value, "item") else value for key, value in metrics.items()}
        missing = [key for key in serializable if key not in self.metric_metadata]
        if missing:
            for key in missing:
                self.metric_metadata[key] = MetricMetadata("run_scalar", "every logged rollout")
            self._write_metric_metadata()
        self.metrics_file.write(json.dumps({"global_step": step, **serializable}, allow_nan=False) + "\n")
        if self.wandb is not None:
            self.wandb.log(serializable, step=step)

    def log_episode(
        self,
        *,
        env_slot: int,
        completion_step: int,
        raw_return: float,
        length: int,
        mode: str | None = None,
        seed: int | None = None,
    ) -> EpisodeRecord:
        record = EpisodeRecord(
            mode=mode or self.retrieval_mode,  # type: ignore[arg-type]
            seed=self.seed if seed is None else seed,
            env_slot=env_slot,
            completion_step=completion_step,
            raw_return=float(raw_return),
            length=length,
        )
        record.validate()
        self.episodes_file.write(json.dumps(asdict(record), allow_nan=False) + "\n")
        return record

    def log_retrieval_table(self, rows: list[list[Any]], columns: list[str], step: int) -> None:
        if self.wandb is not None and rows:
            table = self.wandb.Table(columns=columns, data=rows)
            self.wandb.log({"retrieval/top_memories": table}, step=step)

    def close(self) -> None:
        self.metrics_file.close()
        self.episodes_file.close()
        if self.wandb is not None:
            assert self.wandb_run is not None
            self.wandb_run.finish()
