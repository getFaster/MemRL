from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    ):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        self.metrics_file = (run_dir / "metrics.jsonl").open("a", buffering=1)
        self.wandb = None
        if mode != "disabled":
            import wandb

            self.wandb = wandb
            wandb.init(
                project=project,
                entity=entity,
                group=group,
                name=run_name,
                config=config,
                mode=mode,
                save_code=True,
                dir=str(run_dir),
            )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        serializable = {key: float(value) if hasattr(value, "item") else value for key, value in metrics.items()}
        self.metrics_file.write(json.dumps({"global_step": step, **serializable}, allow_nan=False) + "\n")
        if self.wandb is not None:
            self.wandb.log(serializable, step=step)

    def log_retrieval_table(self, rows: list[list[Any]], columns: list[str], step: int) -> None:
        if self.wandb is not None and rows:
            table = self.wandb.Table(columns=columns, data=rows)
            self.wandb.log({"retrieval/top_memories": table}, step=step)

    def close(self) -> None:
        self.metrics_file.close()
        if self.wandb is not None:
            self.wandb.finish()
