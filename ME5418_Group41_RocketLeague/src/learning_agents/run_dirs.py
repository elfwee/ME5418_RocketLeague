"""Per-run output folders (`runs/YYYYMMDD_hhmm_<model>/`), checkpoint lookup and periodic checkpoints."""

import itertools
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from src.paths import RUNS_DIR

RUN_DIR_TIME_FORMAT = "%Y%m%d_%H%M"


def make_run_dir(model_name: str) -> Path:
    """Create and return this training run's output folder, `runs/YYYYMMDD_hhmm_<model_name>/`.

    A second run of the same model started within the same minute gets a `_2`, `_3`, ... suffix,
    so a run never overwrites or appends to another run's files.
    """
    base = f"{datetime.now().strftime(RUN_DIR_TIME_FORMAT)}_{model_name}"
    for suffix in itertools.chain([""], (f"_{n}" for n in itertools.count(2))):
        run_dir = RUNS_DIR / f"{base}{suffix}"
        try:
            run_dir.mkdir(parents=True)
            return run_dir
        except FileExistsError:
            continue


@dataclass(frozen=True)
class RunPaths:
    """Where one training run writes its outputs (all inside `run_dir`)."""

    run_dir: Path
    model_name: str

    @classmethod
    def new(cls, model_name: str) -> "RunPaths":
        """A fresh `runs/YYYYMMDD_hhmm_<model_name>/` folder for a new training run."""
        return cls(make_run_dir(model_name), model_name)

    @property
    def log_file(self) -> Path:
        return self.run_dir / f"run_{self.model_name}.log"

    @property
    def model_file(self) -> Path:
        """Best model so far (by 100-episode mean return)."""
        return self.run_dir / f"model_{self.model_name}.pt"

    @property
    def graph_file(self) -> Path:
        return self.run_dir / f"graph_{self.model_name}.png"

    @property
    def metrics_file(self) -> Path:
        return self.run_dir / f"metrics_{self.model_name}.csv"

    @property
    def checkpoint_dir(self) -> Path:
        """Periodic snapshots of the current model."""
        return self.run_dir / "checkpoints"


def latest_model_path(model_name: str) -> Path:
    """Default checkpoint for `model_name`: the one in its most recent run folder.

    Falls back to the legacy flat `runs/model_<model_name>.pt` when no run folder holds a
    checkpoint (the returned path may not exist; callers handle a missing file).
    """
    model_file = f"model_{model_name}.pt"
    run_dir_pattern = re.compile(rf"(\d{{8}}_\d{{4}})_{re.escape(model_name)}(?:_(\d+))?")
    candidates = []
    if RUNS_DIR.is_dir():
        for run_dir in RUNS_DIR.iterdir():
            match = run_dir_pattern.fullmatch(run_dir.name)
            if match and (run_dir / model_file).is_file():
                candidates.append((match.group(1), int(match.group(2) or 1), run_dir / model_file))
    if candidates:
        return max(candidates)[2]
    return RUNS_DIR / model_file


def save_periodic_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    model_name: str,
    episode: int,
    interval: int,
) -> Path | None:
    """Save the current weights every `interval` episodes; returns the path, or None when not due.

    `episode` is the 0-based loop index, so the file is named after the number of episodes
    completed: `<checkpoint_dir>/model_<model_name>_ep000500.pt`. Unlike the best-model file,
    these are never overwritten. `interval <= 0` disables periodic checkpoints.
    """
    episodes_done = episode + 1
    if interval <= 0 or episodes_done % interval != 0:
        return None
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"model_{model_name}_ep{episodes_done:06d}.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint after {episodes_done} episodes to: {path}")
    return path
