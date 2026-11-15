"""
Find the best model in a training run, king-of-the-hill style, and print its path.

The run's `model_<name>.pt` starts as champion and every checkpoint in `checkpoints/` challenges it
in training order (one game per side, goal difference decides; see `test_checkpoints.py`).
Match logs go to stderr, so stdout is just the winning model's path.

Run from the repo root, with a run folder name or path:

    python king_of_the_hill.py 20261114_0953_ppo_s2_heuristic
    python king_of_the_hill.py runs/20261114_0953_ppo_s2_heuristic --seconds 300
    BEST=$(python king_of_the_hill.py 20261114_0953_ppo_s2_heuristic 2>/dev/null)
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # keep pygame's banner off stdout

from test_checkpoints import best_model_in_run  # noqa: E402

RUNS_DIR = Path("runs")


def resolve_run_dir(run: str | Path) -> Path:
    """Accept either a path to a run folder or just its name inside `runs/`."""
    run = Path(run)
    if run.is_dir():
        return run
    if (RUNS_DIR / run).is_dir():
        return RUNS_DIR / run
    raise FileNotFoundError(f"No run folder '{run}' (also looked in '{RUNS_DIR / run}')")


def king_of_the_hill(run: str | Path, **match_kwargs) -> Path:
    """Path of the strongest model in the run. Match output is sent to stderr."""
    run_dir = resolve_run_dir(run)
    with contextlib.redirect_stdout(sys.stderr):
        return best_model_in_run(run_dir, **match_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the path of the best model in a training run.")
    parser.add_argument("run", help="Run folder name in runs/ (e.g. 20261114_0953_ppo_s2_heuristic) or a path")
    parser.add_argument("--seconds", type=float, default=300.0, help="Sim seconds per game (default 300)")
    parser.add_argument("--goals", type=int, default=None, help="Stop a game after this many total goals")
    parser.add_argument("--random_spawn", action="store_true", help="Randomise every kickoff")
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable random spawns")
    args = parser.parse_args()

    best = king_of_the_hill(
        args.run,
        max_seconds=args.seconds,
        max_goals=args.goals,
        random_spawn=args.random_spawn,
        seed=args.seed,
    )
    print(best)


if __name__ == "__main__":
    main()
