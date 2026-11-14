"""
Find the strongest model in a training run by playing head-to-head matches.

The run's best model (`model_<name>.pt`, saved by 100-episode mean return) starts as the champion.
Every periodic checkpoint in `checkpoints/` then challenges it, king-of-the-hill style: each pairing
is played once from each side (the challenger as Blue, then as Orange) and goal difference over both
games decides it. A challenger that wins on goal difference becomes the new champion; a tie keeps
the current one. Prints and returns the path of the final champion.

Run from the repo root:

    python test_checkpoints.py runs/20261114_1537_ppo_s3_self_r4
    python test_checkpoints.py runs/20261114_1537_ppo_s3_self_r4 --seconds 300 --random_spawn --seed 0
"""

import argparse
from pathlib import Path

from src.eval.head_to_head import play_head_to_head


def find_models(run_dir: Path) -> tuple[Path | None, list[Path]]:
    """The run's best-model file (or None) and its periodic checkpoints, oldest first."""
    best = sorted(run_dir.glob("model_*.pt"))
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
    return (best[0] if best else None), checkpoints


def goal_difference(model_a: Path, model_b: Path, **match_kwargs) -> int:
    """Goals for `model_a` minus goals for `model_b`, over one game on each side."""
    first = play_head_to_head(blue_model=model_a, orange_model=model_b, render=False, **match_kwargs)
    second = play_head_to_head(blue_model=model_b, orange_model=model_a, render=False, **match_kwargs)
    return (first["blue"] - first["orange"]) + (second["orange"] - second["blue"])


def best_model_in_run(
    run_dir: str | Path,
    max_seconds: float | None = 300.0,
    max_goals: int | None = None,
    random_spawn: bool = False,
    seed: int | None = None,
) -> Path:
    """Path of the strongest model in `run_dir`: the best-model file or one of its checkpoints."""
    run_dir = Path(run_dir)
    best, checkpoints = find_models(run_dir)
    candidates = ([best] if best else []) + checkpoints
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt or checkpoints/*.pt in '{run_dir}'")

    match_kwargs = dict(max_seconds=max_seconds, max_goals=max_goals, random_spawn=random_spawn, seed=seed)
    champion, *challengers = candidates
    results = []
    for challenger in challengers:
        print(f"\n[Checkpoints] {challenger.name}  vs  champion {champion.name}")
        diff = goal_difference(challenger, champion, **match_kwargs)
        results.append((challenger.name, champion.name, diff))
        if diff > 0:
            print(f"[Checkpoints] {challenger.name} wins ({diff:+d}) and is the new champion")
            champion = challenger
        else:
            print(f"[Checkpoints] {champion.name} holds ({-diff:+d})")

    print("\n[Checkpoints] Summary (goal difference for the challenger):")
    for challenger_name, champion_name, diff in results:
        print(f"  {challenger_name:<45} vs {champion_name:<45} {diff:+d}")
    print(f"\n[Checkpoints] Best model: {champion}")
    return champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick the strongest model in a training run.")
    parser.add_argument("run_dir", type=Path, help="Training run folder, e.g. runs/20261114_1537_ppo_s3_self_r4")
    parser.add_argument("--seconds", type=float, default=300.0, help="Sim seconds per game (default 300)")
    parser.add_argument("--goals", type=int, default=None, help="Stop a game after this many total goals")
    parser.add_argument("--random_spawn", action="store_true", help="Randomise every kickoff")
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable random spawns")
    args = parser.parse_args()

    best_model_in_run(
        args.run_dir,
        max_seconds=args.seconds,
        max_goals=args.goals,
        random_spawn=args.random_spawn,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
