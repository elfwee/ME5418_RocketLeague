"""
Find the strongest model in a training run by playing head-to-head matches.

Qualifying: every model first plays the heuristic bot (once from each side); only models with a
positive goal difference against it go forward, so the final champion is guaranteed to beat the
heuristic bot. If no model beats it, a RuntimeError is raised.

Among the qualifiers, the run's best model (`model_<name>.pt`, saved by 100-episode mean return)
starts as the champion. Every qualifying checkpoint in `checkpoints/` then challenges it, king-of-the-hill
style: each pairing is played once from each side (the challenger as Blue, then as Orange) and goal
difference over both games decides it. A challenger whose goal difference exceeds the threshold becomes
the new champion; otherwise the current one holds. Prints and returns the path of the final champion.

Run from the repo root:

    python -m src --best_checkpoint runs/20261114_1537_ppo_s3_self_r4
    python -m src --best_checkpoint runs/20261114_1537_ppo_s3_self_r4 --max_seconds 300 --random_spawn --seed 0
"""

from pathlib import Path

from src.eval.head_to_head import play_head_to_head


def find_models(run_dir: Path) -> tuple[Path | None, list[Path]]:
    """The run's best-model file (or None) and its periodic checkpoints, oldest first."""
    best = sorted(run_dir.glob("model_*.pt"))
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
    return (best[0] if best else None), checkpoints


def goal_difference(model_a: Path | None, model_b: Path | None, **match_kwargs) -> int:
    """Goals for `model_a` minus goals for `model_b`, over one game on each side (None = heuristic bot)."""
    first = play_head_to_head(blue_model=model_a, orange_model=model_b, render=False, **match_kwargs)
    second = play_head_to_head(blue_model=model_b, orange_model=model_a, render=False, **match_kwargs)
    return (first["blue"] - first["orange"]) + (second["orange"] - second["blue"])


def best_model_in_run(
    run_dir: str | Path,
    goal_difference_threshold: int = 3,
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

    print("\n[Checkpoints] Qualifying: every model vs the heuristic bot")
    vs_heuristic = {}
    for model in candidates:
        vs_heuristic[model] = goal_difference(model, None, **match_kwargs)
        verdict = "qualifies" if vs_heuristic[model] > 0 else "eliminated"
        print(f"[Checkpoints] {model.name:<45} vs heuristic bot {vs_heuristic[model]:+d}  {verdict}")
    qualified = [model for model in candidates if vs_heuristic[model] > 0]
    if not qualified:
        closest = max(candidates, key=vs_heuristic.get)
        raise RuntimeError(
            f"No model in '{run_dir}' beats the heuristic bot "
            f"(closest: {closest.name}, {vs_heuristic[closest]:+d})"
        )
    print(f"[Checkpoints] {len(qualified)}/{len(candidates)} models beat the heuristic bot")

    champion, *challengers = qualified
    results = []
    for challenger in challengers:
        print(f"\n[Checkpoints] {challenger.name}  vs  champion {champion.name}")
        diff = goal_difference(challenger, champion, **match_kwargs)
        results.append((challenger.name, champion.name, diff))
        if diff > goal_difference_threshold:
            print(f"[Checkpoints] {challenger.name} wins ({diff:+d}) and is the new champion")
            champion = challenger
        else:
            print(f"[Checkpoints] {champion.name} holds ({-diff:+d})")

    print("\n[Checkpoints] Summary (goal difference for the challenger):")
    for challenger_name, champion_name, diff in results:
        print(f"  {challenger_name:<45} vs {champion_name:<45} {diff:+d}")
    print(f"\n[Checkpoints] Best model: {champion} (vs heuristic bot {vs_heuristic[champion]:+d})")
    return champion
