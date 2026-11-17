"""Headless round-robin evaluation across a folder of checkpoints and HeuristicBot.

Runs every directed (Blue, Orange) pairing in parallel across CPU workers using the headless
continuous simulation engine (`Simulation` + `PolicyController` / `HeuristicController`),
logs per-match outcomes, and exports summary tables and head-to-head matrices.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.paths import ROOT_DIR

DEFAULT_MODELS_DIR = ROOT_DIR / "eval_model"
DEFAULT_OUT_DIR = DEFAULT_MODELS_DIR / "eval_report"


@dataclass(frozen=True)
class BotSpec:
    """Participant in the round-robin tournament (`model_path=None` selects HeuristicBot)."""

    bot_id: str
    model_path: str | None


def discover_bots(models_dir: Path, include_heuristic: bool = True) -> list[BotSpec]:
    """Discover all ``.pt`` checkpoints in ``models_dir``, optionally prepending ``heuristic_bot``."""
    bots: list[BotSpec] = []
    if include_heuristic:
        bots.append(BotSpec(bot_id="heuristic_bot", model_path=None))
    if models_dir.is_dir():
        for pt_file in sorted(models_dir.glob("*.pt")):
            bots.append(BotSpec(bot_id=pt_file.stem, model_path=str(pt_file)))
    return bots


def _run_pairing(task: tuple[int, BotSpec, BotSpec, int, int | None, float, int | None, bool]) -> dict:
    """Worker entrypoint: play ``num_matches`` headless matches for one directed (Blue, Orange) pair."""
    pairing_idx, blue_spec, orange_spec, num_matches, base_seed, max_seconds, max_goals, random_spawn = task

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import torch

    torch.set_num_threads(1)

    from src.config.sim_config import SIM_HZ
    from src.sim.control import HeuristicController
    from src.sim.control.policy import PolicyController
    from src.sim.core.simulation import Simulation

    sim = Simulation(enable_orange=True, random_spawn=False, seed=None)
    blue_ctrl = (
        HeuristicController("blue", sim)
        if blue_spec.model_path is None
        else PolicyController("blue", sim, blue_spec.model_path)
    )
    orange_ctrl = (
        HeuristicController("orange", sim)
        if orange_spec.model_path is None
        else PolicyController("orange", sim, orange_spec.model_path)
    )

    dt = 1.0 / SIM_HZ
    max_steps = int(round(max_seconds * SIM_HZ)) if max_seconds is not None else None
    wall_start = time.perf_counter()

    match_rows = []
    blue_wins = 0
    orange_wins = 0
    draws = 0
    blue_goals_total = 0
    orange_goals_total = 0
    total_sim_seconds = 0.0

    for match_idx in range(num_matches):
        match_seed = (base_seed + match_idx) if base_seed is not None else None
        if match_seed is not None:
            sim.rng.seed(match_seed)
        sim.reset(reset_scores=True, spawn_index=match_idx, random_spawn=random_spawn, clear_events=True)
        blue_ctrl.reset()
        orange_ctrl.reset()

        steps = 0
        while max_steps is None or steps < max_steps:
            steps += 1
            sim.step(action=blue_ctrl.get_action(), dt=dt, action_orange=orange_ctrl.get_action())
            if sim.goal_scored_this_step is not None:
                blue_ctrl.reset()
                orange_ctrl.reset()
            if max_goals is not None and (sim.score_blue + sim.score_orange) >= max_goals:
                break

        sim_seconds = round(steps * dt, 4)
        total_sim_seconds += sim_seconds
        b_goals = int(sim.score_blue)
        o_goals = int(sim.score_orange)
        blue_goals_total += b_goals
        orange_goals_total += o_goals

        if b_goals > o_goals:
            winner = "blue"
            blue_wins += 1
        elif o_goals > b_goals:
            winner = "orange"
            orange_wins += 1
        else:
            winner = "draw"
            draws += 1

        match_rows.append(
            {
                "pairing_id": pairing_idx,
                "blue_bot": blue_spec.bot_id,
                "orange_bot": orange_spec.bot_id,
                "match_index": match_idx,
                "seed": "" if match_seed is None else match_seed,
                "winner": winner,
                "blue_goals": b_goals,
                "orange_goals": o_goals,
                "sim_seconds": sim_seconds,
                "steps": steps,
                "last_touch": sim.last_touch or "",
            }
        )

    wall_seconds = round(time.perf_counter() - wall_start, 2)
    summary = {
        "pairing_id": pairing_idx,
        "blue_bot": blue_spec.bot_id,
        "orange_bot": orange_spec.bot_id,
        "matches": num_matches,
        "blue_wins": blue_wins,
        "orange_wins": orange_wins,
        "draws": draws,
        "blue_goals": blue_goals_total,
        "orange_goals": orange_goals_total,
        "goal_diff_blue": blue_goals_total - orange_goals_total,
        "blue_win_rate_pct": round(100.0 * blue_wins / num_matches, 2),
        "orange_win_rate_pct": round(100.0 * orange_wins / num_matches, 2),
        "draw_rate_pct": round(100.0 * draws / num_matches, 2),
        "avg_match_seconds": round(total_sim_seconds / num_matches, 2),
        "wall_seconds": wall_seconds,
    }
    return {"summary": summary, "matches": match_rows}


def _write_outputs(
    out_dir: Path,
    bots: list[BotSpec],
    results: list[dict],
    num_matches: int,
    base_seed: int | None,
    max_seconds: float,
    max_goals: int | None,
    random_spawn: bool,
    include_mirrors: bool,
    total_wall_seconds: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["summary"]["pairing_id"])
    bot_ids = [b.bot_id for b in bots]
    by_pair = {(r["summary"]["blue_bot"], r["summary"]["orange_bot"]): r["summary"] for r in results}

    # 1. Raw matches CSV
    raw_csv_path = out_dir / "raw_matches.csv"
    match_fields = [
        "pairing_id",
        "blue_bot",
        "orange_bot",
        "match_index",
        "seed",
        "winner",
        "blue_goals",
        "orange_goals",
        "sim_seconds",
        "steps",
        "last_touch",
    ]
    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=match_fields)
        writer.writeheader()
        for r in results:
            writer.writerows(r["matches"])

    # 2. Raw text log
    raw_log_path = out_dir / "raw_matches.log"
    with raw_log_path.open("w", encoding="utf-8") as f:
        f.write(
            f"Headless Round-Robin Evaluation | bots={len(bots)} | pairings={len(results)} | "
            f"matches_per_pairing={num_matches} | base_seed={base_seed} | max_seconds={max_seconds} | "
            f"max_goals={max_goals} | random_spawn={random_spawn} | total_wall_seconds={total_wall_seconds:.2f}\n\n"
        )
        for r in results:
            s = r["summary"]
            f.write(
                f"=== Pairing {s['pairing_id']:02d}: Blue={s['blue_bot']} vs Orange={s['orange_bot']} | "
                f"Score(B-O-D)={s['blue_wins']}-{s['orange_wins']}-{s['draws']} | "
                f"Goals={s['blue_goals']}-{s['orange_goals']} | "
                f"AvgSim={s['avg_match_seconds']:.2f}s | Wall={s['wall_seconds']:.2f}s ===\n"
            )
            for m in r["matches"]:
                f.write(
                    f"  match={m['match_index']:03d} seed={m['seed'] if m['seed'] != '' else 'none'} "
                    f"winner={m['winner']:<6} score={m['blue_goals']}-{m['orange_goals']} "
                    f"sim_s={m['sim_seconds']:6.2f} steps={m['steps']:4d} last_touch={m['last_touch'] or 'none'}\n"
                )
            f.write("\n")

    # 3. Pairings summary CSV
    pairings_csv_path = out_dir / "pairings_summary.csv"
    summary_fields = list(results[0]["summary"].keys())
    with pairings_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r["summary"])

    # 4. Matrix CSVs (rows = Blue bot, cols = Orange bot)
    def cell_or_dash(b_id: str, o_id: str, formatter) -> str:
        s = by_pair.get((b_id, o_id))
        return "-" if s is None else formatter(s)

    def write_matrix(filename: str, formatter):
        path = out_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["blue_vs_orange", *bot_ids])
            for b_id in bot_ids:
                writer.writerow([b_id, *[cell_or_dash(b_id, o_id, formatter) for o_id in bot_ids]])

    write_matrix("matrix_score_b_o_d.csv", lambda s: f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']}")
    write_matrix("matrix_blue_win_rate_pct.csv", lambda s: f"{s['blue_win_rate_pct']:.1f}")
    write_matrix("matrix_blue_goal_diff.csv", lambda s: f"{s['goal_diff_blue']}")

    # 5. Overall standings CSV
    standings_rows = []
    for bot_id in bot_ids:
        b_pairs = [by_pair[(bot_id, o)] for o in bot_ids if (bot_id, o) in by_pair]
        o_pairs = [by_pair[(b, bot_id)] for b in bot_ids if (b, bot_id) in by_pair]

        b_wins = sum(s["blue_wins"] for s in b_pairs)
        b_losses = sum(s["orange_wins"] for s in b_pairs)
        b_draws = sum(s["draws"] for s in b_pairs)
        b_gf = sum(s["blue_goals"] for s in b_pairs)
        b_ga = sum(s["orange_goals"] for s in b_pairs)

        o_wins = sum(s["orange_wins"] for s in o_pairs)
        o_losses = sum(s["blue_wins"] for s in o_pairs)
        o_draws = sum(s["draws"] for s in o_pairs)
        o_gf = sum(s["orange_goals"] for s in o_pairs)
        o_ga = sum(s["blue_goals"] for s in o_pairs)

        tot_matches = (b_wins + b_losses + b_draws) + (o_wins + o_losses + o_draws)
        tot_wins = b_wins + o_wins
        tot_losses = b_losses + o_losses
        tot_draws = b_draws + o_draws

        nm_b = [by_pair[(bot_id, o)] for o in bot_ids if o != bot_id and (bot_id, o) in by_pair]
        nm_o = [by_pair[(b, bot_id)] for b in bot_ids if b != bot_id and (b, bot_id) in by_pair]
        nm_wins = sum(s["blue_wins"] for s in nm_b) + sum(s["orange_wins"] for s in nm_o)
        nm_losses = sum(s["orange_wins"] for s in nm_b) + sum(s["blue_wins"] for s in nm_o)
        nm_draws = sum(s["draws"] for s in nm_b) + sum(s["draws"] for s in nm_o)
        nm_gf = sum(s["blue_goals"] for s in nm_b) + sum(s["orange_goals"] for s in nm_o)
        nm_ga = sum(s["orange_goals"] for s in nm_b) + sum(s["blue_goals"] for s in nm_o)
        nm_matches = nm_wins + nm_losses + nm_draws

        standings_rows.append(
            {
                "bot": bot_id,
                "matches_all": tot_matches,
                "wins_all": tot_wins,
                "losses_all": tot_losses,
                "draws_all": tot_draws,
                "win_rate_all_pct": round(100.0 * tot_wins / tot_matches, 2) if tot_matches else 0.0,
                "goal_diff_all": (b_gf + o_gf) - (b_ga + o_ga),
                "matches_non_mirror": nm_matches,
                "wins_non_mirror": nm_wins,
                "losses_non_mirror": nm_losses,
                "draws_non_mirror": nm_draws,
                "win_rate_non_mirror_pct": round(100.0 * nm_wins / nm_matches, 2) if nm_matches else 0.0,
                "goal_diff_non_mirror": nm_gf - nm_ga,
                "blue_wins": b_wins,
                "blue_losses": b_losses,
                "blue_draws": b_draws,
                "orange_wins": o_wins,
                "orange_losses": o_losses,
                "orange_draws": o_draws,
            }
        )

    standings_rows.sort(key=lambda r: (r["win_rate_non_mirror_pct"], r["goal_diff_non_mirror"]), reverse=True)
    standings_csv_path = out_dir / "overall_standings.csv"
    with standings_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(standings_rows[0].keys()))
        writer.writeheader()
        writer.writerows(standings_rows)

    # 6. Combined Symmetric Head-to-Head Matrix
    sym_csv_path = out_dir / "matrix_combined_200_matches.csv"
    with sym_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_vs_col", *bot_ids])
        for row_bot in bot_ids:
            row_cells = []
            for col_bot in bot_ids:
                s_bo = by_pair.get((row_bot, col_bot))
                s_ob = by_pair.get((col_bot, row_bot))
                if s_bo is None or s_ob is None:
                    row_cells.append("-")
                else:
                    w = s_bo["blue_wins"] + s_ob["orange_wins"]
                    l = s_bo["orange_wins"] + s_ob["blue_wins"]
                    d = s_bo["draws"] + s_ob["draws"]
                    row_cells.append(f"{w}-{l}-{d}")
            writer.writerow([row_bot, *row_cells])

    # 7. Markdown summary report
    nm_cap = standings_rows[0]["matches_non_mirror"] if standings_rows else 0
    side_cap = num_matches * (len(bots) if include_mirrors else max(0, len(bots) - 1))
    n_models = sum(1 for b in bots if b.model_path is not None)
    has_heur = any(b.model_path is None for b in bots)
    bot_desc = f"{n_models} models + `heuristic_bot`" if has_heur else f"{n_models} models"
    seed_desc = f"{base_seed} (match `i` uses `seed = {base_seed} + i`)" if base_seed is not None else "None (unseeded)"

    report_md_path = out_dir / "eval_report.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("# Headless Round-Robin Evaluation Report\n\n")
        f.write(
            f"- **Bots:** {len(bots)} ({bot_desc})\n"
            f"- **Pairings:** {len(results)} (swapped sides, mirrors={'included' if include_mirrors else 'excluded'})\n"
            f"- **Matches per pairing:** {num_matches} (max_goals={max_goals}, max_seconds={max_seconds:.0f}s, `0-0` timeout = Draw)\n"
            f"- **Total matches:** {len(results) * num_matches}\n"
            f"- **Random spawn:** {random_spawn} | **Base seed:** {seed_desc}\n"
            f"- **Wall time:** {total_wall_seconds:.2f}s\n\n"
        )
        f.write("## Overall Standings (Sorted by Non-Mirror Win Rate)\n\n")
        f.write(
            f"| Rank | Bot | W-L-D (Non-Mirror, /{nm_cap}) | Win% (Non-Mirror) | GD (Non-Mirror) | "
            f"Blue W-L-D (/{side_cap}) | Orange W-L-D (/{side_cap}) |\n"
        )
        f.write("| ---: | :--- | :---: | ---: | ---: | :---: | :---: |\n")
        for idx, row in enumerate(standings_rows, start=1):
            f.write(
                f"| {idx} | `{row['bot']}` | "
                f"{row['wins_non_mirror']}-{row['losses_non_mirror']}-{row['draws_non_mirror']} | "
                f"{row['win_rate_non_mirror_pct']:.2f}% | {row['goal_diff_non_mirror']:+d} | "
                f"{row['blue_wins']}-{row['blue_losses']}-{row['blue_draws']} | "
                f"{row['orange_wins']}-{row['orange_losses']}-{row['orange_draws']} |\n"
            )

        f.write(
            f"\n## Directed Head-to-Head Matrix (Row = Blue, Col = Orange; `BlueWins-OrangeWins-Draws` out of {num_matches})\n\n"
        )
        f.write("| Blue \\ Orange | " + " | ".join(f"`{b}`" for b in bot_ids) + " |\n")
        f.write("| :--- | " + " | ".join([":---:"] * len(bot_ids)) + " |\n")
        for b_id in bot_ids:
            cells = [cell_or_dash(b_id, o_id, lambda s: f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']}") for o_id in bot_ids]
            f.write(f"| `{b_id}` | " + " | ".join(cells) + " |\n")

        f.write(
            f"\n## Combined Swapped-Side Matrix (Row vs Col; `RowWins-ColWins-Draws` out of {2 * num_matches})\n\n"
        )
        f.write("| Row \\ Col | " + " | ".join(f"`{b}`" for b in bot_ids) + " |\n")
        f.write("| :--- | " + " | ".join([":---:"] * len(bot_ids)) + " |\n")
        for row_bot in bot_ids:
            cells = []
            for col_bot in bot_ids:
                s_bo = by_pair.get((row_bot, col_bot))
                s_ob = by_pair.get((col_bot, row_bot))
                if s_bo is None or s_ob is None:
                    cells.append("-")
                else:
                    w = s_bo["blue_wins"] + s_ob["orange_wins"]
                    l = s_bo["orange_wins"] + s_ob["blue_wins"]
                    d = s_bo["draws"] + s_ob["draws"]
                    cells.append(f"{w}-{l}-{d}")
            f.write(f"| `{row_bot}` | " + " | ".join(cells) + " |\n")


def run_round_robin_eval(
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    matches: int = 100,
    seed: int | None = 50,
    max_seconds: float = 60.0,
    max_goals: int | None = 1,
    random_spawn: bool = True,
    include_heuristic: bool = True,
    include_mirrors: bool = True,
    workers: int | None = None,
) -> Path:
    """Run a headless side-swapped round-robin tournament and write CSV/log/Markdown reports."""
    models_path = Path(models_dir)
    out_path = Path(out_dir)
    bots = discover_bots(models_path, include_heuristic=include_heuristic)
    if len(bots) < (1 if include_mirrors else 2):
        raise ValueError(f"Not enough bots found in '{models_path}' (found {len(bots)}).")

    worker_count = max(1, workers if workers is not None else (os.cpu_count() or 4) - 1)
    tasks = []
    idx = 0
    for b in bots:
        for o in bots:
            if not include_mirrors and b.bot_id == o.bot_id:
                continue
            tasks.append((idx, b, o, matches, seed, max_seconds, max_goals, random_spawn))
            idx += 1

    print(
        f"[RoundRobin] Discovered {len(bots)} bots -> {len(tasks)} pairings x {matches} matches "
        f"| seed={seed} | max_seconds={max_seconds} | max_goals={max_goals} | "
        f"random_spawn={random_spawn} | workers={worker_count}",
        flush=True,
    )
    t0 = time.perf_counter()
    results = []
    if worker_count == 1:
        for done_count, task in enumerate(tasks, start=1):
            res = _run_pairing(task)
            results.append(res)
            s = res["summary"]
            print(
                f"[{done_count:02d}/{len(tasks):02d}] {s['blue_bot']} (Blue) vs {s['orange_bot']} (Orange): "
                f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']} "
                f"(avg {s['avg_match_seconds']:.1f}s sim, {s['wall_seconds']:.1f}s wall)",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_run_pairing, task) for task in tasks]
            for done_count, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                results.append(res)
                s = res["summary"]
                print(
                    f"[{done_count:02d}/{len(tasks):02d}] {s['blue_bot']} (Blue) vs {s['orange_bot']} (Orange): "
                    f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']} "
                    f"(avg {s['avg_match_seconds']:.1f}s sim, {s['wall_seconds']:.1f}s wall)",
                    flush=True,
                )

    total_wall = time.perf_counter() - t0
    _write_outputs(
        out_path,
        bots,
        results,
        matches,
        seed,
        max_seconds,
        max_goals,
        random_spawn,
        include_mirrors,
        total_wall,
    )
    print(f"[RoundRobin] Completed {len(results)} pairings in {total_wall:.2f}s. Artifacts saved to {out_path}", flush=True)
    return out_path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Headless round-robin evaluation across models + HeuristicBot.")
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max_seconds", type=float, default=60.0)
    parser.add_argument("--max_goals", type=int, default=1)
    parser.add_argument("--no_random_spawn", action="store_true")
    parser.add_argument("--no_heuristic", action="store_true")
    parser.add_argument("--no_mirrors", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    run_round_robin_eval(
        models_dir=args.models_dir,
        out_dir=args.out_dir,
        matches=args.matches,
        seed=args.seed,
        max_seconds=args.max_seconds,
        max_goals=args.max_goals,
        random_spawn=not args.no_random_spawn,
        include_heuristic=not args.no_heuristic,
        include_mirrors=not args.no_mirrors,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
