"""Headless round-robin evaluation of curated checkpoints + HeuristicBot."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Keep single-threaded CPU BLAS/torch per worker process to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[0]


@dataclass(frozen=True)
class BotSpec:
    bot_id: str
    model_path: str | None  # None -> HeuristicBot


def discover_bots(eval_model_dir: Path) -> list[BotSpec]:
    bots = [BotSpec(bot_id="heuristic_bot", model_path=None)]
    for pt_file in sorted(eval_model_dir.glob("*.pt")):
        bots.append(BotSpec(bot_id=pt_file.stem, model_path=str(pt_file)))
    return bots


def run_pairing(task: tuple[str, int, BotSpec, BotSpec, int, int, float]) -> dict:
    repo_root, pairing_idx, blue_spec, orange_spec, num_matches, base_seed, max_seconds = task
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

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
    max_steps = int(round(max_seconds * SIM_HZ))
    wall_start = time.perf_counter()

    match_rows = []
    blue_wins = 0
    orange_wins = 0
    draws = 0
    total_sim_seconds = 0.0

    for match_idx in range(num_matches):
        match_seed = base_seed + match_idx
        sim.rng.seed(match_seed)
        sim.reset(reset_scores=True, spawn_index=match_idx, random_spawn=True, clear_events=True)
        blue_ctrl.reset()
        orange_ctrl.reset()

        steps = 0
        for steps in range(1, max_steps + 1):
            sim.step(action=blue_ctrl.get_action(), dt=dt, action_orange=orange_ctrl.get_action())
            if sim.score_blue + sim.score_orange >= 1:
                break

        sim_seconds = round(steps * dt, 4)
        total_sim_seconds += sim_seconds
        b_goals = int(sim.score_blue)
        o_goals = int(sim.score_orange)
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
                "seed": match_seed,
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
        "blue_goals": blue_wins,
        "orange_goals": orange_wins,
        "goal_diff_blue": blue_wins - orange_wins,
        "blue_win_rate_pct": round(100.0 * blue_wins / num_matches, 2),
        "orange_win_rate_pct": round(100.0 * orange_wins / num_matches, 2),
        "draw_rate_pct": round(100.0 * draws / num_matches, 2),
        "avg_match_seconds": round(total_sim_seconds / num_matches, 2),
        "wall_seconds": wall_seconds,
    }
    return {"summary": summary, "matches": match_rows}


def write_outputs(
    out_dir: Path,
    bots: list[BotSpec],
    results: list[dict],
    num_matches: int,
    base_seed: int,
    max_seconds: float,
    total_wall_seconds: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["summary"]["pairing_id"])
    bot_ids = [b.bot_id for b in bots]
    by_pair = {(r["summary"]["blue_bot"], r["summary"]["orange_bot"]): r["summary"] for r in results}

    # 1. Raw matches CSV (8,100 rows)
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

    # 2. Raw text log (pairing summaries + per-match outcomes)
    raw_log_path = out_dir / "raw_matches.log"
    with raw_log_path.open("w", encoding="utf-8") as f:
        f.write(
            f"Headless Round-Robin Evaluation | bots={len(bots)} | pairings={len(results)} | "
            f"matches_per_pairing={num_matches} | base_seed={base_seed} | max_seconds={max_seconds} | "
            f"total_wall_seconds={total_wall_seconds:.2f}\n\n"
        )
        for r in results:
            s = r["summary"]
            f.write(
                f"=== Pairing {s['pairing_id']:02d}: Blue={s['blue_bot']} vs Orange={s['orange_bot']} | "
                f"Score(B-O-D)={s['blue_wins']}-{s['orange_wins']}-{s['draws']} | "
                f"AvgSim={s['avg_match_seconds']:.2f}s | Wall={s['wall_seconds']:.2f}s ===\n"
            )
            for m in r["matches"]:
                f.write(
                    f"  match={m['match_index']:03d} seed={m['seed']} winner={m['winner']:<6} "
                    f"score={m['blue_goals']}-{m['orange_goals']} sim_s={m['sim_seconds']:6.2f} "
                    f"steps={m['steps']:4d} last_touch={m['last_touch'] or 'none'}\n"
                )
            f.write("\n")

    # 3. Pairings summary CSV (81 rows)
    pairings_csv_path = out_dir / "pairings_summary.csv"
    summary_fields = list(results[0]["summary"].keys())
    with pairings_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r["summary"])

    # 4. Matrix CSVs (rows = Blue bot, cols = Orange bot)
    def write_matrix(filename: str, cell_fn):
        path = out_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["blue_vs_orange", *bot_ids])
            for b_id in bot_ids:
                writer.writerow([b_id, *[cell_fn(by_pair[(b_id, o_id)]) for o_id in bot_ids]])

    write_matrix("matrix_score_b_o_d.csv", lambda s: f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']}")
    write_matrix("matrix_blue_win_rate_pct.csv", lambda s: f"{s['blue_win_rate_pct']:.1f}")
    write_matrix("matrix_blue_goal_diff.csv", lambda s: f"{s['goal_diff_blue']}")

    # 5. Overall standings CSV (combining Blue + Orange sides, both with and without mirror)
    standings_rows = []
    for bot_id in bot_ids:
        b_wins = sum(by_pair[(bot_id, o)]["blue_wins"] for o in bot_ids)
        b_losses = sum(by_pair[(bot_id, o)]["orange_wins"] for o in bot_ids)
        b_draws = sum(by_pair[(bot_id, o)]["draws"] for o in bot_ids)

        o_wins = sum(by_pair[(b, bot_id)]["orange_wins"] for b in bot_ids)
        o_losses = sum(by_pair[(b, bot_id)]["blue_wins"] for b in bot_ids)
        o_draws = sum(by_pair[(b, bot_id)]["draws"] for b in bot_ids)

        tot_matches = (b_wins + b_losses + b_draws) + (o_wins + o_losses + o_draws)
        tot_wins = b_wins + o_wins
        tot_losses = b_losses + o_losses
        tot_draws = b_draws + o_draws

        # Non-mirror totals (against the 8 distinct opponents)
        nm_b_wins = sum(by_pair[(bot_id, o)]["blue_wins"] for o in bot_ids if o != bot_id)
        nm_b_losses = sum(by_pair[(bot_id, o)]["orange_wins"] for o in bot_ids if o != bot_id)
        nm_b_draws = sum(by_pair[(bot_id, o)]["draws"] for o in bot_ids if o != bot_id)
        nm_o_wins = sum(by_pair[(b, bot_id)]["orange_wins"] for b in bot_ids if b != bot_id)
        nm_o_losses = sum(by_pair[(b, bot_id)]["blue_wins"] for b in bot_ids if b != bot_id)
        nm_o_draws = sum(by_pair[(b, bot_id)]["draws"] for b in bot_ids if b != bot_id)
        nm_matches = (nm_b_wins + nm_b_losses + nm_b_draws) + (nm_o_wins + nm_o_losses + nm_o_draws)
        nm_wins = nm_b_wins + nm_o_wins
        nm_losses = nm_b_losses + nm_o_losses
        nm_draws = nm_b_draws + nm_o_draws

        standings_rows.append(
            {
                "bot": bot_id,
                "matches_all": tot_matches,
                "wins_all": tot_wins,
                "losses_all": tot_losses,
                "draws_all": tot_draws,
                "win_rate_all_pct": round(100.0 * tot_wins / tot_matches, 2),
                "goal_diff_all": tot_wins - tot_losses,
                "matches_non_mirror": nm_matches,
                "wins_non_mirror": nm_wins,
                "losses_non_mirror": nm_losses,
                "draws_non_mirror": nm_draws,
                "win_rate_non_mirror_pct": round(100.0 * nm_wins / nm_matches, 2),
                "goal_diff_non_mirror": nm_wins - nm_losses,
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

    # 6. Combined Symmetric Head-to-Head Matrix (200 matches per pair: 100 Blue + 100 Orange)
    sym_csv_path = out_dir / "matrix_combined_200_matches.csv"
    with sym_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_vs_col", *bot_ids])
        for row_bot in bot_ids:
            row_cells = []
            for col_bot in bot_ids:
                s_bo = by_pair[(row_bot, col_bot)]
                s_ob = by_pair[(col_bot, row_bot)]
                w = s_bo["blue_wins"] + s_ob["orange_wins"]
                l = s_bo["orange_wins"] + s_ob["blue_wins"]
                d = s_bo["draws"] + s_ob["draws"]
                row_cells.append(f"{w}-{l}-{d}")
            writer.writerow([row_bot, *row_cells])

    # 7. Markdown summary report
    report_md_path = out_dir / "eval_report.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("# Headless Round-Robin Evaluation Report\n\n")
        f.write(
            f"- **Bots:** {len(bots)} (8 curated models + `heuristic_bot`)\n"
            f"- **Pairings:** {len(results)} ($9 \\times 9$, including swapped sides and mirror matches)\n"
            f"- **Matches per pairing:** {num_matches} (1-goal / {max_seconds:.0f}s episode, `0-0` timeout = Draw)\n"
            f"- **Total matches:** {len(results) * num_matches}\n"
            f"- **Base seed:** {base_seed} (match `i` uses `seed = {base_seed} + i` with `random_spawn=True`)\n"
            f"- **Wall time:** {total_wall_seconds:.2f}s\n\n"
        )
        f.write("## Overall Standings (Sorted by Non-Mirror Win Rate)\n\n")
        f.write("| Rank | Bot | W-L-D (Non-Mirror, /1600) | Win% (Non-Mirror) | GD (Non-Mirror) | Blue W-L-D (/900) | Orange W-L-D (/900) |\n")
        f.write("| ---: | :--- | :---: | ---: | ---: | :---: | :---: |\n")
        for idx, row in enumerate(standings_rows, start=1):
            f.write(
                f"| {idx} | `{row['bot']}` | "
                f"{row['wins_non_mirror']}-{row['losses_non_mirror']}-{row['draws_non_mirror']} | "
                f"{row['win_rate_non_mirror_pct']:.2f}% | {row['goal_diff_non_mirror']:+d} | "
                f"{row['blue_wins']}-{row['blue_losses']}-{row['blue_draws']} | "
                f"{row['orange_wins']}-{row['orange_losses']}-{row['orange_draws']} |\n"
            )
        f.write("\n## Directed Head-to-Head Matrix (Row = Blue, Col = Orange; `BlueWins-OrangeWins-Draws` out of 100)\n\n")
        f.write("| Blue \\ Orange | " + " | ".join(f"`{b}`" for b in bot_ids) + " |\n")
        f.write("| :--- | " + " | ".join([":---:"] * len(bot_ids)) + " |\n")
        for b_id in bot_ids:
            cells = [
                f"{by_pair[(b_id, o_id)]['blue_wins']}-{by_pair[(b_id, o_id)]['orange_wins']}-{by_pair[(b_id, o_id)]['draws']}"
                for o_id in bot_ids
            ]
            f.write(f"| `{b_id}` | " + " | ".join(cells) + " |\n")

        f.write("\n## Combined Swapped-Side Matrix (Row vs Col; `RowWins-ColWins-Draws` out of 200)\n\n")
        f.write("| Row \\ Col | " + " | ".join(f"`{b}`" for b in bot_ids) + " |\n")
        f.write("| :--- | " + " | ".join([":---:"] * len(bot_ids)) + " |\n")
        for row_bot in bot_ids:
            cells = []
            for col_bot in bot_ids:
                s_bo = by_pair[(row_bot, col_bot)]
                s_ob = by_pair[(col_bot, row_bot)]
                w = s_bo["blue_wins"] + s_ob["orange_wins"]
                l = s_bo["orange_wins"] + s_ob["blue_wins"]
                d = s_bo["draws"] + s_ob["draws"]
                cells.append(f"{w}-{l}-{d}")
            f.write(f"| `{row_bot}` | " + " | ".join(cells) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--eval-model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--limit-pairings", type=int, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_root))

    bots = discover_bots(args.eval_model_dir)
    tasks = []
    idx = 0
    for b in bots:
        for o in bots:
            tasks.append((str(args.repo_root), idx, b, o, args.matches, args.seed, args.max_seconds))
            idx += 1
    if args.limit_pairings is not None:
        tasks = tasks[: args.limit_pairings]

    print(
        f"[RoundRobin] Discovered {len(bots)} bots -> {len(tasks)} pairings x {args.matches} matches "
        f"| seed={args.seed} | max_seconds={args.max_seconds} | workers={args.workers}",
        flush=True,
    )
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_pairing, task) for task in tasks]
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
    write_outputs(args.out_dir, bots, results, args.matches, args.seed, args.max_seconds, total_wall)
    print(f"[RoundRobin] Completed {len(results)} pairings in {total_wall:.2f}s. Artifacts saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
