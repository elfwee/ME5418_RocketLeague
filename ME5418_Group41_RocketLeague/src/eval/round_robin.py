"""Headless round-robin evaluation across a folder of checkpoints and HeuristicBot.

Runs every directed (Blue, Orange) pairing in parallel across CPU workers using the headless
continuous simulation engine (`Simulation` + `PolicyController` / `HeuristicController`),
logs per-match outcomes, and exports summary tables, per-round sample logs, confidence
intervals, and head-to-head matrices. Supports finite rounds or unlimited continuous runs
with advancing, non-overlapping seed blocks.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.paths import ROOT_DIR

DEFAULT_MODELS_DIR = ROOT_DIR / "eval_model"
DEFAULT_OUT_DIR = DEFAULT_MODELS_DIR / "eval_report"

_Z_95 = 1.959963984540054


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


def _ignore_sigint() -> None:
    """Ignore SIGINT inside worker processes so the coordinator shuts down cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _student_t_half_cdf(t: float, df: int) -> float:
    """Compute P(0 <= T <= t) for Student-t(df) via Simpson integration of the exact PDF."""
    if t <= 0.0:
        return 0.0
    log_norm = (
        math.lgamma(0.5 * (df + 1))
        - math.lgamma(0.5 * df)
        - 0.5 * math.log(df * math.pi)
    )
    norm = math.exp(log_norm)
    power = -0.5 * (df + 1)
    n_steps = 200
    h = t / n_steps

    def pdf(x: float) -> float:
        return norm * ((1.0 + (x * x) / df) ** power)

    total = pdf(0.0) + pdf(t)
    for k in range(1, n_steps):
        total += (4.0 if (k & 1) else 2.0) * pdf(k * h)
    return (h / 3.0) * total


def _t_critical_95(df: int) -> float:
    """Return exact two-sided 95% Student-t critical value t_{0.975, df} for df >= 1."""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if df == 1:
        # Cauchy quantile tan(0.475 * pi) avoids wide-tail quadrature bounds.
        return math.tan(0.475 * math.pi)
    if df > 10000:
        return _Z_95

    lo, hi = _Z_95, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _student_t_half_cdf(mid, df) < 0.475:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _wilson_ci_95(wins: int, total: int) -> tuple[float, float, float]:
    """Return (binomial_se_pct, ci95_low_pct, ci95_high_pct) for a single-round binomial sample."""
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = wins / total
    se = math.sqrt(max(0.0, p * (1.0 - p)) / total)
    z2 = _Z_95 * _Z_95
    denom = 1.0 + z2 / total
    center = (p + 0.5 * z2 / total) / denom
    margin = (_Z_95 / denom) * math.sqrt(max(0.0, p * (1.0 - p)) / total + z2 / (4.0 * total * total))
    return 100.0 * se, 100.0 * max(0.0, center - margin), 100.0 * min(1.0, center + margin)


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
        "total_sim_seconds": round(total_sim_seconds, 4),
        "blue_win_rate_pct": round(100.0 * blue_wins / num_matches, 2),
        "orange_win_rate_pct": round(100.0 * orange_wins / num_matches, 2),
        "draw_rate_pct": round(100.0 * draws / num_matches, 2),
        "avg_match_seconds": round(total_sim_seconds / num_matches, 2),
        "wall_seconds": wall_seconds,
    }
    return {"summary": summary, "matches": match_rows}


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via a sibling temp file and os.replace."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _atomic_write_dict_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write dict rows to CSV atomically via a sibling temp file and os.replace."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _atomic_write_rows_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Write list rows to CSV atomically via a sibling temp file and os.replace."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _aggregate_bot_pair_stats(
    bot_id: str,
    bot_ids: list[str],
    by_pair: dict[tuple[str, str], dict],
) -> dict:
    """Aggregate all-pairing, non-mirror, and side-specific statistics for one bot."""
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

    tot_wins = b_wins + o_wins
    tot_losses = b_losses + o_losses
    tot_draws = b_draws + o_draws
    tot_matches = tot_wins + tot_losses + tot_draws

    nm_b = [by_pair[(bot_id, o)] for o in bot_ids if o != bot_id and (bot_id, o) in by_pair]
    nm_o = [by_pair[(b, bot_id)] for b in bot_ids if b != bot_id and (b, bot_id) in by_pair]
    nm_wins = sum(s["blue_wins"] for s in nm_b) + sum(s["orange_wins"] for s in nm_o)
    nm_losses = sum(s["orange_wins"] for s in nm_b) + sum(s["blue_wins"] for s in nm_o)
    nm_draws = sum(s["draws"] for s in nm_b) + sum(s["draws"] for s in nm_o)
    nm_gf = sum(s["blue_goals"] for s in nm_b) + sum(s["orange_goals"] for s in nm_o)
    nm_ga = sum(s["orange_goals"] for s in nm_b) + sum(s["blue_goals"] for s in nm_o)
    nm_matches = nm_wins + nm_losses + nm_draws

    return {
        "matches_all": tot_matches,
        "wins_all": tot_wins,
        "losses_all": tot_losses,
        "draws_all": tot_draws,
        "win_rate_all_pct": (100.0 * tot_wins / tot_matches) if tot_matches else 0.0,
        "goal_diff_all": (b_gf + o_gf) - (b_ga + o_ga),
        "matches_non_mirror": nm_matches,
        "wins_non_mirror": nm_wins,
        "losses_non_mirror": nm_losses,
        "draws_non_mirror": nm_draws,
        "win_rate_non_mirror_pct": (100.0 * nm_wins / nm_matches) if nm_matches else 0.0,
        "goal_diff_non_mirror": nm_gf - nm_ga,
        "blue_wins": b_wins,
        "blue_losses": b_losses,
        "blue_draws": b_draws,
        "orange_wins": o_wins,
        "orange_losses": o_losses,
        "orange_draws": o_draws,
    }


def _accumulate_round_pairings(
    cum_by_pair: dict[tuple[str, str], dict],
    round_results: list[dict],
) -> None:
    """Fold one completed round's pairing summaries into cumulative pairing totals."""
    for res in round_results:
        s = res["summary"]
        key = (s["blue_bot"], s["orange_bot"])
        agg = cum_by_pair.get(key)
        if agg is None:
            agg = {
                "pairing_id": s["pairing_id"],
                "blue_bot": s["blue_bot"],
                "orange_bot": s["orange_bot"],
                "rounds": 0,
                "matches": 0,
                "blue_wins": 0,
                "orange_wins": 0,
                "draws": 0,
                "blue_goals": 0,
                "orange_goals": 0,
                "goal_diff_blue": 0,
                "total_sim_seconds": 0.0,
                "wall_seconds": 0.0,
            }
            cum_by_pair[key] = agg

        agg["rounds"] += 1
        agg["matches"] += s["matches"]
        agg["blue_wins"] += s["blue_wins"]
        agg["orange_wins"] += s["orange_wins"]
        agg["draws"] += s["draws"]
        agg["blue_goals"] += s["blue_goals"]
        agg["orange_goals"] += s["orange_goals"]
        agg["goal_diff_blue"] = agg["blue_goals"] - agg["orange_goals"]
        agg["total_sim_seconds"] = round(agg["total_sim_seconds"] + s["total_sim_seconds"], 4)
        agg["wall_seconds"] = round(agg["wall_seconds"] + s["wall_seconds"], 2)

        m = agg["matches"]
        agg["blue_win_rate_pct"] = round(100.0 * agg["blue_wins"] / m, 2) if m else 0.0
        agg["orange_win_rate_pct"] = round(100.0 * agg["orange_wins"] / m, 2) if m else 0.0
        agg["draw_rate_pct"] = round(100.0 * agg["draws"] / m, 2) if m else 0.0
        agg["avg_match_seconds"] = round(agg["total_sim_seconds"] / m, 2) if m else 0.0


def _score_and_rank_round(
    bot_ids: list[str],
    round_by_pair: dict[tuple[str, str], dict],
) -> tuple[dict[str, dict], dict[str, tuple[int, float, int]]]:
    """Compute per-bot single-round aggregates and competition ranks (1 = best in this round)."""
    agg_by_bot: dict[str, dict] = {}
    scored: list[tuple[float, int, str]] = []
    for bot_id in bot_ids:
        agg = _aggregate_bot_pair_stats(bot_id, bot_ids, round_by_pair)
        agg_by_bot[bot_id] = agg
        use_non_mirror = agg["matches_non_mirror"] > 0
        primary_rate = agg["win_rate_non_mirror_pct"] if use_non_mirror else agg["win_rate_all_pct"]
        primary_gd = agg["goal_diff_non_mirror"] if use_non_mirror else agg["goal_diff_all"]
        scored.append((round(primary_rate, 6), primary_gd, bot_id))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    round_ranks: dict[str, tuple[int, float, int]] = {}
    prev_score: tuple[float, int] | None = None
    cur_rank = 0
    for pos, (rate, gd, bot_id) in enumerate(scored, start=1):
        score = (rate, gd)
        if score != prev_score:
            cur_rank = pos
            prev_score = score
        round_ranks[bot_id] = (cur_rank, rate, gd)
    return agg_by_bot, round_ranks


def _update_round_peak_stats(
    round_peak_stats: dict[str, dict],
    round_ranks: dict[str, tuple[int, float, int]],
    round_seed: int | None,
) -> None:
    """Update per-bot single-round peak rank, reproducible round seed, and Rank-1 count."""
    for bot_id, (round_rank, primary_rate, primary_gd) in round_ranks.items():
        score = (round(primary_rate, 6), primary_gd)
        entry = round_peak_stats.get(bot_id)
        if entry is None:
            round_peak_stats[bot_id] = {
                "best_rank": round_rank,
                "best_score": score,
                "best_seed": round_seed,
                "rounds_rank_1": 1 if round_rank == 1 else 0,
            }
            continue
        if round_rank == 1:
            entry["rounds_rank_1"] += 1
        if round_rank < entry["best_rank"] or (
            round_rank == entry["best_rank"] and score > entry["best_score"]
        ):
            entry["best_rank"] = round_rank
            entry["best_score"] = score
            entry["best_seed"] = round_seed


def _compute_standings(
    bot_ids: list[str],
    cum_by_pair: dict[tuple[str, str], dict],
    round_bot_rates: dict[str, list[float]],
    round_peak_stats: dict[str, dict] | None = None,
) -> list[dict]:
    """Compute standings with raw win%, cross-round mean win%, SE, and 95% confidence interval."""
    rows: list[dict] = []
    for bot_id in bot_ids:
        agg = _aggregate_bot_pair_stats(bot_id, bot_ids, cum_by_pair)
        use_non_mirror = agg["matches_non_mirror"] > 0
        primary_matches = agg["matches_non_mirror"] if use_non_mirror else agg["matches_all"]
        primary_wins = agg["wins_non_mirror"] if use_non_mirror else agg["wins_all"]
        raw_win_pct = (100.0 * primary_wins / primary_matches) if primary_matches else 0.0

        samples = round_bot_rates.get(bot_id, [])
        n_rounds = len(samples)
        if n_rounds >= 2:
            mean_pct = sum(samples) / n_rounds
            var = sum((x - mean_pct) ** 2 for x in samples) / (n_rounds - 1)
            sd_pct = math.sqrt(max(0.0, var))
            se_pct = sd_pct / math.sqrt(n_rounds)
            t_crit = _t_critical_95(n_rounds - 1)
            half_width = t_crit * se_pct
            ci_low = max(0.0, mean_pct - half_width)
            ci_high = min(100.0, mean_pct + half_width)
            ci_method = f"t-CI ({n_rounds} rounds, df={n_rounds - 1})"
        elif n_rounds == 1:
            mean_pct = samples[0]
            sd_pct = None
            se_pct, ci_low, ci_high = _wilson_ci_95(primary_wins, primary_matches)
            half_width = 0.5 * (ci_high - ci_low)
            ci_method = f"Wilson (1 round, N={primary_matches} matches)"
        else:
            mean_pct = raw_win_pct
            sd_pct = None
            se_pct, ci_low, ci_high, half_width = 0.0, 0.0, 0.0, 0.0
            ci_method = "none"

        entry = round_peak_stats.get(bot_id) if round_peak_stats is not None else None
        rounds_rank_1 = entry["rounds_rank_1"] if entry is not None else 0
        rank_1_rate_pct = round(100.0 * rounds_rank_1 / n_rounds, 2) if n_rounds else 0.0

        rows.append(
            {
                "rank": 0,
                "best_rank": 0,
                "best_rank_seed": "",
                "rounds_rank_1": rounds_rank_1,
                "rank_1_rate_pct": rank_1_rate_pct,
                "bot": bot_id,
                "rounds": n_rounds,
                "matches_non_mirror": agg["matches_non_mirror"],
                "wins_non_mirror": agg["wins_non_mirror"],
                "losses_non_mirror": agg["losses_non_mirror"],
                "draws_non_mirror": agg["draws_non_mirror"],
                "raw_win_rate_pct": round(raw_win_pct, 2),
                "mean_win_rate_pct": round(mean_pct, 2),
                "std_dev_pct": "" if sd_pct is None else round(sd_pct, 3),
                "std_error_pct": round(se_pct, 3),
                "ci95_low_pct": round(ci_low, 2),
                "ci95_high_pct": round(ci_high, 2),
                "ci95_half_width_pct": round(half_width, 2),
                "ci_method": ci_method,
                "goal_diff_non_mirror": agg["goal_diff_non_mirror"],
                "matches_all": agg["matches_all"],
                "wins_all": agg["wins_all"],
                "losses_all": agg["losses_all"],
                "draws_all": agg["draws_all"],
                "win_rate_all_pct": round(agg["win_rate_all_pct"], 2),
                "goal_diff_all": agg["goal_diff_all"],
                "blue_wins": agg["blue_wins"],
                "blue_losses": agg["blue_losses"],
                "blue_draws": agg["blue_draws"],
                "orange_wins": agg["orange_wins"],
                "orange_losses": agg["orange_losses"],
                "orange_draws": agg["orange_draws"],
            }
        )

    rows.sort(
        key=lambda r: (
            r["mean_win_rate_pct"],
            r["ci95_low_pct"],
            r["raw_win_rate_pct"],
            r["goal_diff_non_mirror"],
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        bot_id = row["bot"]
        entry = round_peak_stats.get(bot_id) if round_peak_stats is not None else None
        if entry is not None:
            best_rank = entry["best_rank"]
            best_seed = entry["best_seed"]
        else:
            best_rank = idx
            best_seed = None
        row["best_rank"] = best_rank
        row["best_rank_seed"] = "" if best_seed is None else best_seed
    return rows


def _leader_note(standings_rows: list[dict]) -> str:
    """Summarize statistical separation between #1 and #2 in the standings."""
    if not standings_rows:
        return "No bots evaluated."
    top = standings_rows[0]
    top_desc = (
        f"`{top['bot']}` leads with Mean **{top['mean_win_rate_pct']:.2f}%** "
        f"(Raw {top['raw_win_rate_pct']:.2f}%, SE {top['std_error_pct']:.2f}%, "
        f"95% CI `[{top['ci95_low_pct']:.2f}%, {top['ci95_high_pct']:.2f}%]`)"
    )
    if len(standings_rows) < 2:
        return top_desc + "."
    runner_up = standings_rows[1]
    ru_desc = (
        f"`#{runner_up['rank']}` `{runner_up['bot']}` "
        f"({runner_up['mean_win_rate_pct']:.2f}%, 95% CI `[{runner_up['ci95_low_pct']:.2f}%, {runner_up['ci95_high_pct']:.2f}%]`)"
    )
    if top["ci95_low_pct"] > runner_up["ci95_high_pct"]:
        return f"{top_desc} — 95% CI lower bound clears {ru_desc} (non-overlapping 95% CIs)."
    return f"{top_desc} vs {ru_desc} — 95% CIs overlap (continue sampling to tighten intervals)."


def _init_stream_logs(
    out_dir: Path,
    bots: list[BotSpec],
    num_pairings: int,
    matches_per_round: int,
    target_rounds: int | None,
    base_seed: int | None,
    max_seconds: float,
    max_goals: int | None,
    random_spawn: bool,
) -> tuple[Path, Path, Path, list[str], list[str]]:
    """Create output directory and initialize append-only raw match and per-round CSV/log files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = out_dir / "raw_matches.csv"
    raw_log_path = out_dir / "raw_matches.log"
    round_samples_path = out_dir / "round_samples.csv"

    match_fields = [
        "round",
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
    sample_fields = [
        "round",
        "round_seed",
        "bot",
        "round_rank",
        "matches_non_mirror",
        "wins_non_mirror",
        "losses_non_mirror",
        "draws_non_mirror",
        "win_rate_non_mirror_pct",
        "goal_diff_non_mirror",
        "matches_all",
        "wins_all",
        "win_rate_all_pct",
    ]

    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=match_fields).writeheader()

    with round_samples_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=sample_fields).writeheader()

    rounds_desc = "unlimited (continuous)" if target_rounds is None else str(target_rounds)
    with raw_log_path.open("w", encoding="utf-8") as f:
        f.write(
            f"Headless Round-Robin Evaluation | bots={len(bots)} | pairings={num_pairings} | "
            f"matches_per_pairing_per_round={matches_per_round} | target_rounds={rounds_desc} | "
            f"base_seed={base_seed} | max_seconds={max_seconds} | max_goals={max_goals} | "
            f"random_spawn={random_spawn}\n\n"
        )

    return raw_csv_path, raw_log_path, round_samples_path, match_fields, sample_fields


def _append_round_logs(
    raw_csv_path: Path,
    raw_log_path: Path,
    round_samples_path: Path,
    match_fields: list[str],
    sample_fields: list[str],
    round_idx: int,
    round_seed: int | None,
    round_wall_seconds: float,
    round_results: list[dict],
    round_sample_rows: list[dict],
) -> None:
    """Append one completed round's raw matches, log text, and bot sample rows."""
    sorted_results = sorted(round_results, key=lambda r: r["summary"]["pairing_id"])

    with raw_csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=match_fields)
        for r in sorted_results:
            writer.writerows({"round": round_idx, **m} for m in r["matches"])

    with round_samples_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writerows(round_sample_rows)

    with raw_log_path.open("a", encoding="utf-8") as f:
        seed_label = "none" if round_seed is None else str(round_seed)
        f.write(f"########## Round {round_idx:03d} | seed_start={seed_label} | wall={round_wall_seconds:.2f}s ##########\n")
        for r in sorted_results:
            s = r["summary"]
            f.write(
                f"=== Round {round_idx:03d} | Pairing {s['pairing_id']:02d}: "
                f"Blue={s['blue_bot']} vs Orange={s['orange_bot']} | "
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


def _write_summary_outputs(
    out_dir: Path,
    bots: list[BotSpec],
    cum_by_pair: dict[tuple[str, str], dict],
    standings_rows: list[dict],
    round_history_rows: list[dict],
    rounds_completed: int,
    target_rounds: int | None,
    matches_per_round: int,
    base_seed: int | None,
    max_seconds: float,
    max_goals: int | None,
    random_spawn: bool,
    include_mirrors: bool,
    total_wall_seconds: float,
    interrupted: bool = False,
) -> None:
    """Atomically write cumulative summary CSVs, head-to-head matrices, and eval_report.md."""
    bot_ids = [b.bot_id for b in bots]
    cum_summaries = sorted(cum_by_pair.values(), key=lambda s: s["pairing_id"])

    # 1. Pairings summary CSV
    summary_fields = [
        "pairing_id",
        "blue_bot",
        "orange_bot",
        "rounds",
        "matches",
        "blue_wins",
        "orange_wins",
        "draws",
        "blue_goals",
        "orange_goals",
        "goal_diff_blue",
        "blue_win_rate_pct",
        "orange_win_rate_pct",
        "draw_rate_pct",
        "avg_match_seconds",
        "wall_seconds",
    ]
    pairings_export = [{k: s[k] for k in summary_fields} for s in cum_summaries]
    _atomic_write_dict_csv(out_dir / "pairings_summary.csv", summary_fields, pairings_export)

    # 2. Directed Matrix CSVs (rows = Blue bot, cols = Orange bot)
    def cell_or_dash(b_id: str, o_id: str, formatter) -> str:
        s = cum_by_pair.get((b_id, o_id))
        return "-" if s is None else formatter(s)

    def write_matrix(filename: str, formatter) -> None:
        header = ["blue_vs_orange", *bot_ids]
        rows = [[b_id, *[cell_or_dash(b_id, o_id, formatter) for o_id in bot_ids]] for b_id in bot_ids]
        _atomic_write_rows_csv(out_dir / filename, header, rows)

    write_matrix("matrix_score_b_o_d.csv", lambda s: f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']}")
    write_matrix("matrix_blue_win_rate_pct.csv", lambda s: f"{s['blue_win_rate_pct']:.1f}")
    write_matrix("matrix_blue_goal_diff.csv", lambda s: f"{s['goal_diff_blue']}")

    # 3. Overall standings CSV
    if standings_rows:
        _atomic_write_dict_csv(out_dir / "overall_standings.csv", list(standings_rows[0].keys()), standings_rows)

    # 4. Combined Symmetric Head-to-Head Matrix CSV
    sym_rows: list[list[str]] = []
    for row_bot in bot_ids:
        row_cells = []
        for col_bot in bot_ids:
            s_bo = cum_by_pair.get((row_bot, col_bot))
            s_ob = cum_by_pair.get((col_bot, row_bot))
            if s_bo is None or s_ob is None:
                row_cells.append("-")
            else:
                w = s_bo["blue_wins"] + s_ob["orange_wins"]
                l = s_bo["orange_wins"] + s_ob["blue_wins"]
                d = s_bo["draws"] + s_ob["draws"]
                row_cells.append(f"{w}-{l}-{d}")
        sym_rows.append([row_bot, *row_cells])
    _atomic_write_rows_csv(out_dir / "matrix_combined_matches.csv", ["row_vs_col", *bot_ids], sym_rows)

    # 5. Markdown summary report
    total_matches_per_pairing = rounds_completed * matches_per_round
    nm_cap = standings_rows[0]["matches_non_mirror"] if standings_rows else 0
    side_cap = total_matches_per_pairing * (len(bots) if include_mirrors else max(0, len(bots) - 1))
    n_models = sum(1 for b in bots if b.model_path is not None)
    has_heur = any(b.model_path is None for b in bots)
    bot_desc = f"{n_models} models + `heuristic_bot`" if has_heur else f"{n_models} models"
    if base_seed is not None:
        end_seed = base_seed + rounds_completed * matches_per_round - 1
        seed_desc = (
            f"{base_seed} (round `r` uses `[{base_seed} + (r-1)*{matches_per_round}, ...]`, "
            f"covered `{base_seed}..{end_seed}`)"
        )
    else:
        seed_desc = "None (unseeded)"

    rounds_status = (
        f"{rounds_completed} (stopped cleanly on Ctrl+C)"
        if interrupted
        else (f"{rounds_completed}" if target_rounds is None else f"{rounds_completed}/{target_rounds}")
    )

    lines: list[str] = [
        "# Headless Round-Robin Evaluation Report\n\n",
        f"- **Bots:** {len(bots)} ({bot_desc})\n",
        f"- **Rounds completed:** {rounds_status}\n",
        f"- **Pairings:** {len(cum_summaries)} (swapped sides, mirrors={'included' if include_mirrors else 'excluded'})\n",
        f"- **Matches per pairing:** {matches_per_round}/round -> {total_matches_per_pairing} cumulative "
        f"(max_goals={max_goals}, max_seconds={max_seconds:.0f}s, `0-0` timeout = Draw)\n",
        f"- **Total matches:** {len(cum_summaries) * total_matches_per_pairing}\n",
        f"- **Random spawn:** {random_spawn} | **Base seed:** {seed_desc}\n",
        f"- **Wall time:** {total_wall_seconds:.2f}s\n\n",
        "## Overall Standings (Sorted by Mean Win Rate & Statistical Confidence)\n\n",
        f"> **Statistical Leader:** {_leader_note(standings_rows)}\n\n",
        f"| Rank | Best Rank | Best Seed | Rank-1 Rounds | Bot | Rounds | W-L-D (/{nm_cap}) | Raw Win% | Mean ± SE | 95% CI | GD | Blue W-L-D (/{side_cap}) | Orange W-L-D (/{side_cap}) |\n",
        "| ---: | ---: | ---: | :---: | :--- | ---: | :---: | ---: | ---: | :---: | ---: | :---: | :---: |\n",
    ]
    for row in standings_rows:
        seed_disp = "none" if row["best_rank_seed"] == "" else str(row["best_rank_seed"])
        lines.append(
            f"| {row['rank']} | {row['best_rank']} | `{seed_disp}` | {row['rounds_rank_1']}/{row['rounds']} | `{row['bot']}` | {row['rounds']} | "
            f"{row['wins_non_mirror']}-{row['losses_non_mirror']}-{row['draws_non_mirror']} | "
            f"{row['raw_win_rate_pct']:.2f}% | "
            f"{row['mean_win_rate_pct']:.2f} ± {row['std_error_pct']:.2f}% | "
            f"[{row['ci95_low_pct']:.2f}%, {row['ci95_high_pct']:.2f}%] | "
            f"{row['goal_diff_non_mirror']:+d} | "
            f"{row['blue_wins']}-{row['blue_losses']}-{row['blue_draws']} | "
            f"{row['orange_wins']}-{row['orange_losses']}-{row['orange_draws']} |\n"
        )

    ci_note = standings_rows[0]["ci_method"] if standings_rows else "none"
    lines.append(
        f"\n- **CI methodology:** `{ci_note}` — `Raw Win%` pools all non-mirror matches (`wins / matches`). "
        "`Mean ± SE` and `95% CI` treat each completed round (with a fresh non-overlapping seed block) "
        "as an independent sample once `rounds >= 2`, falling back to Binomial SE & Wilson 95% CI on Round 1.\n"
        "- **Per-round Best Rank / Best Seed / Rank-1 Rounds:** `Best Rank` and `Best Seed` track each bot's "
        "highest standalone single-round rank (breaking rank ties by single-round win%, then goal diff, then earliest seed), "
        "so `--rounds 1 --seed <Best Seed>` replays that exact round; `Rank-1 Rounds` counts how many individual rounds finished at single-round rank 1.\n"
    )

    if round_history_rows:
        recent_rows = round_history_rows[-min(len(round_history_rows), 20):]
        heading_suffix = (
            f" (last {len(recent_rows)} of {len(round_history_rows)} rounds; full history in `round_samples.csv`)"
            if len(round_history_rows) > len(recent_rows)
            else ""
        )
        lines.append(f"\n## Round-by-Round Non-Mirror Win% Samples{heading_suffix}\n\n")
        lines.append("| Round | Seed Range | " + " | ".join(f"`{b}`" for b in bot_ids) + " |\n")
        lines.append("| ---: | :---: | " + " | ".join([":---:"] * len(bot_ids)) + " |\n")
        for rr in recent_rows:
            bot_cells = [f"{rr['rates'].get(b, 0.0):.2f}%" for b in bot_ids]
            lines.append(f"| {rr['round']} | `{rr['seed_range']}` | " + " | ".join(bot_cells) + " |\n")

    lines.append(
        f"\n## Directed Head-to-Head Matrix (Row = Blue, Col = Orange; `BlueWins-OrangeWins-Draws` out of {total_matches_per_pairing})\n\n"
    )
    lines.append("| Blue \\ Orange | " + " | ".join(f"`{b}`" for b in bot_ids) + " |\n")
    lines.append("| :--- | " + " | ".join([":---:"] * len(bot_ids)) + " |\n")
    for b_id in bot_ids:
        cells = [cell_or_dash(b_id, o_id, lambda s: f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']}") for o_id in bot_ids]
        lines.append(f"| `{b_id}` | " + " | ".join(cells) + " |\n")

    lines.append(
        f"\n## Combined Swapped-Side Matrix (Row vs Col; `RowWins-ColWins-Draws` out of {2 * total_matches_per_pairing})\n\n"
    )
    lines.append("| Row \\ Col | " + " | ".join(f"`{b}`" for b in bot_ids) + " |\n")
    lines.append("| :--- | " + " | ".join([":---:"] * len(bot_ids)) + " |\n")
    for row_bot, *cells in sym_rows:
        lines.append(f"| `{row_bot}` | " + " | ".join(cells) + " |\n")

    _atomic_write_text(out_dir / "eval_report.md", "".join(lines))


def _print_standings_summary(
    standings_rows: list[dict],
    round_idx: int,
    target_rounds: int | None,
    elapsed_s: float,
) -> None:
    """Print an aligned live standings summary after each completed round."""
    round_label = f"{round_idx}/inf" if target_rounds is None else f"{round_idx}/{target_rounds}"
    nm_cap = standings_rows[0]["matches_non_mirror"] if standings_rows else 0
    bot_w = max(8, *(len(r["bot"]) for r in standings_rows)) if standings_rows else 8

    print(
        f"\n[RoundRobin] Live Standings after Round {round_label} "
        f"(elapsed {elapsed_s:.1f}s, N={nm_cap} non-mirror matches/bot):",
        flush=True,
    )
    header = (
        f"  {'Rank':>4}  {'Best':>4}  {'BestSeed':>8}  {'R#1':>7}  {'Bot':<{bot_w}}  {'W-L-D':^13}  {'Raw%':>7}  "
        f"{'Mean ± SE':>15}  {'95% CI':^17}  {'GD':>6}"
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for row in standings_rows:
        wld = f"{row['wins_non_mirror']}-{row['losses_non_mirror']}-{row['draws_non_mirror']}"
        mean_se = f"{row['mean_win_rate_pct']:5.2f} ± {row['std_error_pct']:5.2f}%"
        ci_str = f"[{row['ci95_low_pct']:5.2f}%, {row['ci95_high_pct']:5.2f}%]"
        seed_disp = "-" if row["best_rank_seed"] == "" else str(row["best_rank_seed"])
        r1_disp = f"{row['rounds_rank_1']}/{row['rounds']}"
        print(
            f"  {row['rank']:>4}  {row['best_rank']:>4}  {seed_disp:>8}  {r1_disp:>7}  {row['bot']:<{bot_w}}  {wld:^13}  "
            f"{row['raw_win_rate_pct']:6.2f}%  {mean_se:>15}  {ci_str:^17}  {row['goal_diff_non_mirror']:+6d}",
            flush=True,
        )


def _run_single_round(
    pool: ProcessPoolExecutor | None,
    pairings: list[tuple[int, BotSpec, BotSpec]],
    matches: int,
    round_seed: int | None,
    max_seconds: float,
    max_goals: int | None,
    random_spawn: bool,
    round_idx: int,
    target_rounds: int | None,
    show_pairing_progress: bool,
) -> list[dict]:
    """Execute every directed pairing once for one round."""
    tasks = [
        (idx, b, o, matches, round_seed, max_seconds, max_goals, random_spawn)
        for idx, b, o in pairings
    ]
    round_label = f"Round {round_idx}" if target_rounds is None else f"Round {round_idx}/{target_rounds}"
    results: list[dict] = []

    if pool is None:
        for done_count, task in enumerate(tasks, start=1):
            res = _run_pairing(task)
            results.append(res)
            if show_pairing_progress:
                s = res["summary"]
                print(
                    f"[{round_label} | {done_count:02d}/{len(tasks):02d}] {s['blue_bot']} (Blue) vs {s['orange_bot']} (Orange): "
                    f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']} "
                    f"(avg {s['avg_match_seconds']:.1f}s sim, {s['wall_seconds']:.1f}s wall)",
                    flush=True,
                )
    else:
        futures = [pool.submit(_run_pairing, task) for task in tasks]
        for done_count, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            results.append(res)
            if show_pairing_progress:
                s = res["summary"]
                print(
                    f"[{round_label} | {done_count:02d}/{len(tasks):02d}] {s['blue_bot']} (Blue) vs {s['orange_bot']} (Orange): "
                    f"{s['blue_wins']}-{s['orange_wins']}-{s['draws']} "
                    f"(avg {s['avg_match_seconds']:.1f}s sim, {s['wall_seconds']:.1f}s wall)",
                    flush=True,
                )

    return results


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
    rounds: int = 1,
    continuous: bool = False,
) -> Path:
    """Run a headless side-swapped round-robin tournament across one or unlimited rounds."""
    if matches < 1:
        raise ValueError(f"matches must be >= 1, got {matches}")
    if rounds < 0:
        raise ValueError(f"rounds must be >= 0 (0 = unlimited), got {rounds}")
    target_rounds: int | None = None if (continuous or rounds == 0) else rounds

    models_path = Path(models_dir)
    out_path = Path(out_dir)
    bots = discover_bots(models_path, include_heuristic=include_heuristic)
    if len(bots) < (1 if include_mirrors else 2):
        raise ValueError(f"Not enough bots found in '{models_path}' (found {len(bots)}).")

    worker_count = max(1, workers if workers is not None else (os.cpu_count() or 4) - 1)
    pairings: list[tuple[int, BotSpec, BotSpec]] = []
    idx = 0
    for b in bots:
        for o in bots:
            if not include_mirrors and b.bot_id == o.bot_id:
                continue
            pairings.append((idx, b, o))
            idx += 1

    rounds_desc = "unlimited (Ctrl+C to stop)" if target_rounds is None else str(target_rounds)
    print(
        f"[RoundRobin] Discovered {len(bots)} bots -> {len(pairings)} pairings x {matches} matches/round "
        f"| rounds={rounds_desc} | base_seed={seed} | max_seconds={max_seconds} | max_goals={max_goals} | "
        f"random_spawn={random_spawn} | workers={worker_count}",
        flush=True,
    )

    raw_csv_path, raw_log_path, round_samples_path, match_fields, sample_fields = _init_stream_logs(
        out_path,
        bots,
        len(pairings),
        matches,
        target_rounds,
        seed,
        max_seconds,
        max_goals,
        random_spawn,
    )

    bot_ids = [b.bot_id for b in bots]
    cum_by_pair: dict[tuple[str, str], dict] = {}
    round_bot_rates: dict[str, list[float]] = {b: [] for b in bot_ids}
    round_peak_stats: dict[str, dict] = {}
    round_history_rows: list[dict] = []
    rounds_completed = 0
    interrupted = False

    t0 = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=worker_count, initializer=_ignore_sigint) if worker_count > 1 else None
    round_iter = itertools.count(1) if target_rounds is None else range(1, target_rounds + 1)

    try:
        for round_idx in round_iter:
            round_seed = (seed + (round_idx - 1) * matches) if seed is not None else None
            seed_range = (
                f"{round_seed}..{round_seed + matches - 1}" if round_seed is not None else "unseeded"
            )
            round_t0 = time.perf_counter()
            if target_rounds is None or target_rounds > 1:
                print(
                    f"\n[RoundRobin] Starting Round {round_idx}"
                    f"{'' if target_rounds is None else f'/{target_rounds}'} (seed range: {seed_range})...",
                    flush=True,
                )

            round_results = _run_single_round(
                pool=pool,
                pairings=pairings,
                matches=matches,
                round_seed=round_seed,
                max_seconds=max_seconds,
                max_goals=max_goals,
                random_spawn=random_spawn,
                round_idx=round_idx,
                target_rounds=target_rounds,
                show_pairing_progress=True,
            )
            round_wall = time.perf_counter() - round_t0

            round_by_pair = {(r["summary"]["blue_bot"], r["summary"]["orange_bot"]): r["summary"] for r in round_results}
            round_agg_by_bot, round_ranks = _score_and_rank_round(bot_ids, round_by_pair)
            _update_round_peak_stats(round_peak_stats, round_ranks, round_seed)

            round_sample_rows: list[dict] = []
            rates_this_round: dict[str, float] = {}
            for bot_id in bot_ids:
                r_agg = round_agg_by_bot[bot_id]
                primary_rate = (
                    r_agg["win_rate_non_mirror_pct"]
                    if r_agg["matches_non_mirror"] > 0
                    else r_agg["win_rate_all_pct"]
                )
                round_bot_rates[bot_id].append(primary_rate)
                rates_this_round[bot_id] = round(primary_rate, 2)
                round_sample_rows.append(
                    {
                        "round": round_idx,
                        "round_seed": "" if round_seed is None else round_seed,
                        "bot": bot_id,
                        "round_rank": round_ranks[bot_id][0],
                        "matches_non_mirror": r_agg["matches_non_mirror"],
                        "wins_non_mirror": r_agg["wins_non_mirror"],
                        "losses_non_mirror": r_agg["losses_non_mirror"],
                        "draws_non_mirror": r_agg["draws_non_mirror"],
                        "win_rate_non_mirror_pct": round(r_agg["win_rate_non_mirror_pct"], 2),
                        "goal_diff_non_mirror": r_agg["goal_diff_non_mirror"],
                        "matches_all": r_agg["matches_all"],
                        "wins_all": r_agg["wins_all"],
                        "win_rate_all_pct": round(r_agg["win_rate_all_pct"], 2),
                    }
                )

            round_history_rows.append(
                {"round": round_idx, "seed_range": seed_range, "rates": rates_this_round}
            )
            _accumulate_round_pairings(cum_by_pair, round_results)
            rounds_completed = round_idx
            standings_rows = _compute_standings(
                bot_ids,
                cum_by_pair,
                round_bot_rates,
                round_peak_stats=round_peak_stats,
            )
            total_wall = time.perf_counter() - t0

            _append_round_logs(
                raw_csv_path,
                raw_log_path,
                round_samples_path,
                match_fields,
                sample_fields,
                round_idx,
                round_seed,
                round_wall,
                round_results,
                round_sample_rows,
            )
            _write_summary_outputs(
                out_dir=out_path,
                bots=bots,
                cum_by_pair=cum_by_pair,
                standings_rows=standings_rows,
                round_history_rows=round_history_rows,
                rounds_completed=rounds_completed,
                target_rounds=target_rounds,
                matches_per_round=matches,
                base_seed=seed,
                max_seconds=max_seconds,
                max_goals=max_goals,
                random_spawn=random_spawn,
                include_mirrors=include_mirrors,
                total_wall_seconds=total_wall,
                interrupted=False,
            )
            _print_standings_summary(standings_rows, round_idx, target_rounds, total_wall)
            print(f"[RoundRobin] Updated artifacts in {out_path}", flush=True)

    except KeyboardInterrupt:
        interrupted = True
        total_wall = time.perf_counter() - t0
        print(
            f"\n[RoundRobin] Interrupted by user after {rounds_completed} completed round(s). "
            "Shutting down workers cleanly...",
            flush=True,
        )
        if rounds_completed > 0:
            standings_rows = _compute_standings(
                bot_ids,
                cum_by_pair,
                round_bot_rates,
                round_peak_stats=round_peak_stats,
            )
            _write_summary_outputs(
                out_dir=out_path,
                bots=bots,
                cum_by_pair=cum_by_pair,
                standings_rows=standings_rows,
                round_history_rows=round_history_rows,
                rounds_completed=rounds_completed,
                target_rounds=target_rounds,
                matches_per_round=matches,
                base_seed=seed,
                max_seconds=max_seconds,
                max_goals=max_goals,
                random_spawn=random_spawn,
                include_mirrors=include_mirrors,
                total_wall_seconds=total_wall,
                interrupted=True,
            )
            print(f"[RoundRobin] Final summary saved for {rounds_completed} completed round(s) in {out_path}", flush=True)
    finally:
        if pool is not None:
            if interrupted:
                procs = list((getattr(pool, "_processes", None) or {}).values())
                pool.shutdown(wait=False, cancel_futures=True)
                for proc in procs:
                    try:
                        if proc.is_alive():
                            proc.terminate()
                    except Exception:
                        pass
                for proc in procs:
                    try:
                        proc.join(timeout=2.0)
                    except Exception:
                        pass
            else:
                pool.shutdown(wait=True)

    total_wall = time.perf_counter() - t0
    if not interrupted:
        print(
            f"[RoundRobin] Completed {rounds_completed} round(s) ({len(pairings)} pairings/round) "
            f"in {total_wall:.2f}s. Artifacts saved to {out_path}",
            flush=True,
        )
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
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of round-robin passes to run (0 runs continuously until stopped; default: 1).",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run unlimited round-robin passes with advancing seeds until Ctrl+C.",
    )
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
        rounds=args.rounds,
        continuous=args.continuous,
    )


if __name__ == "__main__":
    main()
