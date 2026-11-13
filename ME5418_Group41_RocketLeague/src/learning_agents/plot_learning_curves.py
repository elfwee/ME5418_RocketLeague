"""
Learning Curves & Training Dashboard Plotting Tool for ME5418 Rocket League Learning Agents.

Supports:
1. Plotting comprehensive 4-panel dashboards (Return, Goal Rate %, Exploration, Loss) from CSV metrics.
2. Overlaying multi-algorithm comparison curves (e.g., PPO vs Dueling DDQN vs Double DQN).
3. Parsing existing `.log` checkpoint records if `.csv` metrics are not yet available.

Usage:
    # Auto-discover all metrics in runs/ and its per-run folders and generate comparison plot:
    python -m src --plot

    # Plot comparison between specific runs (each training run has its own runs/YYYYMMDD_hhmm_<model>/ folder):
    python -m src --plot --runs runs/20261113_1655_ppo/metrics_ppo.csv runs/20261113_1702_dueling_ddqn/metrics_dueling_ddqn.csv

    # Generate single 4-panel dashboard for a specific CSV (saved next to it unless --output is given):
    python -m src --plot --dashboard runs/20261113_1655_ppo/metrics_ppo.csv
"""

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.learning_agents.dashboard import (
    EXPLORATION_COLUMNS,
    LOSS_COLUMNS,
    metric_panel,
    rolling_mean,
    save_training_dashboard,
)
from src.paths import RUNS_DIR


def run_label(path: Path, prefix: str) -> str:
    """Legend label for a metrics/log file: the model name, prefixed with the run folder's
    timestamp when the file lives in a per-run folder (`runs/YYYYMMDD_hhmm_<model>/`)."""
    name = path.stem.replace(prefix, "").upper()
    stamp = re.match(r"\d{8}_\d{4}", path.parent.name)
    return f"{stamp.group(0)} {name}" if stamp else name


def load_metrics_csv(csv_path: str | Path) -> dict[str, np.ndarray]:
    """Load a metrics CSV file and return a dictionary of numpy arrays."""
    data: dict[str, list[float]] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(np.nan)

    return {k: np.array(v) for k, v in data.items()}


def parse_log_file(log_path: str | Path) -> dict[str, np.ndarray]:
    """Fallback parser for checkpoint log files (e.g. `run_ppo.log`)."""
    steps = []
    episodes = []
    mean_rewards = []
    goal_rates = []
    ep_rewards = []

    pattern = re.compile(
        r"New best 100-ep mean reward:\s*([-\d.]+)\s*\(Goal rate:\s*([-\d.]+)%,\s*Ep reward:\s*([-\d.]+)\)\s*at episode\s*(\d+)\s*\(step\s*(\d+)\)"
    )

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                mr, gr, er, ep, st = match.groups()
                mean_rewards.append(float(mr))
                goal_rates.append(float(gr))
                ep_rewards.append(float(er))
                episodes.append(int(ep))
                steps.append(int(st))

    return {
        "episode": np.array(episodes),
        "step": np.array(steps),
        "reward": np.array(ep_rewards),
        "mean_reward_100": np.array(mean_rewards),
        "goal_rate_100": np.array(goal_rates),
    }


def plot_single_dashboard(data: dict[str, np.ndarray], title: str, output_path: str | Path):
    """Generate a publication-ready 4-panel dashboard for a single run (the same figure training saves)."""
    n = len(data.get("reward", []))
    if n == 0:
        print(f"[Plotting] Warning: No data points found for '{title}'. Skipping.")
        return

    exploration_col = next((c for c in EXPLORATION_COLUMNS if c in data), None)
    loss_col = next((c for c in LOSS_COLUMNS if c in data), None)
    save_training_dashboard(
        output_path,
        f"{title} Training Dashboard",
        data["reward"],
        steps_per_episode=data.get("step"),
        goals_per_episode=data.get("goal"),
        exploration=metric_panel(exploration_col, data[exploration_col]) if exploration_col else None,
        loss=metric_panel(loss_col, data[loss_col]) if loss_col else None,
        mean_rewards=data.get("mean_reward_100"),
        goal_rates=data.get("goal_rate_100"),
    )
    print(f"[Plotting] Saved dashboard to: {output_path}")


def plot_comparison(runs_data: dict[str, dict[str, np.ndarray]], output_path: str | Path):
    """Plot multi-algorithm comparative overlay curves across Environment Steps."""
    if not runs_data:
        print("[Plotting] No valid run data to compare.")
        return

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Sample Efficiency (Mean Return vs Steps)
    for idx, (label, data) in enumerate(runs_data.items()):
        color = colors[idx % len(colors)]
        x = data.get("step") if "step" in data else np.arange(len(data.get("reward", [])))
        if "mean_reward_100" in data:
            y = data["mean_reward_100"]
        elif "reward" in data:
            y = rolling_mean(data["reward"], 100)
        else:
            continue

        axs[0].plot(x, y, label=label, color=color, linewidth=2.2)

    axs[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axs[0].set_title("Sample Efficiency Comparison (100-Ep Rolling Return)", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Environment Steps")
    axs[0].set_ylabel("Rolling Mean Return")
    axs[0].grid(True, alpha=0.3, linestyle="--")
    axs[0].legend(loc="lower right", framealpha=0.8)

    # Subplot 2: Goal Conversion Rate (%)
    for idx, (label, data) in enumerate(runs_data.items()):
        color = colors[idx % len(colors)]
        x = data.get("step") if "step" in data else np.arange(len(data.get("reward", [])))
        if "goal_rate_100" in data:
            axs[1].plot(x, data["goal_rate_100"], label=label, color=color, linewidth=2.2)

    axs[1].axhline(50.0, color="#7f7f7f", linestyle=":", linewidth=1.0, alpha=0.7, label="50% Parity")
    axs[1].set_ylim([-2, 105])
    axs[1].set_title("Objective Competency Comparison (Goal Rate %)", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Environment Steps")
    axs[1].set_ylabel("Goal Rate (%)")
    axs[1].grid(True, alpha=0.3, linestyle="--")
    axs[1].legend(loc="lower right", framealpha=0.8)

    fig.suptitle("Comparative Evaluation: Reinforcement Learning Agents", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plotting] Saved comparative plot to: {output_path}")


def main(args_list: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="python -m src --plot", description="Learning Curve Plotter for ME5418 Rocket League Agents"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="List of CSV metric files or log files to plot/compare",
    )
    parser.add_argument(
        "--dashboard",
        type=str,
        default=None,
        help="Generate a 4-panel single-run dashboard for the specified file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for the output PNG image",
    )
    args = parser.parse_args(args_list)

    if args.dashboard:
        path = Path(args.dashboard)
        if not path.exists():
            print(f"[Plotting] Error: File '{path}' does not exist.")
            return
        data = load_metrics_csv(path) if path.suffix == ".csv" else parse_log_file(path)
        out = args.output if args.output else path.parent / f"dashboard_{path.stem}.png"
        plot_single_dashboard(data, title=path.stem, output_path=out)
        return

    # Multi-run comparison
    run_files = args.runs
    if not run_files:
        # Search for all metrics CSVs in runs/ and its per-run folders
        run_files = sorted(RUNS_DIR.glob("metrics_*.csv")) + sorted(RUNS_DIR.glob("*/metrics_*.csv"))
        if not run_files:
            # Fallback to .log files
            run_files = sorted(RUNS_DIR.glob("run_*.log")) + sorted(RUNS_DIR.glob("*/run_*.log"))

    if not run_files:
        print("[Plotting] No metrics or log files found in runs/ directory.")
        return

    runs_data = {}
    for rf in run_files:
        path = Path(rf)
        if path.suffix == ".csv":
            runs_data[run_label(path, "metrics_")] = load_metrics_csv(path)
        elif path.suffix == ".log":
            parsed = parse_log_file(path)
            if len(parsed.get("step", [])) > 0:
                runs_data[run_label(path, "run_")] = parsed

    out = args.output if args.output else RUNS_DIR / "comparison_learning_curves.png"
    if len(runs_data) == 1:
        name, data = next(iter(runs_data.items()))
        plot_single_dashboard(data, title=name, output_path=out)
    else:
        plot_comparison(runs_data, output_path=out)


if __name__ == "__main__":
    main()
