"""The 2x2 training dashboard (return, goal rate, exploration, loss)."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib


def rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over up to `window` values (shorter at the start), vectorized O(N)."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr
    csum = np.empty(n + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(arr, out=csum[1:])
    idx = np.arange(n)
    starts = np.maximum(0, idx - (window - 1))
    counts = (idx - starts + 1).astype(np.float64)
    return (csum[idx + 1] - csum[starts]) / counts


@dataclass
class DashboardPanel:
    """One of the two agent-specific dashboard panels (bottom row)."""

    values: list[float] | None
    title: str
    ylabel: str
    label: str
    ylim: tuple[float, float] | None = None


def save_training_dashboard(
    path: str | Path,
    title: str,
    rewards_per_episode: list[float],
    steps_per_episode: list[int] | None,
    goals_per_episode: list[int] | None,
    exploration: DashboardPanel,
    loss: DashboardPanel,
    dpi: int = 150,
):
    """Plot and save the 2x2 training dashboard.

    Subplots:
        1. Sample Efficiency: 100-episode rolling mean reward & episodic return vs. Environment Steps.
        2. Task Competency: 100-episode rolling goal conversion rate (%) vs. Environment Steps.
        3. `exploration`: the agent's exploration signal (epsilon / policy entropy).
        4. `loss`: the agent's optimization loss, raw and 50-episode smoothed.
    """
    import matplotlib.pyplot as plt

    n = len(rewards_per_episode)
    if n == 0:
        return

    x = np.array(
        steps_per_episode if steps_per_episode is not None and len(steps_per_episode) == n else list(range(1, n + 1))
    )
    x_label = "Environment Steps" if steps_per_episode is not None else "Episodes"

    mean_rewards = rolling_mean(rewards_per_episode, 100)
    has_goals = goals_per_episode is not None and len(goals_per_episode) == n

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Sample Efficiency
    ax = axs[0, 0]
    ax.plot(x, rewards_per_episode, color="#1f77b4", alpha=0.18, linewidth=0.8, label="Episodic Return")
    ax.plot(x, mean_rewards, color="#1f77b4", linewidth=2.2, label="100-Ep Rolling Return")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Sample Efficiency (Return vs. Steps)", fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Episodic Return")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.8)

    # 2. Objective Goal Conversion Rate (%)
    ax = axs[0, 1]
    if has_goals:
        goal_rates = rolling_mean(goals_per_episode, 100) * 100.0
        ax.plot(x, goal_rates, color="#2ca02c", linewidth=2.2, label="100-Ep Goal Rate (%)")
        ax.axhline(50.0, color="#9467bd", linestyle=":", linewidth=1.0, alpha=0.7, label="50% Win Parity")
        ax.set_ylim([-2, 105])
    ax.set_title("Objective Competency (Goal Rate %)", fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Goal Rate (%)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.8)

    # 3. Exploration signal
    ax = axs[1, 0]
    values = np.array(exploration.values) if exploration.values is not None else np.array([])
    if len(values) == n:
        ax.plot(x, values, color="#ff7f0e", linewidth=2.0, label=exploration.label)
    elif len(values) > 0:
        ax.plot(values, color="#ff7f0e", linewidth=2.0, label=exploration.label)
    ax.set_title(exploration.title, fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(exploration.ylabel)
    if exploration.ylim is not None:
        ax.set_ylim(list(exploration.ylim))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.8)

    # 4. Optimization loss
    ax = axs[1, 1]
    if loss.values is not None and len(loss.values) == n:
        losses = np.array(loss.values)
        ax.plot(x, losses, color="#d62728", alpha=0.25, linewidth=0.8, label=loss.label)
        ax.plot(x, rolling_mean(losses, 50), color="#d62728", linewidth=2.0, label="50-Ep Smoothed Loss")
        ax.legend(loc="upper right", framealpha=0.8)
    ax.set_title(loss.title, fontsize=11, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(loss.ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
