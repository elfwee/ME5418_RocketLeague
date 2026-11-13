"""
Shared plumbing for the learning agents (`dueling_ddqn_agent.py`, `ppo_agent.py`, `mixed_eval.py`).

Everything here is algorithm-agnostic: environment creation, action repeat, checkpoint loading,
per-run output folders, the evaluation episode loop, training bookkeeping (CSV / log / best-model tracking), the 2x2
training dashboard and GIF export. The learning rules themselves
(replay + TD targets, GAE + clipped surrogate) stay in the agent files.
"""

import itertools
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

from src.config.gym_config import GymConfig
from src.config.sim_config import GIF_FRAME_SIZE
from src.gym_env.rocket_league_env import RocketLeagueEnv  # noqa: F401  (import registers the gym env id)
from src.paths import RUNS_DIR

RUN_DIR_TIME_FORMAT = "%Y%m%d_%H%M"
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
MAX_GIF_FRAMES = 1200

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------------


def normalize_control_car(control_car: str) -> str:
    """Map the bot-vs-bot aliases onto the env's `"both"` mode."""
    return "both" if control_car in ("bot_vs_bot", "bots", "both") else control_car


def make_env(
    bot_type: str,
    discrete_action: bool,
    control_car: str,
    render: bool = False,
    save_gif: bool = False,
    random_spawn: bool = True,
    max_episode_steps: int | None = None,
    env_id: str = GymConfig.gym_id,
) -> gym.Env:
    """Create the Rocket League env; renders to a window, to RGB frames (for GIFs), or not at all."""
    if render:
        render_mode = "human"
    elif save_gif:
        render_mode = "rgb_array"
    else:
        render_mode = None

    make_kwargs = {}
    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = max_episode_steps

    return gym.make(
        env_id,
        render_mode=render_mode,
        bot_type=bot_type,
        discrete_action=discrete_action,
        control_car=control_car,
        random_spawn=random_spawn,
        **make_kwargs,
    )


def obs_to_tensor(obs) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device)


def scored_goal(info: dict, terminated: bool, reward: float) -> bool:
    """True when the decision step ended with the controlled car scoring."""
    return bool(info.get("is_success", False) or (terminated and reward > 0.0))


def clear_jump(action):
    """Return `action` with its jump bit cleared: a flat Discrete(36) index or factored [dir, jump, boost]."""
    if isinstance(action, (list, tuple, np.ndarray)):
        return [int(action[0]), 0, int(action[2])]
    jump_bit = (action % 4) // 2
    return (action - 2) if jump_bit == 1 else action


def step_with_action_repeat(env: gym.Env, action, action_orange=None, action_repeat: int = 1):
    """Repeat chosen action for `action_repeat` frames, releasing jump after frame 0 so
    edge-triggered jumping in Car._apply_jump can fire again on subsequent decisions.

    Observations are only built on the last frame (`_defer_obs`). Returns the usual gym 5-tuple
    with the reward summed over the repeated frames.
    """
    total_reward = 0.0
    terminated = False
    truncated = False
    info = {}
    next_state = None

    # Clear jump bit on repeat frames > 0 so _prev_jump_action resets between decisions
    held_action = clear_jump(action)
    held_action_orange = clear_jump(action_orange) if action_orange is not None else None

    repeat = max(1, action_repeat)
    unwrapped = env.unwrapped
    deferred = False
    for sub in range(repeat):
        deferred = sub < (repeat - 1)
        unwrapped._defer_obs = deferred
        act = action if sub == 0 else held_action
        if action_orange is not None:
            act_orange = action_orange if sub == 0 else held_action_orange
            next_state, reward, terminated, truncated, info = env.step((act, act_orange))
        else:
            next_state, reward, terminated, truncated, info = env.step(act)
        total_reward += float(reward)
        if terminated or truncated or not unwrapped.isopen:
            break

    unwrapped._defer_obs = False
    if deferred:
        next_state = unwrapped._get_obs()
        if unwrapped.control_car == "both":
            info["obs_orange"] = unwrapped._get_obs("orange")

    return next_state, total_reward, terminated, truncated, info


# --------------------------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------


def load_state_dict_if_exists(path: str | Path) -> dict | None:
    """Load a `.pt` state_dict onto `device`, or return None when the file does not exist."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=True)


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


def infer_ppo_arch(state_dict: dict) -> tuple[int, int | list[int] | None]:
    """(hidden_dim, action_dim) of an `ActorCritic` checkpoint. `action_dim` is a list for factored
    MultiDiscrete heads, an int for a flat Discrete head, or None if no actor layer is found."""
    hidden_dim = state_dict["fc1.weight"].shape[0]
    if "actor_heads.0.weight" in state_dict:
        n_heads = len({k.split(".")[1] for k in state_dict if k.startswith("actor_heads.")})
        return hidden_dim, [state_dict[f"actor_heads.{i}.weight"].shape[0] for i in range(n_heads)]
    if "actor.weight" in state_dict:
        return hidden_dim, state_dict["actor.weight"].shape[0]
    return hidden_dim, None


def infer_dqn_arch(state_dict: dict) -> tuple[int, int, bool]:
    """(hidden_dim, action_dim, is_dueling) of a `DQN` checkpoint."""
    dueling = "advantages.weight" in state_dict
    out_key = "advantages.weight" if dueling else "fc3.weight"
    return state_dict["fc1.weight"].shape[0], state_dict[out_key].shape[0], dueling


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


def resolve_eval_episodes(num_episodes: int | None, render: bool, save_gif: bool, default: int) -> int | None:
    """Episodes to evaluate: unlimited (None) only when watching a rendered window with no limit given."""
    if render and num_episodes is None and not save_gif:
        return None
    return num_episodes if num_episodes is not None else default


def summarize_episodes(env: gym.Env, rewards: Sequence[float], goals: Sequence[int]) -> dict[str, float]:
    """Run summary returned by every train/eval entry point."""
    return {
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "goal_rate": float(np.mean(goals)) * 100.0 if goals else 0.0,
        "score_blue": int(env.unwrapped.sim.score_blue),
        "score_orange": int(env.unwrapped.sim.score_orange),
    }


def run_eval_episodes(
    env: gym.Env,
    select_blue: Callable[[torch.Tensor], object],
    select_orange: Callable[[torch.Tensor], object] | None = None,
    *,
    action_repeat: int,
    num_episodes: int | None,
    gif_path: str | Path | None = None,
    gif_tag: str = "Eval",
) -> dict[str, float]:
    """Play greedy evaluation episodes and return the run summary.

    `select_blue` / `select_orange` map an observation tensor to an env action. `select_orange` is
    only used when the env reports an orange observation (`control_car="both"`). When `gif_path`
    is set, the first episode is recorded and saved as a GIF.
    """
    rewards_per_episode = []
    goals_per_episode = []
    gif_frames = []
    recording = gif_path is not None

    episode_iter = range(num_episodes) if num_episodes is not None else itertools.count()
    for episode in episode_iter:
        state, reset_info = env.reset()
        state = obs_to_tensor(state)
        state_orange = None
        if select_orange is not None and "obs_orange" in reset_info:
            state_orange = obs_to_tensor(reset_info["obs_orange"])

        if not env.unwrapped.isopen:
            break

        terminated = False
        truncated = False
        episode_reward = 0.0
        goal = 0

        while not (terminated or truncated):
            action = select_blue(state)
            action_orange = select_orange(state_orange) if state_orange is not None else None

            next_state, reward, terminated, truncated, info = step_with_action_repeat(
                env, action, action_orange, action_repeat
            )

            if recording and len(gif_frames) < MAX_GIF_FRAMES:
                frame = env.unwrapped.get_frame()
                if frame is not None:
                    gif_frames.append(frame)

            episode_reward += reward
            if scored_goal(info, terminated, reward):
                goal = 1

            state = obs_to_tensor(next_state)
            if select_orange is not None and "obs_orange" in info:
                state_orange = obs_to_tensor(info["obs_orange"])

            if not env.unwrapped.isopen:
                break

        rewards_per_episode.append(episode_reward)
        goals_per_episode.append(goal)

        if recording and len(gif_frames) > 0:
            save_gif(gif_frames, gif_path, tag=gif_tag)
            recording = False

        print(
            f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
            f"Score(Blue-Orange) = {int(env.unwrapped.sim.score_blue)}-{int(env.unwrapped.sim.score_orange)} | "
            f"Goal = {bool(goal)} | Terminated = {terminated} | Truncated = {truncated}"
        )

    return summarize_episodes(env, rewards_per_episode, goals_per_episode)


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 15, tag: str = "Eval"):
    """Compile and save a list of RGB frames as an optimized animated GIF."""
    from PIL import Image

    if not frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Downscale by 50% (GIF_FRAME_SIZE) for smooth playback and lightweight GIF file size
    pil_images = [Image.fromarray(f).resize(GIF_FRAME_SIZE, Image.Resampling.BILINEAR) for f in frames]
    pil_images[0].save(
        str(path),
        save_all=True,
        append_images=pil_images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"[{tag} Eval] Saved evaluation GIF ({len(pil_images)} frames, {fps} FPS) to: {path}")


# --------------------------------------------------------------------------------------------
# Training bookkeeping
# --------------------------------------------------------------------------------------------


class TrainingLogger:
    """Per-episode training records shared by both agents.

    Tracks episodic return / goal / step histories, appends one row per episode to the metrics
    CSV, decides when the 100-episode rolling mean return is a new best (the agent then saves its
    model), and throttles how often the dashboard graph is redrawn.

    `extra_columns` names the two agent-specific CSV columns (e.g. `("epsilon", "td_loss")`).
    """

    MIN_EPISODES_FOR_BEST = 20
    GRAPH_INTERVAL = timedelta(seconds=30)

    def __init__(self, log_file: str, metrics_file: str, extra_columns: tuple[str, str]):
        self.log_file = log_file
        self.metrics_file = metrics_file
        self.extra_columns = extra_columns

        self.rewards: list[float] = []
        self.goals: list[int] = []
        self.steps: list[int] = []

        self.mean_reward_100 = 0.0
        self.goal_rate_100 = 0.0
        self.best_mean_reward = float("-inf")
        self._last_graph_time = datetime.now()

    def record_episode(self, episode_reward: float, goal: int) -> tuple[float, float]:
        """Store an episode's outcome; returns the (100-episode mean return, 100-episode goal rate %)."""
        self.rewards.append(episode_reward)
        self.goals.append(goal)
        self.mean_reward_100 = float(np.mean(self.rewards[-100:]))
        self.goal_rate_100 = float(np.mean(self.goals[-100:])) * 100.0
        return self.mean_reward_100, self.goal_rate_100

    def write_metrics(self, episode: int, step_count: int, extras: tuple[float, float]):
        """Append the last recorded episode to the CSV log for offline plotting & evaluation."""
        self.steps.append(step_count)
        write_header = not os.path.exists(self.metrics_file) or os.path.getsize(self.metrics_file) == 0
        with open(self.metrics_file, "a", newline="") as f:
            if write_header:
                f.write(
                    "episode,step,reward,mean_reward_100,goal,goal_rate_100,"
                    f"{self.extra_columns[0]},{self.extra_columns[1]}\n"
                )
            f.write(
                f"{episode},{step_count},{self.rewards[-1]:.4f},{self.mean_reward_100:.4f},"
                f"{self.goals[-1]},{self.goal_rate_100:.2f},{extras[0]:.4f},{extras[1]:.6f}\n"
            )

    def is_new_best(self, episode: int, num_episodes: int | None) -> bool:
        """True when the rolling mean return beats the best so far (after a short warm-up), or on
        the final episode of a short run that never saved a model."""
        improved = len(self.rewards) >= self.MIN_EPISODES_FOR_BEST and self.mean_reward_100 > self.best_mean_reward
        last_chance = (
            num_episodes is not None and episode == num_episodes - 1 and self.best_mean_reward == float("-inf")
        )
        return improved or last_chance

    def log_new_best(self, episode: int, step_count: int):
        """Record the new best rolling mean return in the console and the run's `.log` file."""
        self.best_mean_reward = self.mean_reward_100
        log_message = (
            f"{datetime.now().strftime(DATE_FORMAT)} - New best 100-ep mean reward: "
            f"{self.mean_reward_100:.2f} (Goal rate: {self.goal_rate_100:.1f}%, Ep reward: {self.rewards[-1]:.2f}) "
            f"at episode {episode} (step {step_count}). Saving model."
        )
        print(log_message)
        with open(self.log_file, "a") as log_file:
            log_file.write(log_message + "\n")

    def graph_due(self) -> bool:
        """True (and restarts the timer) when it is time to redraw the training dashboard."""
        now = datetime.now()
        if now - self._last_graph_time > self.GRAPH_INTERVAL:
            self._last_graph_time = now
            return True
        return False


# --------------------------------------------------------------------------------------------
# Training dashboard
# --------------------------------------------------------------------------------------------


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
