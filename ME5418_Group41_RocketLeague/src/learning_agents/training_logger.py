"""Per-episode training records: metrics CSV, best-model tracking and the run log."""

import os
from datetime import datetime, timedelta

import numpy as np

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"


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
