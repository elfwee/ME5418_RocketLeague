"""
    Curriculum Learning

    PPO first learns against no opponent (bot_type="none") until its success rate
    (rolling goal rate) plateaus, then continues training the same weights against
    the heuristic bot (bot_type="heuristic_bot").
"""

from pathlib import Path
import sys

import numpy as np

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.learning_agents.ppo_agent import RUNS_DIR, PPOAgent


class SuccessRatePlateau:
    """Stop condition: True once the rolling success rate stops improving.

    A plateau is declared when the best `window`-episode success rate has not
    improved by more than `min_delta` for `patience` episodes.
    """

    def __init__(self, window: int = 100, patience: int = 500, min_delta: float = 0.02, max_episodes: int | None = None):
        self.window = window
        self.patience = patience
        self.min_delta = min_delta
        self.max_episodes = max_episodes
        self.best_rate = float("-inf")
        self.best_episode = 0

    def __call__(self, goals_per_episode: list[int]) -> bool:
        n = len(goals_per_episode)
        if self.max_episodes is not None and n >= self.max_episodes:
            print(f"[Curriculum] Reached max_episodes={self.max_episodes}.")
            return True
        if n < self.window:
            return False

        rate = float(np.mean(goals_per_episode[-self.window :]))
        if rate > self.best_rate + self.min_delta:
            self.best_rate = rate
            self.best_episode = n
            return False

        if n - self.best_episode >= self.patience:
            print(
                f"[Curriculum] Success rate plateaued at {self.best_rate * 100:.1f}% "
                f"(no +{self.min_delta * 100:.1f}% gain in {self.patience} episodes)."
            )
            return True
        return False


def curriculum_learning():
    stages = ["none", "heuristic_bot"]
    ppo_agent = PPOAgent(hyperparameter_set="ppo_curriculum")

    for bot_type in stages:
        print(f"\n[Curriculum] ===== Stage: bot_type={bot_type} =====")
        stats = ppo_agent.run(
            is_training=True,
            bot_type=bot_type,
            stop_fn=SuccessRatePlateau(),  # fresh detector per stage
            model_path=RUNS_DIR / f"model_ppo_curriculum_{bot_type}.pt",  # best checkpoint per stage
        )
        print(f"[Curriculum] Stage '{bot_type}' done: {stats}")


if __name__ == "__main__":
    curriculum_learning()
