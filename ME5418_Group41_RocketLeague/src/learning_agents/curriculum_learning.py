"""
Curriculum Learning

PPO first learns against no opponent (bot_type="none") until its success rate
(rolling goal rate) plateaus, then continues training the same weights against
the heuristic bot (bot_type="heuristic_bot") until it plateaus again, and finally
plays against frozen snapshots of itself (bot_type="self_play") for a fixed budget.
"""

import os
from pathlib import Path
import sys

import numpy as np

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from src.learning_agents.ppo_agent import RUNS_DIR, PPOAgent, device
from src.neural_nets.ppo_torch import ActorCritic

STAGES = ("none", "heuristic_bot", "self_play")
SELF_PLAY_EPISODES = 5_000  # self-play has no fixed target, so it runs for a fixed episode budget
SNAPSHOT_EVERY = 200  # episodes between opponent snapshot refreshes


class SuccessRatePlateau:
    """Stop condition: True once the rolling success rate stops improving.

    A plateau is declared when the best `window`-episode success rate has not
    improved by more than `min_delta` for `patience` episodes.
    """

    def __init__(
        self, window: int = 100, patience: int = 500, min_delta: float = 0.02, max_episodes: int | None = None
    ):
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


def _atomic_save(state_dict, path: Path):
    """Write then rename, so the env never loads a half-written opponent checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


class SelfPlaySnapshot:
    """stop_fn for self-play: every `every` episodes, freeze the current policy as the new
    opponent (the env reloads it on its next reset). Stops after `max_episodes`."""

    def __init__(
        self, agent: PPOAgent, opponent_path: Path, every: int = SNAPSHOT_EVERY, max_episodes: int = SELF_PLAY_EPISODES
    ):
        self.agent = agent
        self.opponent_path = opponent_path
        self.every = every
        self.max_episodes = max_episodes

    def __call__(self, goals_per_episode: list[int]) -> bool:
        n = len(goals_per_episode)
        if n % self.every == 0:
            _atomic_save(self.agent.policy_net.state_dict(), self.opponent_path)
            print(f"[Self-Play] Episode {n}: opponent updated to current policy.")
        return n >= self.max_episodes


def _load_initial_weights(agent: PPOAgent, model_path: str | Path):
    """Seed the agent with a trained checkpoint so a later stage can resume from it."""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    net = ActorCritic(state_dict["fc1.weight"].shape[1], state_dict["actor.weight"].shape[0], agent.fcl_nodes)
    net.load_state_dict(state_dict)
    agent.policy_net = net.to(device)
    print(f"[Curriculum] Initialized agent from: {model_path}")


def curriculum_learning(stages=STAGES, init_model_path: str | Path | None = None):
    """Run the given curriculum stages in order, optionally resuming from a trained checkpoint."""
    ppo_agent = PPOAgent(hyperparameter_set="ppo_curriculum")
    if init_model_path is not None:
        _load_initial_weights(ppo_agent, init_model_path)

    for bot_type in stages:
        print(f"\n[Curriculum] ===== Stage: bot_type={bot_type} =====")
        model_path = RUNS_DIR / f"model_ppo_curriculum_{bot_type}.pt"  # best checkpoint per stage

        if bot_type == "self_play":
            if ppo_agent.policy_net is None:
                raise ValueError("Self-play needs a trained policy: run earlier stages or pass init_model_path.")
            opponent_path = RUNS_DIR / "model_ppo_curriculum_self_play_opponent.pt"
            _atomic_save(ppo_agent.policy_net.state_dict(), opponent_path)  # opponent starts as current self
            stop_fn = SelfPlaySnapshot(ppo_agent, opponent_path)
            env_kwargs = {"opponent_model_path": str(opponent_path)}
        else:
            stop_fn = SuccessRatePlateau()  # fresh detector per stage
            env_kwargs = None

        stats = ppo_agent.run(
            is_training=True,
            bot_type=bot_type,
            stop_fn=stop_fn,
            model_path=model_path,
            env_kwargs=env_kwargs,
        )
        print(f"[Curriculum] Stage '{bot_type}' done: {stats}")


def continue_with_self_play(init_model_path: str | Path = RUNS_DIR / "model_ppo_curriculum_heuristic_bot.pt"):
    """Resume from the heuristic_bot-stage checkpoint and train against itself."""
    curriculum_learning(stages=("self_play",), init_model_path=init_model_path)


if __name__ == "__main__":
    if "--self_play" in sys.argv:
        continue_with_self_play()
    else:
        curriculum_learning()
