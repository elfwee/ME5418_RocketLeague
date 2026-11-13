"""Learning Agents package for training and evaluating RL policies on RocketLeagueEnv."""

from src.learning_agents.dueling_ddqn_agent import (
    Agent,
    DuelingDDQNAgent,
    evaluate_dueling_ddqn,
    train_dueling_ddqn,
)
from src.learning_agents.experience_replay import ReplayMemory
from src.learning_agents.mixed_eval import evaluate_mixed_policies
from src.learning_agents.ppo_agent import (
    PPOAgent,
    evaluate_ppo,
    train_ppo,
)
from src.learning_agents.plot_learning_curves import (
    plot_comparison,
    plot_single_dashboard,
)

__all__ = [
    "Agent",
    "DuelingDDQNAgent",
    "PPOAgent",
    "ReplayMemory",
    "train_dueling_ddqn",
    "evaluate_dueling_ddqn",
    "train_ppo",
    "evaluate_ppo",
    "evaluate_mixed_policies",
    "plot_comparison",
    "plot_single_dashboard",
]
