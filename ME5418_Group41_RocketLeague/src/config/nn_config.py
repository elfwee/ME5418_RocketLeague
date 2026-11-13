# nn_config
"""Hyperparameters for the DQN and PPO agents.

Read defaults straight off the class (`DQNConfig.hidden_size`), or build a variant for an
experiment with `dataclasses.replace(DQNConfig(), hidden_size=128)`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Settings shared by both agents, kept identical so DQN and PPO runs are comparable."""

    # Architecture
    hidden_size: int = 256

    # Optimization
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    action_repeat: int = 4  # Simulation steps per agent decision

    # Environment & Checkpointing
    bot_type: str = "heuristic_bot"  # "none" or "heuristic_bot"
    eval_episodes: int = 5
    runs_dir: str = "runs"
    checkpoint_interval: int = 500  # Episodes between periodic checkpoints of the current model (0 disables)


@dataclass(frozen=True)
class DQNConfig(AgentConfig):
    # Architecture (150,949 trainable parameters when enable_dueling_dqn=True)
    enable_double_dqn: bool = True
    enable_dueling_dqn: bool = True

    # Optimization & Replay Buffer
    buffer_size: int = 100_000
    learning_start: int = 2_000
    batch_size: int = 64
    train_freq: int = 4
    target_update_interval: int = 1_000
    max_grad_norm: float = 10.0

    # Exploration Schedule (per-episode exponential decay)
    epsilon_init: float = 1.0
    epsilon_finetune: float = 0.20
    epsilon_decay: float = 0.9995
    epsilon_min: float = 0.05

    # Checkpointing
    hyperparameter_set: str = "dueling_ddqn"


@dataclass(frozen=True)
class PPOConfig(AgentConfig):
    # Architecture
    factored_action: bool = True  # Use MultiDiscrete([9, 2, 2])
    action_dims: tuple[int, int, int] = (9, 2, 2)  # (direction: 9, jump: 2, boost: 2)

    # Optimization & Returns
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.02  # Early stopping threshold to preserve trust region

    # Rollout & Multi-Epoch Minibatch Schedule
    rollout_steps: int = 4096
    update_epochs: int = 6
    minibatch_size: int = 256
    demo_rollout_steps: int = 32

    # Checkpointing
    hyperparameter_set: str = "ppo"
