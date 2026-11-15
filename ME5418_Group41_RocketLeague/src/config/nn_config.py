# nn_config
"""Hyperparameters for the DQN and PPO agents.

Read defaults straight off the class (`DQNConfig.hidden_size`), or build a variant for an
experiment with `dataclasses.replace(DQNConfig(), hidden_size=128)`.
"""

from dataclasses import dataclass

from src.config.gym_config import GymConfig


@dataclass(frozen=True)
class AgentConfig:
    """Settings shared by both agents, kept identical so DQN and PPO runs are comparable."""

    # Architecture
    hidden_size: int = 256
    split_trunk: bool = True  # Each stream (actor/critic, value/advantage) gets its own feature trunk

    # Optimization
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    action_repeat: int = 4  # Simulation steps per agent decision

    # Environment & Checkpointing
    bot_type: str = "heuristic_bot"  # "none" or "heuristic_bot"
    env_id: str = GymConfig.gym_id  # Registered gym id the agent trains / evaluates on
    eval_episodes: int = 5
    runs_dir: str = "runs"
    checkpoint_interval: int = 500  # Episodes between periodic checkpoints of the current model (0 disables)
    seed: int | None = None  # Seeds the agent's own generators (DQN replay sampling); None: fresh entropy


@dataclass(frozen=True)
class DQNConfig(AgentConfig):
    # Architecture (231,333 trainable parameters with dueling + split_trunk; 150,949 with a shared trunk)
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
    # Rewards are multiplied by this before the critic sees them (goal +/-20 -> +/-1), so the critic's
    # gradient does not dominate the global grad-norm clip and shrink the actor's step. Logged returns stay raw.
    reward_scale: float = 0.05
    critic_warmup_updates: int = 6  # Fine-tuning: first updates train the critic only (~6 to refit a raw-scale critic)

    # Rollout & Multi-Epoch Minibatch Schedule
    rollout_steps: int = 4096
    update_epochs: int = 6
    minibatch_size: int = 256
    demo_rollout_steps: int = 32

    # Checkpointing
    hyperparameter_set: str = "ppo"
