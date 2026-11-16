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
    # Architecture (150,949 trainable parameters when factored_action=False)
    factored_action: bool = False  # Discrete(36) matching DDQN; factored checkpoints auto-detect
    action_dims: tuple[int, int, int] = (9, 2, 2)  # (direction: 9, jump: 2, boost: 2)

    # Optimization & Returns
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02  # Early stopping threshold to preserve trust region
    reward_scale: float = 1.0 / GymConfig.reward_goal  # Scales rewards inside PPO learner only (0.05)

    # Rollout & Multi-Epoch Minibatch Schedule
    rollout_steps: int = 4096
    update_epochs: int = 6
    minibatch_size: int = 256
    demo_rollout_steps: int = 32

    # Checkpointing
    hyperparameter_set: str = "ppo"


@dataclass(frozen=True)
class PPOv2Config(AgentConfig):
    # Architecture (164,366 trainable parameters: separate actor & critic MLPs, factored heads)
    action_dims: tuple[int, int, int] = (9, 2, 2)

    # Optimization & Returns
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    reward_scale: float = 1.0 / GymConfig.reward_goal  # Scales rewards inside PPO learner only (0.05)

    # Parallel Rollout & Multi-Epoch Minibatch Schedule
    num_envs: int = 8  # parallel environments
    rollout_steps: int = 512  # decision steps per env per rollout (8 x 512 = 4096 samples per update)
    update_epochs: int = 6
    minibatch_size: int = 256
    env_workers: str = "subproc"  # "subproc": one process per env; "sync": all envs in this process

    # Periodic Greedy Evaluation
    eval_interval_steps: int = 250_000  # greedy evaluation every this many env steps (0 disables)
    periodic_eval_episodes: int = 50

    # Checkpointing
    hyperparameter_set: str = "ppo_v2"
