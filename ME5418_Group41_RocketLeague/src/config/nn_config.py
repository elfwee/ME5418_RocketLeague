# nn_config
"""Hyperparameters for the DQN and PPO agents.

Read defaults straight off the class (`DQNConfig.hidden_size`), or build a variant for an
experiment with `dataclasses.replace(DQNConfig(), hidden_size=128)`.
"""

import math
import os
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
    factored_action: bool = True  # Discrete(36) matching DDQN; factored checkpoints auto-detect
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
    # Architecture: separate actor & critic MLPs (not parameter-matched to DDQN). Flat by default: one
    # softmax over all 36 (direction, jump, boost) combinations, like the DQN. A flat-36 PPO policy relies
    # on direction x jump x boost dependencies (0.37 nats total correlation, >0.1 in 74% of states), which
    # independent factored heads cannot represent (they explore e.g. dodges in random directions).
    factored_action: bool = False  # True: independent [9, 2, 2] heads (old v2 checkpoints load either way)
    action_dims: tuple[int, int, int] = (9, 2, 2)

    # Optimization & Returns
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    # Loose safety stop only: at 0.02 the update stopped after 1-2 epochs (~32 gradient steps per rollout)
    target_kl: float | None = 0.05
    reward_scale: float = 1.0 / GymConfig.reward_goal  # Scales rewards inside PPO learner only (0.05)

    # Parallel Rollout & Multi-Epoch Minibatch Schedule (larger batches: better-directed steps per unit of KL)
    # One env worker per logical CPU, leaving two for the learner and the OS (throughput plateaus there;
    # 22 envs on a 24-thread CPU: ~2,900 decision steps/s, 46 envs on a 48-thread Threadripper 9960X).
    num_envs: int = max(1, (os.cpu_count() or 3) - 2)
    samples_per_update: int = 8192  # each env collects ceil(samples_per_update / num_envs) steps per rollout
    update_epochs: int = 4
    minibatch_size: int = 1024
    env_workers: str = "subproc"  # "subproc": one process per env; "sync": all envs in this process
    # Learner-process torch threads. Rollouts batch only `num_envs` states, and extra OpenMP threads steal
    # CPU from the env workers (16 default threads ran rollouts ~40% slower than 1-2 threads).
    rollout_threads: int = 1
    update_threads: int = 4

    # Evaluation-based model selection and training length (steps = decision steps summed over all envs).
    # Every `eval_interval_steps` the greedy policy plays `eval_episodes_per_opponent` fixed-seed games
    # against each evaluation opponent; the best score (win rate - loss rate) is saved as the best model,
    # and training stops after `eval_patience` evaluations without a new best (once `min_train_steps`).
    # 20 games per opponent gave a +-0.2 standard error per opponent, so the plateau stop fired on noise.
    # Curriculum self-play rounds use a fixed budget instead (`CurriculumConfig.ppo_v2_self_play_steps`).
    eval_interval_steps: int = 500_000  # 0 disables evaluation (best model then = best training return)
    eval_episodes_per_opponent: int = 50
    eval_patience: int = 4  # 0: never stop on a plateau
    min_train_steps: int = 1_000_000
    max_train_steps: int | None = None

    # Checkpointing
    hyperparameter_set: str = "ppo_v2"

    @property
    def policy_action_dims(self) -> tuple[int, ...]:
        """Policy head sizes: `(36,)` for the flat head, `action_dims` when factored."""
        return self.action_dims if self.factored_action else (math.prod(self.action_dims),)

    @property
    def rollout_steps(self) -> int:
        """Decision steps per env per rollout, so one update sees about `samples_per_update` samples."""
        return math.ceil(self.samples_per_update / self.num_envs)
