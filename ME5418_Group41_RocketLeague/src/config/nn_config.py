# nn_config
from dataclasses import dataclass


@dataclass
class DQNConfig:
    # Architecture (150,949 trainable parameters when enable_dueling_dqn=True)
    hidden_size = 256
    enable_double_dqn = True
    enable_dueling_dqn = True

    # Optimization & Replay Buffer
    learning_rate = 3e-4
    discount_factor = 0.99
    buffer_size = 100_000
    learning_start = 2_000
    batch_size = 64
    train_freq = 4
    action_repeat = 4
    target_update_interval = 1_000
    max_grad_norm = 10.0

    # Exploration Schedule (per-episode exponential decay)
    epsilon_init = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.05

    # Environment & Checkpointing
    bot_type = "none"  # "none" or "heuristic_bot"
    eval_episodes = 5
    runs_dir = "runs"
    hyperparameter_set = "dueling_ddqn"


@dataclass
class PPOConfig:
    # Architecture
    hidden_size = 256
    factored_action = True              # Use MultiDiscrete([9, 2, 2])
    action_dims = (9, 2, 2)             # (direction: 9, jump: 2, boost: 2)

    # Optimization & Returns (matched with DQNConfig where applicable)
    learning_rate = 3e-4
    discount_factor = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    target_kl = 0.02  # Early stopping threshold to preserve trust region

    # Rollout, Action Repeat & Multi-Epoch Minibatch Schedule
    action_repeat = 4
    rollout_steps = 4096
    update_epochs = 6
    minibatch_size = 256
    demo_rollout_steps = 32

    # Environment & Checkpointing
    bot_type = "none"  # "none" or "heuristic_bot"
    eval_episodes = 5
    runs_dir = "runs"
    hyperparameter_set = "ppo"
