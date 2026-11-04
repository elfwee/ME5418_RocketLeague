# nn_config
from dataclasses import dataclass


@dataclass
class DQNConfig:
    # Architecture
    input_dim = (52,)
    action_dims = (9, 2, 2)  # env MultiDiscrete action space
    output_dim = 36  # prod(action_dims): one Q-value per flattened action
    hidden_layer = 2
    hidden_size = 256
    enable_double_dqn = True
    enable_dueling_dqn = True

    # Optimization & Replay Buffer (proven hyperparameters from algorithms/dqn)
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
    epsilon_decay_steps = 8000

    # Environment, Stopping & Checkpointing
    bot_type = "heuristic_bot"  # "none" or "heuristic_bot"
    stop_on_reward = 100_000
    demo_max_steps = 5000
    eval_episodes = 5
    runs_dir = "runs"
    hyperparameter_set = "dueling_ddqn"

@dataclass
class PPOConfig:
    # Architecture (matched 1-to-1 with Dueling DQN: 150,949 trainable parameters)
    input_dim = (52,)
    action_dims = (9, 2, 2)
    output_dim = 36
    hidden_layer = 2
    hidden_size = 256

    # Optimization & Returns (matched with DQNConfig where applicable)
    learning_rate = 3e-4
    discount_factor = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    # Rollout, Action Repeat & Multi-Epoch Minibatch Schedule
    action_repeat = 4
    rollout_steps = 2048
    update_epochs = 4
    minibatch_size = 64
    demo_rollout_steps = 32

    # Environment, Stopping & Checkpointing
    bot_type = "heuristic_bot"  # "none" or "heuristic_bot"
    stop_on_reward = 100_000
    eval_episodes = 5
    runs_dir = "runs"
    hyperparameter_set = "ppo"

