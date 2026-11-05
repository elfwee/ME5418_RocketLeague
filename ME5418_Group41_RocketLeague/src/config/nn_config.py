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

    # Optimization & Returns
    learning_rate = 3e-4
    discount_factor = 0.995  # ~200-decision (~13 s at 15 Hz) horizon; differs from DQNConfig (0.99)
    gae_lambda = 0.95
    clip_coef = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    reward_scale = 0.1  # scales buffered rewards (goal +-20 -> +-2) so value loss doesn't swamp policy grads
    target_kl = 0.02  # early-stop update epochs when approx KL > 1.5 * target_kl (None disables)

    # Learning-rate annealing (linear over decisions across all curriculum stages, floored)
    anneal_lr = True
    lr_anneal_steps = 3_000_000
    lr_min_frac = 0.1

    # Rollout, Action Repeat & Multi-Epoch Minibatch Schedule
    action_repeat = 4
    rollout_steps = 4096
    update_epochs = 6
    minibatch_size = 256
    demo_rollout_steps = 32

    # Environment, Stopping & Checkpointing
    bot_type = "none"  # "none" or "heuristic_bot"
    stop_on_reward = 100_000
    eval_episodes = 5
    runs_dir = "runs"
    hyperparameter_set = "ppo"

