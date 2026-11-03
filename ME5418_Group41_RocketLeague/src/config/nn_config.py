# nn_config
from dataclasses import dataclass

@dataclass
class DQNConfig:
    input_dim = (52,)
    action_dims = (9, 2, 2)  # env MultiDiscrete action space
    output_dim = 36  # prod(action_dims): one Q-value per flattened action
    hidden_layer = 2
    hidden_size = 256
    learning_rate = 1e-3
    discount_factor = 0.99
    buffer_size = 16_384
    learning_start = 128
    batch_size = 64
    target_update_interval = 500
    epsilon_decay_steps = 8000
    demo_max_steps = 5000


