# nn_config
from dataclasses import dataclass

@dataclass
class DQNConfig:
    input_dim = (17,)
    output_dim = 4
    hidden_layer = 2
    hidden_size = 256
    learning_rate = 1e-3
    discount_factor = 0.99
    buffer_size = 20_000
    learning_start = 500
    batch_size = 500
    target_update_interval = 500
    epsilon_decay_steps = 8000


