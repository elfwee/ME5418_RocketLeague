# gym_config
from dataclasses import dataclass


@dataclass
class GymConfig:
    gym_id = "rocketleague_v1"
    episode_max_steps = 3_000
    train_max_steps = 500_000
    directions = [(0, 0), (1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 1), (-1, -1), (-1, 1)]
