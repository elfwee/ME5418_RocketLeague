# gym_config
from dataclasses import dataclass


@dataclass
class GymConfig:
    gym_id: str = "rocketleague_v1"
    episode_max_steps: int = 3_000
    train_max_steps: int = 500_000
