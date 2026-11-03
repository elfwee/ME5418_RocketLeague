# gym_config
from dataclasses import dataclass


@dataclass
class GymConfig:
    gym_id = "rocketleague_v1"
    episode_max_steps = 3_000
    train_max_steps = 500_000
    directions = [(0, 0), (1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 1), (-1, -1), (-1, 1)]


@dataclass
class BentjhGymConfig:
    """Config for bentjh's RocketLeagueSideSwipeGym (src/gym_env/bentjh_gym.py)."""

    gym_id: str = "bentjh01_2d_rocketleague_v1"
    max_steps: int = 3_000
    normalize_reward: bool = False
    score_gain: float = 100.0
