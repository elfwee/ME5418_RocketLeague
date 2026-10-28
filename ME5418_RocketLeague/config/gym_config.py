# gym_config
from dataclasses import dataclass


@dataclass
class GymConfig:
    gym_id: str = "bentjh01_2d_rocketleague_v1"
    max_steps: int = 3_000
    normalize_reward: bool = False
    time_penalty: float = -0.000
    action_gain: float = 0.007
    progress_gain: float = 0.01
    score_gain: float = 100.0
    orientation_gain: float = 0.0
    train_bot_type: str = "none"
    train_render_mode: str = "terminal"
    train_max_steps: int = 500_000
    train_model_name: str = "a2c_v2"
    test_bot_type: str = "none"
    test_render_mode: str = "human"
