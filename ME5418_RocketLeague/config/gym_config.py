# gym_config
from dataclasses import dataclass


@dataclass
class GymConfig:
    gym_id: str = "bentjh01_2d_rocketleague_v1"
    max_steps: int = 1_000
    normalize_reward: bool = False
    time_penalty: float = -0.001
    action_gain: float = 0.001
    progress_gain: float = 0.1
    score_gain: float = 10.0
    orientation_gain: float = 0.0
    train_bot_type: str = "none"
    train_render_mode: str = "terminal"
    train_max_steps: int = 500_000
    train_model_name: str = "a2c_v13"
    test_bot_type: str = "none"
    test_render_mode: str = "human"
