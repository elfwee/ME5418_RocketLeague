# gym_config
from dataclasses import dataclass

@dataclass
class GymConfig():
    gym_id:str              = "bentjh01_2d_rocketleague_v1"
    max_steps:int           = 10000
    action_gain:float       = 1.0
    progress_gain:float     = 5.0
    score_gain:float        = 10.0
    train_bot_type:str      = "none"
    train_render_mode:str   = "terminal"
    test_bot_type:str       = "none"
    test_render_mode:str    = "human"
