"""Gymnasium environment settings: registration, episode length, action directions and rewards.

Read defaults straight off the class (`GymConfig.reward_goal`), or build a variant with
`dataclasses.replace(GymConfig(), reward_goal=10.0)`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GymConfig:
    gym_id: str = "rocketleague_v1"
    episode_max_steps: int = 3_600  # Frames per episode before time-limit truncation (60 s at 60 Hz)

    # Direction inputs (dir_x, dir_y) indexed by the action's direction component:
    # none, right, left, down, up, down-right, up-right, down-left, up-left
    directions: tuple[tuple[int, int], ...] = (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 1),
        (-1, -1),
        (-1, 1),
    )

    # Terminal rewards
    reward_goal: float = 20.0  # Ego team scores
    reward_opponent_own_goal: float = 2.0  # Ego team scores, but the opponent touched the ball last
    reward_concede: float = -20.0

    # Shaping rewards
    ball_progress_deadband: float = 0.0001  # Ignore ball x movement smaller than this (normalised units)
    ground_ball_y: float = -0.55  # Normalised ball height at or below which the ball counts as rolling
    ground_ball_y_factor: float = 0.25
    approach_reward_scale: float = 1.5
    distance_penalty_scale: float = 0.005
    time_penalty: float = 0.001
