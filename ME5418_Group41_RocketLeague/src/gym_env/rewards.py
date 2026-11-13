"""Reward terms of the Rocket League env, as pure functions of the sim state.

Teams are given relative to the learning car: "ego" (the car the policy drives), "opponent", or
None. The env keeps the only state the shaping needs (last step's ball x and car-ball distance).

Per step the ego car receives:
- on a goal: +reward_goal when it scores (+reward_opponent_own_goal if the opponent touched the
  ball last), reward_concede when it concedes; the episode ends;
- otherwise: ball progress towards the opponent goal (credited only to the car that caused it)
  + potential-based approach shaping - a distance penalty - a time penalty.
"""

import math

import numpy as np

from src.config.gym_config import GymConfig
from src.config.sim_config import ARENA_DIAGONAL


def relative_team(team: str | None, ego_team: str) -> str | None:
    """Map a sim team name to "ego" / "opponent" (None stays None)."""
    if team is None:
        return None
    return "ego" if team == ego_team else "opponent"


def goal_reward(goal: str, last_touch: str | None, cfg: GymConfig) -> float:
    """Terminal reward for a goal by `goal` ("ego" or "opponent")."""
    if goal == "ego":
        return cfg.reward_opponent_own_goal if last_touch == "opponent" else cfg.reward_goal
    return cfg.reward_concede


def ball_progress_reward(delta_x: float, ball_x: float, ball_y: float, last_touch: str | None, cfg: GymConfig) -> float:
    """Reward ball movement `delta_x` towards the opponent goal (+x in the ego frame).

    `ball_x`, `ball_y` are the ball's normalised position in the ego frame after the move.
    """
    if abs(delta_x) <= cfg.ball_progress_deadband:
        return 0.0

    # Non-linear weighting: x_factor exponentially amplifies rewards near the goal mouth (|ball_x| -> 1);
    # y_factor keeps a baseline for ground-rolling balls (ball_y <= ground_ball_y) while rewarding elevated ones.
    x_factor = np.exp(1 / (1.2 - ball_x**2))
    y_factor = (1 - abs(ball_y)) ** 2 if ball_y > cfg.ground_ball_y else cfg.ground_ball_y_factor
    progress = delta_x * x_factor * y_factor

    # Only credit what the last toucher caused: no penalty for the ego car when its own touch
    # sends the ball backwards, and no reward when the opponent's touch sends it forwards.
    if (progress <= 0.0 and last_touch == "ego") or (progress >= 0.0 and last_touch == "opponent"):
        return 0.0
    return progress


def car_ball_distance(car_pos: tuple[float, float], ball_pos: tuple[float, float]) -> float:
    """Car-to-ball distance as a fraction of the arena diagonal, capped at 1."""
    return min(1.0, math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1]) / ARENA_DIAGONAL)


def shaping_reward(progress: float, dist: float, prev_dist: float, cfg: GymConfig) -> float:
    """Non-terminal step reward: ball progress + approach shaping - distance penalty - time penalty."""
    reward = 0.0
    reward += progress
    reward += (prev_dist - dist) * cfg.approach_reward_scale  # potential-based: rewards closing in on the ball
    reward -= dist * cfg.distance_penalty_scale
    reward -= cfg.time_penalty
    return reward
