"""
Orange Perspective: reuse a policy trained as the blue car to drive the orange car.

The arena is left-right symmetric about CENTER_X, so the orange car's situation is the
blue car's situation reflected through the vertical centre line (x -> 2*CENTER_X - x).
We build the 52-float observation the blue-trained policy *would* see if it were the
orange car in that mirrored world, then mirror the chosen action back.

Mirror transform (normalized coordinates are already centred, so x -> -x):
    - positions, velocities, direction vectors: negate the x component
    - orientation (sin theta, cos theta): body angle theta -> pi - theta, so (sin, -cos)
    - angular velocity: negated (rotation sense flips under reflection)
    - 8 body-relative raycasts at theta + i*45 deg: ray i <-> ray (8 - i) % 8
    - goals swap: orange's opponent goal is the blue goal, and vice versa
    - action: dir_x -> -dir_x (jump / boost unchanged)

Layout must match Simulation.get_state_norm(as_flat_array=True) exactly.
"""

import math
from types import SimpleNamespace
from typing import List

from src.config.sim_config import (
    BALL_MAX_SPEED,
    CAR_MAX_BOOST,
    CAR_MAX_SPEED,
    CENTER_X,
    CENTER_Y,
    HALF_HEIGHT,
    HALF_WIDTH,
)
from src.sim.ai.heuristic_bot import HeuristicBot
from src.sim.core.actions import CarAction
from src.sim.core.simulation import Simulation


def get_orange_perspective_obs(sim: Simulation) -> List[float]:
    """52-float observation for the orange car, mirrored so it looks like the blue car's view."""
    if sim.orange_car is None:
        raise ValueError("Orange perspective requires an orange car in the simulation.")
    return sim.get_state_norm_mirror(as_flat_array=True)


def mirror_action(action: CarAction) -> CarAction:
    """Map an action chosen in the mirrored (blue-view) world back to the real orange car."""
    return CarAction(dir_x=-action.dir_x, dir_y=action.dir_y, jump=action.jump, boost=action.boost)


def _world_pos(x: float, y: float) -> tuple:
    """Undo Simulation's position normalization."""
    return (CENTER_X + x * HALF_WIDTH, CENTER_Y + y * HALF_HEIGHT)


def _mirrored_sim_view(sim: Simulation, obs: List[float]) -> SimpleNamespace:
    """Decode a 52-float observation into the sim attributes HeuristicBot reads for its own car and
    the ball. Arena geometry is left-right symmetric and static, so it comes straight from `sim`.
    is_grounded is approximate: the observation has no chassis contact flag, so a car touching the
    ground only by its chassis while boosting reads as airborne."""
    sin_t, cos_t = obs[4], obs[5]
    wheels = round(obs[8] * 2)
    both_wheels = wheels == 2
    pitch = abs(math.degrees(math.atan2(sin_t, abs(cos_t))))
    car = SimpleNamespace(
        position=_world_pos(obs[0], obs[1]),
        velocity=(obs[2] * CAR_MAX_SPEED, obs[3] * CAR_MAX_SPEED),
        boost_amount=obs[7] * CAR_MAX_BOOST,
        wheel_contact_count=wheels,
        both_wheels_grounded=both_wheels,
        can_ground_jump=both_wheels or (wheels >= 1 and pitch < 75.0),
        has_jump2=obs[9] > 0.5,
        is_grounded=wheels > 0 or obs[12] > 0.5,
    )
    ball = SimpleNamespace(
        position=_world_pos(obs[21], obs[22]),
        velocity=(obs[23] * BALL_MAX_SPEED, obs[24] * BALL_MAX_SPEED),
    )
    return SimpleNamespace(
        blue_car=car,
        ball=ball,
        arena=sim.arena,
        center_x=sim.center_x,
        ball_spawn_y=sim.ball_spawn_y,
        time_elapsed=sim.time_elapsed,
    )


class OrangePerspectiveBot:
    """Drive the orange car with a blue HeuristicBot that only sees the mirrored observation."""

    def __init__(self, sim: Simulation):
        self.sim = sim
        self.bot = HeuristicBot(team="blue")

    def get_action(self) -> CarAction:
        if self.sim.orange_car is None:  # orange toggled off in-game
            return CarAction()
        view = _mirrored_sim_view(self.sim, get_orange_perspective_obs(self.sim))
        return mirror_action(self.bot.compute_action(view))

    def reset(self):
        self.bot.reset()
