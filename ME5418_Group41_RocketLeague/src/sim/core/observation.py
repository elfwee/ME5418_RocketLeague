"""The 52-float observation, built straight from the sim's entities for either car.

Layout (every value normalised to [-1, 1], or [0, 1] for distances and fractions):

    index   feature
    0-1     ego car position (x, y), centred on the arena
    2-3     ego car velocity / CAR_MAX_SPEED
    4-5     ego car orientation (sin theta, cos theta)
    6       ego car angular velocity / CAR_MAX_ANGULAR_SPEED
    7       ego boost / CAR_MAX_BOOST
    8       ego wheel contacts / 2
    9-12    ego flags: second jump available, flipping, turtling, boost recovery
    13-20   8 body-relative boundary raycasts / ARENA_DIAGONAL (0, 45, ..., 315 degrees)
    21-22   ball position
    23-24   ball velocity / BALL_MAX_SPEED
    25      ball angular velocity / BALL_MAX_SPIN
    26-28   ego car -> ball           (distance / ARENA_DIAGONAL, unit direction x, y)
    29-31   ball -> opponent's goal   (same)
    32-34   ball -> own goal          (same)
    35      opponent present (1.0 / 0.0)
    36-48   opponent car: as 0-12, with "is boosting" in place of "boost recovery"
    49-51   opponent car -> ball
    (36-51 are zero when there is no opponent)

The Orange car sees the arena mirrored through its vertical centre line, so a policy trained on
Blue can drive Orange: build the observation with Orange as the ego car, then `mirror_observation`
negates every x-like feature and reverses the cyclic ray order (ray i <-> ray (8 - i) % 8).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.config.sim_config import (
    ARENA_DIAGONAL,
    BALL_MAX_SPEED,
    BALL_MAX_SPIN,
    CAR_MAX_ANGULAR_SPEED,
    CAR_MAX_BOOST,
    CAR_MAX_SPEED,
    CENTER_X,
    CENTER_Y,
    HALF_HEIGHT,
    HALF_WIDTH,
)

if TYPE_CHECKING:
    from src.sim.core.car import Car
    from src.sim.core.simulation import Simulation

OBS_DIM = 52
NUM_RAYS = 8

# Index layout
EGO_CAR = slice(0, 13)
EGO_RAYS = slice(13, 21)
BALL = slice(21, 26)
EGO_TO_BALL = slice(26, 29)
BALL_TO_OPPONENT_GOAL = slice(29, 32)
BALL_TO_OWN_GOAL = slice(32, 35)
OPPONENT_PRESENT = 35
OPPONENT_CAR = slice(36, 49)
OPPONENT_TO_BALL = slice(49, 52)
OPPONENT_FEATURES = slice(36, 52)

# Car block offsets (within EGO_CAR / OPPONENT_CAR)
_CAR_POS_X, _CAR_VEL_X, _CAR_COS, _CAR_SPIN = 0, 2, 5, 6

# Features whose sign flips under the left-right mirror: x positions / velocities / directions,
# cos(theta) (theta -> pi - theta) and angular velocities (rotation sense flips).
MIRROR_SIGN_INDICES = tuple(
    sorted(
        [EGO_CAR.start + i for i in (_CAR_POS_X, _CAR_VEL_X, _CAR_COS, _CAR_SPIN)]
        + [BALL.start, BALL.start + 2, BALL.start + 4]
        + [EGO_TO_BALL.start + 1, BALL_TO_OPPONENT_GOAL.start + 1, BALL_TO_OWN_GOAL.start + 1]
        + [OPPONENT_CAR.start + i for i in (_CAR_POS_X, _CAR_VEL_X, _CAR_COS, _CAR_SPIN)]
        + [OPPONENT_TO_BALL.start + 1]
    )
)
# Ray i of the mirrored car points where ray (8 - i) % 8 of the real car does.
MIRROR_RAY_ORDER = tuple((NUM_RAYS - i) % NUM_RAYS for i in range(NUM_RAYS))


def _clamp(val: float, low: float, high: float) -> float:
    return low if val < low else (high if val > high else val)


def normalize_position(p: tuple[float, float]) -> tuple[float, float]:
    """Map world coordinates (x, y) symmetrically around the arena centre to [-1, 1]."""
    nx = _clamp((p[0] - CENTER_X) / HALF_WIDTH, -1.0, 1.0)
    ny = _clamp((p[1] - CENTER_Y) / HALF_HEIGHT, -1.0, 1.0)
    return (nx, ny)


def vector_relation(p_from: tuple[float, float], p_to: tuple[float, float]) -> tuple[float, tuple[float, float]]:
    """(Euclidean distance, unit direction) from `p_from` to `p_to`; the direction is (0, 0) when they coincide."""
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    magnitude = math.hypot(dx, dy)
    if magnitude > 1e-9:
        return magnitude, (dx / magnitude, dy / magnitude)
    return magnitude, (0.0, 0.0)


def _relation_features(p_from: tuple[float, float], p_to: tuple[float, float]) -> list[float]:
    magnitude, (dir_x, dir_y) = vector_relation(p_from, p_to)
    return [min(1.0, magnitude / ARENA_DIAGONAL), dir_x, dir_y]


def _car_features(car: Car, last_flag: bool) -> list[float]:
    """The 13 per-car features; `last_flag` is boost recovery for the ego car, is-boosting for the opponent."""
    cvx, cvy = car.velocity
    angle = car.angle
    return [
        *normalize_position(car.position),
        _clamp(cvx / CAR_MAX_SPEED, -1.0, 1.0),
        _clamp(cvy / CAR_MAX_SPEED, -1.0, 1.0),
        math.sin(angle),
        math.cos(angle),
        _clamp(car.body.angular_velocity / CAR_MAX_ANGULAR_SPEED, -1.0, 1.0),
        _clamp(car.boost_amount / CAR_MAX_BOOST, 0.0, 1.0),
        car.wheel_contact_count / 2.0,
        1.0 if car.has_jump2 else 0.0,
        1.0 if car.is_flipping else 0.0,
        1.0 if car.is_turtling else 0.0,
        1.0 if last_flag else 0.0,
    ]


def _ray_features(car: Car) -> list[float]:
    return [min(1.0, max(0.0, ray["distance"] / ARENA_DIAGONAL)) for ray in car.compute_boundary_raycasts()]


def _ball_features(sim: Simulation) -> list[float]:
    bvx, bvy = sim.ball.velocity
    return [
        *normalize_position(sim.ball.position),
        _clamp(bvx / BALL_MAX_SPEED, -1.0, 1.0),
        _clamp(bvy / BALL_MAX_SPEED, -1.0, 1.0),
        _clamp(sim.ball.body.angular_velocity / BALL_MAX_SPIN, -1.0, 1.0),
    ]


def mirror_observation(obs: list[float]) -> list[float]:
    """Reflect an observation through the arena's vertical centre line (an involution)."""
    out = list(obs)
    for i in MIRROR_SIGN_INDICES:
        out[i] = -out[i]
    rays = obs[EGO_RAYS]
    out[EGO_RAYS] = [rays[j] for j in MIRROR_RAY_ORDER]
    return out


def build_observation(sim: Simulation, ego: str = "blue") -> list[float]:
    """The 52-float observation from `ego`'s ("blue" or "orange") point of view.

    Orange's view is mirrored so that, like Blue, it attacks towards +x. Raises ValueError for an
    Orange view when the sim has no Orange car.
    """
    if ego == "orange":
        if sim.orange_car is None:
            raise ValueError("An Orange-perspective observation needs an Orange car in the simulation.")
        ego_car, opponent, own_goal, opponent_goal = sim.orange_car, sim.blue_car, "orange", "blue"
    else:
        ego_car, opponent, own_goal, opponent_goal = sim.blue_car, sim.orange_car, "blue", "orange"

    ball_pos = sim.ball.position
    obs = [
        *_car_features(ego_car, ego_car.is_boost_recovery),
        *_ray_features(ego_car),
        *_ball_features(sim),
        *_relation_features(ego_car.position, ball_pos),
        *_relation_features(ball_pos, sim.get_goal_center(opponent_goal)),
        *_relation_features(ball_pos, sim.get_goal_center(own_goal)),
    ]
    has_opponent = opponent is not None and (ego == "orange" or sim.enable_orange)
    if has_opponent:
        obs.append(1.0)
        obs.extend(_car_features(opponent, opponent.is_boosting))
        obs.extend(_relation_features(opponent.position, ball_pos))
    else:
        obs.append(0.0)
        obs.extend([0.0] * (OPPONENT_FEATURES.stop - OPPONENT_FEATURES.start))

    return mirror_observation(obs) if ego == "orange" else obs
