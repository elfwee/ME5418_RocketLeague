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

from typing import Any, Dict, List

from src.sim.core.actions import CarAction
from src.sim.core.simulation import Simulation

NUM_RAYS = 8


def _mx(v) -> List[float]:
    """Mirror a 2D (x, y) position / velocity / direction about the vertical centre line."""
    return [-v[0], v[1]]


def _mirror_rays(distances: List[float]) -> List[float]:
    """Body-relative ray i of the mirrored car is ray (8 - i) % 8 of the real car."""
    return [distances[(NUM_RAYS - i) % NUM_RAYS] for i in range(NUM_RAYS)]


def _mirror_relation(rel: Dict[str, Any]) -> List[float]:
    return [rel["magnitude"], *_mx(rel["direction"])]


def _mirror_car_core(car: Dict[str, Any]) -> List[float]:
    """First 12 car features shared by the ego and opponent blocks, mirrored."""
    sin_t, cos_t = car["orientation"]
    return [
        *_mx(car["position"]),
        *_mx(car["velocity"]),
        sin_t,
        -cos_t,
        -car["angular_velocity"],
        car["boost"],
        car["wheel_contacts"],
        car["has_jump2"],
        car["is_flipping"],
        car["is_turtling"],
    ]


def get_orange_perspective_obs(sim: Simulation) -> List[float]:
    """52-float observation for the orange car, mirrored so it looks like the blue car's view."""
    if sim.orange_car is None:
        raise ValueError("Orange perspective requires an orange car in the simulation.")

    s = sim.get_state_norm(as_flat_array=False)
    ego = s["car_orange"]
    opp = s["blue_car"]  # blue car is always present, so it is always the opponent
    ball = s["ball"]
    rel = s["relations"]

    obs = [
        # 1. Ego car (21): orange car mirrored
        *_mirror_car_core(ego),
        ego["is_boost_recovery"],
        *_mirror_rays(ego["boundary_distances"]),
        # 2. Ball (5)
        *_mx(ball["position"]),
        *_mx(ball["velocity"]),
        -ball["angular_velocity"],
        # 3. Relations (9): ego->ball, ball->opponent goal, ball->own goal
        *_mirror_relation(rel["orange_car_to_ball"]),
        *_mirror_relation(rel["ball_to_blue_goal"]),
        *_mirror_relation(rel["ball_to_orange_goal"]),
        # 4. Opponent present flag (1)
        1.0,
        # 5. Opponent (16): blue car mirrored + blue_car_to_ball
        *_mirror_car_core(opp),
        opp["is_boosting"],
        *_mirror_relation(rel["blue_car_to_ball"]),
    ]
    return obs


def mirror_action(action: CarAction) -> CarAction:
    """Map an action chosen in the mirrored (blue-view) world back to the real orange car."""
    return CarAction(dir_x=-action.dir_x, dir_y=action.dir_y, jump=action.jump, boost=action.boost)
