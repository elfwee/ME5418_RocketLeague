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

from src.sim.core.actions import CarAction
from src.sim.core.simulation import Simulation


def get_orange_perspective_obs(sim: Simulation) -> list[float]:
    """52-float observation for the orange car, mirrored so it looks like the blue car's view."""
    if sim.orange_car is None:
        raise ValueError("Orange perspective requires an orange car in the simulation.")
    return sim.get_state_norm_mirror(as_flat_array=True)


def mirror_action(action: CarAction) -> CarAction:
    """Map an action chosen in the mirrored (blue-view) world back to the real orange car."""
    return CarAction(dir_x=-action.dir_x, dir_y=action.dir_y, jump=action.jump, boost=action.boost)
