"""The discrete action encoding: the only module that knows the 9 x 2 x 2 layout.

An env action is either a flat ``Discrete(36)`` index or a factored ``[direction, jump, boost]``
triple (``MultiDiscrete([9, 2, 2])``). The flat index packs the triple as

    index = direction * 4 + jump * 2 + boost

and ``direction`` indexes ``GymConfig.directions`` as (dir_x, dir_y): none, right, left, down, up,
down-right, up-right, down-left, up-left.
"""

from collections.abc import Sequence

import numpy as np

from src.config.gym_config import GymConfig
from src.sim.core.actions import CarAction

DIRECTIONS = GymConfig.directions
NUM_DIRECTIONS = len(DIRECTIONS)
NUM_JUMP = 2
NUM_BOOST = 2
FACTORED_DIMS = (NUM_DIRECTIONS, NUM_JUMP, NUM_BOOST)
NUM_DISCRETE_ACTIONS = NUM_DIRECTIONS * NUM_JUMP * NUM_BOOST

_FACTORED_TYPES = (list, tuple, np.ndarray)


def is_factored(action) -> bool:
    """True for a ``[direction, jump, boost]`` triple, False for a flat index."""
    return isinstance(action, _FACTORED_TYPES)


def encode(direction: int, jump: int, boost: int) -> int:
    """Flat ``Discrete(36)`` index of a (direction, jump, boost) triple."""
    return int(direction) * NUM_JUMP * NUM_BOOST + int(jump) * NUM_BOOST + int(boost)


def split(action) -> tuple[int, int, int]:
    """(direction, jump, boost) of a flat index or a factored triple."""
    if is_factored(action):
        return int(action[0]), int(action[1]), int(action[2])
    index = int(action)
    return index // (NUM_JUMP * NUM_BOOST), (index % (NUM_JUMP * NUM_BOOST)) // NUM_BOOST, index % NUM_BOOST


def decode(action, mirror_x: bool = False) -> CarAction:
    """CarAction for an env action; ``mirror_x`` negates the x input (a policy playing Orange)."""
    direction, jump, boost = split(action)
    dir_x, dir_y = DIRECTIONS[direction]
    if mirror_x:
        dir_x = -dir_x
    return CarAction(dir_x=float(dir_x), dir_y=float(dir_y), jump=jump == 1, boost=boost == 1).clamp()


def clear_jump(action):
    """The same action with its jump bit released, in the same form (flat index or [d, j, b] list).

    Used on the repeated frames of an action-repeat decision so edge-triggered jumping can re-fire.
    """
    if is_factored(action):
        return [int(action[0]), 0, int(action[2])]
    jump_bit = (action % (NUM_JUMP * NUM_BOOST)) // NUM_BOOST
    return (action - NUM_BOOST) if jump_bit == 1 else action


def mirror(action: CarAction) -> CarAction:
    """Reflect a CarAction through the arena's vertical centre line (Blue view <-> Orange view)."""
    return CarAction(dir_x=-action.dir_x, dir_y=action.dir_y, jump=action.jump, boost=action.boost)


def describe(action) -> str:
    """Human-readable form for the demos, e.g. ``dir=(1, 0), jump=True, boost=False``."""
    direction, jump, boost = split(action)
    return f"dir={DIRECTIONS[direction]}, jump={bool(jump)}, boost={bool(boost)}"


def as_list(action: Sequence[int]) -> list[int]:
    """A factored action as a plain ``[direction, jump, boost]`` list of ints."""
    return [int(a) for a in action]
