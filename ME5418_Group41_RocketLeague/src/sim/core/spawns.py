"""Kickoff spawn rotation and random spawn sampling.

A spawn is `(x, y, angle, facing_x)`. Kickoffs alternate between a defensive and an attacking
position on each team's half, mirrored across the centre line; random spawns put each car on its
own half of the floor and the ball anywhere in the arena, clear of corners and both cars.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from src.config.sim_config import (
    ARENA_SEGMENT_RADIUS,
    BALL_RADIUS,
    CAR_SPAWN_X_ATTACK,
    CAR_SPAWN_X_DEFENSIVE,
    CORNER_RADIUS,
    RANDOM_SPAWN_BALL_ATTEMPTS,
    RANDOM_SPAWN_BALL_CLEARANCE,
    RANDOM_SPAWN_BLUE_X,
    RANDOM_SPAWN_MIN_CAR_GAP,
    RANDOM_SPAWN_ORANGE_X,
)

if TYPE_CHECKING:
    from src.sim.core.arena import Arena

Spawn = tuple[float, float, float, int]

KICKOFF_X_BLUE = (CAR_SPAWN_X_DEFENSIVE, CAR_SPAWN_X_ATTACK)


def kickoff_spawn(team: str, spawn_index: int, center_x: float, spawn_y: float) -> Spawn:
    """Kickoff spawn of `team` for kickoff round `spawn_index` (rounds alternate defensive / attacking)."""
    x_blue = KICKOFF_X_BLUE[spawn_index % len(KICKOFF_X_BLUE)]
    if team == "orange":
        # Exact symmetrical position on Orange side, facing Left (angle = pi)
        return (2.0 * center_x - x_blue, spawn_y, math.pi, -1)
    return (x_blue, spawn_y, 0.0, 1)


def sample_random_spawns(
    rng: random.Random,
    arena: Arena,
    spawn_y: float,
    ball_spawn_y: float,
    center_x: float,
    with_orange: bool,
) -> tuple[Spawn, Spawn | None, tuple[float, float]]:
    """Sample valid, non-overlapping random spawn positions for both cars and the ball.

    Requirements:
    1. Blue car: on the ground in the left half of the field (x in RANDOM_SPAWN_BLUE_X, facing Right).
    2. Orange car: on the ground in the right half of the field (x in RANDOM_SPAWN_ORANGE_X, facing Left).
       Neither car spawns inside goal posts or clipping arena corner fillets.
    3. Ball: anywhere in the arena, strictly outside goal posts and corner fillets.
    4. Overlap prevention: ball and cars keep RANDOM_SPAWN_BALL_CLEARANCE apart (no physical overlap).

    Returns:
        (blue_spawn, orange_spawn, ball_pos); `orange_spawn` is None without an Orange car.
    """
    # 1. Blue car on the ground in the left half of the field
    bx_car = rng.uniform(*RANDOM_SPAWN_BLUE_X)
    by_car = spawn_y
    blue_spawn = (bx_car, by_car, 0.0, 1)

    # 2. Orange car on the ground in the right half of the field
    orange_spawn = None
    if with_orange:
        ox_car = rng.uniform(*RANDOM_SPAWN_ORANGE_X)
        if ox_car - bx_car < RANDOM_SPAWN_MIN_CAR_GAP:
            mid_x = 0.5 * (bx_car + ox_car)
            half_gap = 0.5 * RANDOM_SPAWN_MIN_CAR_GAP
            bx_car = max(RANDOM_SPAWN_BLUE_X[0], mid_x - half_gap)
            ox_car = min(RANDOM_SPAWN_ORANGE_X[1], mid_x + half_gap)
            blue_spawn = (bx_car, by_car, 0.0, 1)
        orange_spawn = (ox_car, spawn_y, math.pi, -1)

    # 3. Ball within playable arena bounds, avoiding corners, goal pockets, and car overlap
    r_eff = BALL_RADIUS + ARENA_SEGMENT_RADIUS + 0.05
    ball_x_min = arena.x_left + r_eff
    ball_x_max = arena.x_right - r_eff
    ball_y_min = arena.y_floor + r_eff
    ball_y_max = arena.y_ceil - r_eff
    max_corner_dist = CORNER_RADIUS - r_eff
    # Corner fillet arc centres; (sx, sy) is the quadrant of the cut-out corner relative to the centre
    corner_fillets = (
        (arena.x_left + CORNER_RADIUS, arena.y_floor + CORNER_RADIUS, -1, -1),
        (arena.x_left + CORNER_RADIUS, arena.y_ceil - CORNER_RADIUS, -1, 1),
        (arena.x_right - CORNER_RADIUS, arena.y_floor + CORNER_RADIUS, 1, -1),
        (arena.x_right - CORNER_RADIUS, arena.y_ceil - CORNER_RADIUS, 1, 1),
    )

    ball_pos = (center_x, ball_spawn_y + 3.0)  # Fallback when every attempt is rejected
    for _ in range(RANDOM_SPAWN_BALL_ATTEMPTS):
        bx = rng.uniform(ball_x_min, ball_x_max)
        by = rng.uniform(ball_y_min, ball_y_max)

        # Points in a corner's quadrant farther than max_corner_dist from its arc centre lie in the cut-out
        if any(
            (bx < cx_c if sx < 0 else bx > cx_c)
            and (by < cy_c if sy < 0 else by > cy_c)
            and math.hypot(bx - cx_c, by - cy_c) > max_corner_dist
            for cx_c, cy_c, sx, sy in corner_fillets
        ):
            continue
        if math.hypot(bx - bx_car, by - by_car) < RANDOM_SPAWN_BALL_CLEARANCE:
            continue
        if orange_spawn is not None and math.hypot(bx - orange_spawn[0], by - orange_spawn[1]) < (
            RANDOM_SPAWN_BALL_CLEARANCE
        ):
            continue

        ball_pos = (bx, by)
        break

    return blue_spawn, orange_spawn, ball_pos
