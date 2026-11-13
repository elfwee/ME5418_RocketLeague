"""Headless simulation manager orchestrating Pymunk space, entities, and collisions."""

import math
import random
from typing import Any

import pymunk

from src.config.sim_config import (
    ARENA_SEGMENT_RADIUS,
    BALL_RADIUS,
    CAR_BALL_RESTITUTION,
    CAR_CAR_RESTITUTION,
    CAR_RIDE_HEIGHT,
    COLLISION_BALL,
    COLLISION_CAR_BODY,
    COLLISION_GOAL_SENSOR,
    FIELD_HEIGHT,
    FIELD_WIDTH,
    GRAVITY,
    MARGIN_X,
    MARGIN_Y,
    PHYSICS_SUBSTEPS,
    SIM_HZ,
    SOLVER_ITERATIONS,
)
from src.sim.core.actions import CarAction
from src.sim.core.arena import Arena
from src.sim.core.ball import Ball
from src.sim.core.car import Car
from src.sim.core.observation import build_observation, vector_relation
from src.sim.core.spawns import Spawn, kickoff_spawn, sample_random_spawns


class Simulation:
    """Headless 2D physics simulation environment for Rocket League."""

    def __init__(self, enable_orange: bool = True, random_spawn: bool = False, seed: int | None = None):
        # Random spawns draw from this generator only (seed it for repeatable spawn sequences)
        self.rng = random.Random(seed)
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        # No global damping: drag belongs to the entities that model it, otherwise it
        # silently bleeds energy out of every body including the player's blue_car.
        self.space.damping = 1.0
        self.space.iterations = SOLVER_ITERATIONS

        # Score & Event tracking
        self.score_blue: int = 0
        self.score_orange: int = 0
        self.last_goal_team: str | None = None
        self.goal_scored_this_step: str | None = None
        self.last_touch: str | None = None
        self._spawn_index: int = 0
        self._orange_solver_first: bool = False
        self.time_elapsed: float = 0.0
        self.match_time: float = 0.0
        self.random_spawn: bool = random_spawn

        # Center coordinates
        self.center_x = MARGIN_X + FIELD_WIDTH / 2.0
        self.center_y = MARGIN_Y + FIELD_HEIGHT / 2.0

        # Instantiate entities
        self.arena = Arena(self.space)
        # 1. Ball spawns resting on the ground at kickoff
        self.ball = Ball(self.space, x=self.center_x, y=self.ball_spawn_y)
        # 2. Blue blue_car spawns on its side at initial kickoff position
        init_x, init_y, init_ang, init_fac = self.get_spawn_position("blue")
        self.blue_car = Car(self.space, x=init_x, y=init_y, angle=init_ang, team="blue")
        self.blue_car.reset(init_x, init_y, angle=init_ang, facing_x=init_fac)

        # 3. Orange blue_car and AI Bot (optional, toggleable)
        self.enable_orange: bool = enable_orange
        self.orange_car: Car | None = None

        if self.enable_orange:
            ox, oy, oang, ofac = self.get_spawn_position("orange")
            self.orange_car = Car(self.space, x=ox, y=oy, angle=oang, team="orange")
            self.orange_car.reset(ox, oy, angle=oang, facing_x=ofac)

        self._spawn_index += 1
        self._setup_collision_handlers()
        if self.random_spawn:
            self.reset(reset_scores=True, random_spawn=True)

    def set_orange_enabled(self, enabled: bool):
        """Enable or disable the Orange car dynamically."""
        if enabled == self.enable_orange:
            return

        self.enable_orange = enabled

        if enabled:
            if self.orange_car is None:
                ox, oy, oang, ofac = self.get_spawn_position("orange")
                self.orange_car = Car(self.space, x=ox, y=oy, angle=oang, team="orange")
                self.orange_car.reset(ox, oy, angle=oang, facing_x=ofac)
        else:
            if self.orange_car is not None:
                self._set_orange_solver_first(False)
                self.space.remove(self.orange_car.chassis_shape, self.orange_car.body)
                self.orange_car = None

    @property
    def spawn_y(self) -> float:
        """Floor height that puts the blue_car exactly at its settled ride height."""
        return MARGIN_Y + ARENA_SEGMENT_RADIUS + CAR_RIDE_HEIGHT

    @property
    def ball_spawn_y(self) -> float:
        """Floor elevation that places the ball resting on the arena floor."""
        return MARGIN_Y + ARENA_SEGMENT_RADIUS + BALL_RADIUS

    def get_spawn_position(self, team: str | None = None, spawn_index: int | None = None) -> Spawn:
        """Get the kickoff spawn (x, y, angle, facing_x) on the team's side for the kickoff round."""
        team = team or self.blue_car.team
        idx = self._spawn_index if spawn_index is None else spawn_index
        return kickoff_spawn(team, idx, self.center_x, self.spawn_y)

    def sample_random_spawns(self) -> tuple[Spawn, Spawn | None, tuple[float, float]]:
        """Random (blue_spawn, orange_spawn, ball_pos) drawn from `self.rng` (see `spawns.sample_random_spawns`)."""
        return sample_random_spawns(
            self.rng, self.arena, self.spawn_y, self.ball_spawn_y, self.center_x, with_orange=self.enable_orange
        )

    def _setup_collision_handlers(self):
        """Configure contact listeners for blue_car/ball strikes and goal detection.

        Supports both the legacy handler-object API (pymunk < 7) and the
        functional `on_collision` API (pymunk >= 7, which removed
        `add_collision_handler`).
        """

        def _car_ball_pre_solve(arbiter, space, data=None):
            arbiter.restitution = CAR_BALL_RESTITUTION
            shape_a, shape_b = arbiter.shapes
            car_shape = shape_a if shape_a.collision_type == COLLISION_CAR_BODY else shape_b
            car_obj = getattr(car_shape, "car", None)
            if car_obj is not None:
                car_x = car_obj.position[0]
                # Only update last touch if the blue_car is on the playable field, not inside either goal
                if self.arena.x_left <= car_x <= self.arena.x_right:
                    self.last_touch = car_obj.team
            return True

        def _car_car_pre_solve(arbiter, space, data=None):
            arbiter.restitution = CAR_CAR_RESTITUTION
            return True

        def _ball_goal_begin(arbiter, space, data=None):
            return False

        if hasattr(self.space, "on_collision"):
            # pymunk >= 7
            self.space.on_collision(COLLISION_CAR_BODY, COLLISION_BALL, pre_solve=_car_ball_pre_solve)
            self.space.on_collision(COLLISION_CAR_BODY, COLLISION_CAR_BODY, pre_solve=_car_car_pre_solve)
            self.space.on_collision(COLLISION_BALL, COLLISION_GOAL_SENSOR, begin=_ball_goal_begin)
        else:
            # pymunk < 7
            h_car_ball = self.space.add_collision_handler(COLLISION_CAR_BODY, COLLISION_BALL)
            h_car_ball.pre_solve = _car_ball_pre_solve

            h_car_car = self.space.add_collision_handler(COLLISION_CAR_BODY, COLLISION_CAR_BODY)
            h_car_car.pre_solve = _car_car_pre_solve

            h_ball_goal = self.space.add_collision_handler(COLLISION_BALL, COLLISION_GOAL_SENSOR)
            h_ball_goal.begin = _ball_goal_begin

    def _check_goal(self) -> str | None:
        """Check if the ball is 100% inside either goal pocket.

        Returns:
            'orange': ball is 100% inside Left Goal pocket (Orange scores)
            'blue': ball is 100% inside Right Goal pocket (Blue scores)
            None: ball has not completely crossed either goal line
        """
        bx, by = self.ball.position
        r = self.ball.radius
        g_bot = self.arena.goal_y_bot
        g_top = self.arena.goal_y_top

        # Vertical check: ball must be within the vertical goal opening
        if (by - r < g_bot - 0.1) or (by + r > g_top + 0.1):
            return None

        # Left Goal Pocket (Orange scores):
        # 100% inside when the ball's rightmost point has crossed the left goal line
        if bx + r <= self.arena.x_left:
            return "orange"

        # Right Goal Pocket (Blue scores):
        # 100% inside when the ball's leftmost point has crossed the right goal line
        if bx - r >= self.arena.x_right:
            return "blue"

        return None

    def step(self, action: CarAction, dt: float = 1.0 / SIM_HZ, action_orange: CarAction | None = None):
        """Advance the physics simulation by dt using sub-stepping for stability (Orange idles when no action is given)."""
        if action_orange is None:
            action_orange = CarAction()
        if self.goal_scored_this_step is not None:
            # Previous frame was a goal: reset touch tracking for the new kickoff round
            self.last_touch = None

        self.goal_scored_this_step = None

        sub_dt = dt / PHYSICS_SUBSTEPS
        for _ in range(PHYSICS_SUBSTEPS):
            if self._orange_solver_first and self.orange_car is not None:
                self.orange_car.update(action_orange, sub_dt)
                self.blue_car.update(action, sub_dt)
            else:
                self.blue_car.update(action, sub_dt)
                if self.orange_car is not None:
                    self.orange_car.update(action_orange, sub_dt)
            self.ball.apply_aerodynamics(sub_dt)
            self.space.step(sub_dt)

        self.ball.record_trail()
        self.time_elapsed += dt
        self.match_time += dt

        # Clear boundary raycast cache for the new frame
        self.blue_car.clear_raycast_cache()
        if self.orange_car is not None:
            self.orange_car.clear_raycast_cache()

        # Score logic: goal only counts when ball is 100% inside the goal pocket
        scoring_team = self._check_goal()
        if scoring_team is not None:
            if scoring_team == "blue":
                self.score_blue += 1
            elif scoring_team == "orange":
                self.score_orange += 1
            self.last_goal_team = scoring_team
            self.goal_scored_this_step = scoring_team
            # Reset kickoff after goal
            self.reset(reset_scores=False)

    def _set_orange_solver_first(self, orange_first: bool):
        """Set whether orange_car or blue_car holds the primary Pymunk solver/update order."""
        self._orange_solver_first = orange_first
        if self.orange_car is None:
            return
        currently_orange_first = self.orange_car._filter_group < self.blue_car._filter_group
        if currently_orange_first != orange_first:
            c1, c2 = self.blue_car, self.orange_car
            c1.body, c2.body = c2.body, c1.body
            c1.chassis_shape, c2.chassis_shape = c2.chassis_shape, c1.chassis_shape
            c1._filter_group, c2._filter_group = c2._filter_group, c1._filter_group
            c1._query_filter, c2._query_filter = c2._query_filter, c1._query_filter
            c1.facing_x, c2.facing_x = c2.facing_x, c1.facing_x
            c1.chassis_shape.car = c1
            c2.chassis_shape.car = c2

    def reset(
        self,
        reset_scores: bool = False,
        spawn_pos: Spawn | None = None,
        spawn_index: int | None = None,
        random_spawn: bool | None = None,
        clear_events: bool = False,
    ):
        """Reset the ball and cars for a new kickoff round (fixed kickoff or random spawns).

        Args:
            reset_scores: Also zero the scores, match clock and kickoff rotation.
            spawn_pos: Explicit Blue spawn (Orange gets the mirror image); implies a fixed kickoff.
            spawn_index: Kickoff round to play (picks the kickoff position and the solver order).
            random_spawn: Override the sim's `random_spawn` setting for this reset.
            clear_events: Forget the last goal-scored event and the last touch (a new episode).
        """
        if clear_events:
            self.goal_scored_this_step = None
            self.last_touch = None
        if reset_scores:
            self.score_blue = 0
            self.score_orange = 0
            self.last_goal_team = None
            self.goal_scored_this_step = None
            self.last_touch = None
            self.match_time = 0.0
            self._spawn_index = 0

        if spawn_index is not None:
            self._spawn_index = spawn_index

        use_random = self.random_spawn if random_spawn is None else random_spawn
        if spawn_pos is not None:
            use_random = False

        # Rotate Pymunk solver priority over a 4-kickoff cycle so both teams share
        # equal solver priority across both defensive and attacking kickoff spawns
        self._set_orange_solver_first(((self._spawn_index // 2) % 2) == 1)

        if use_random:
            blue_spawn, orange_spawn, ball_pos = self.sample_random_spawns()
            # 1. Ball spawns randomly within whole field
            self.ball.reset(ball_pos[0], ball_pos[1], vx=0.0, vy=0.0, angular_velocity=0.0)

            # 2. Blue blue_car spawns randomly on ground in left half of field
            self.blue_car.reset(blue_spawn[0], blue_spawn[1], angle=blue_spawn[2], facing_x=blue_spawn[3])

            # 3. Orange blue_car spawns randomly on ground in right half of field
            if self.orange_car is not None and orange_spawn is not None:
                self.orange_car.reset(orange_spawn[0], orange_spawn[1], angle=orange_spawn[2], facing_x=orange_spawn[3])
        else:
            # 1. Ball spawns resting on the ground at center
            self.ball.reset(self.center_x, self.ball_spawn_y)

            # 2. Blue blue_car spawns at kickoff position for current round
            if spawn_pos is None:
                x, y, angle, facing = self.get_spawn_position("blue")
            else:
                x, y, angle, facing = spawn_pos

            self.blue_car.reset(x, y, angle=angle, facing_x=facing)

            # 3. Orange blue_car spawns at kickoff position
            if self.orange_car is not None:
                if spawn_pos is None:
                    ox, oy, oang, ofacing = self.get_spawn_position("orange")
                else:
                    ox = 2.0 * self.center_x - x
                    oy = y
                    oang = math.pi
                    ofacing = -1
                self.orange_car.reset(ox, oy, angle=oang, facing_x=ofacing)

        # Advance kickoff spawn index for the next kickoff round
        self._spawn_index += 1

        self.time_elapsed = 0.0

    def get_goal_center(self, team: str) -> tuple[float, float]:
        """Get (x, y) coordinates of the center of a team's goal opening, shifted into the pocket by the ball diameter."""
        ball_diameter = 2.0 * self.ball.radius
        gy = self.arena.goal_y_center
        if team == "blue":
            return (self.arena.x_left - ball_diameter, gy)
        return (self.arena.x_right + ball_diameter, gy)

    def _extract_car_state(self, car: Car) -> dict[str, Any]:
        """Extract un-normalized physical state dictionary for a vehicle."""
        raycasts = car.compute_boundary_raycasts()
        car_dict = {
            "position": car.position,
            "nose_position": car.nose_position,
            "velocity": car.velocity,
            "angle": car.angle,
            "angular_velocity": car.body.angular_velocity,
            "is_grounded": car.is_grounded,
            "ground_normal": car.ground_normal,
            "wheel_contacts": car.wheel_contact_count,
            "both_wheels_grounded": car.both_wheels_grounded,
            "has_jump2": car.has_jump2,
            "is_flipping": car.is_flipping,
            "is_turtling": car.is_turtling,
            "is_turtling_recovery": car.is_turtling_recovery,
            "boost": car.boost_amount,
            "is_boosting": car.is_boosting,
            "is_boost_recovery": car.is_boost_recovery,
            "input_vector": car.last_input_vector,
            "facing_x": car.facing_x,
            "boundary_raycasts": raycasts,
            "boundary_distances": [r["distance"] for r in raycasts],
        }
        return car_dict

    def _compute_relations(self) -> dict[str, Any]:
        """Distance and unit direction between the cars, the ball and both goals (world units)."""

        def relation(p_from, p_to) -> dict[str, Any]:
            magnitude, direction = vector_relation(p_from, p_to)
            return {"magnitude": magnitude, "direction": direction}

        return {
            "blue_car_to_ball": relation(self.blue_car.position, self.ball.position),
            "orange_car_to_ball": (
                relation(self.orange_car.position, self.ball.position) if self.orange_car is not None else None
            ),
            "ball_to_orange_goal": relation(self.ball.position, self.get_goal_center("orange")),
            "ball_to_blue_goal": relation(self.ball.position, self.get_goal_center("blue")),
        }

    def get_state(self) -> dict[str, Any]:
        """Query complete simulation state for headless evaluation or RL."""
        state = {
            "time": self.time_elapsed,
            "match_time": self.match_time,
            "goal_scored_step": self.goal_scored_this_step,
            "ball": {
                "position": self.ball.position,
                "velocity": self.ball.velocity,
                "angular_velocity": self.ball.body.angular_velocity,
                "angle": self.ball.angle,
            },
            "blue_car": self._extract_car_state(self.blue_car),
            "relations": self._compute_relations(),
            "score": {"blue": self.score_blue, "orange": self.score_orange},
            "last_goal": self.last_goal_team,
            "last_touch": self.last_touch,
            "orange_enabled": self.enable_orange,
        }

        if self.orange_car is not None:
            state["car_orange"] = self._extract_car_state(self.orange_car)

        return state

    def get_state_norm(self, as_flat_array: bool = True) -> list[float]:
        """52-float observation from Blue's point of view (see `src/sim/core/observation.py`)."""
        _require_flat(as_flat_array)
        return build_observation(self, "blue")

    def get_state_norm_mirror(self, as_flat_array: bool = True) -> list[float]:
        """52-float observation from Orange's point of view, mirrored to look like Blue's."""
        _require_flat(as_flat_array)
        return build_observation(self, "orange")


def _require_flat(as_flat_array: bool):
    if not as_flat_array:
        raise ValueError(
            "The nested normalised-state dict was removed; use the flat observation, or get_state() for raw values."
        )
