"""Headless simulation manager orchestrating Pymunk space, entities, and collisions."""
from typing import Dict, Any, Optional, Tuple
import pymunk
from src.config import (
    FIELD_WIDTH, FIELD_HEIGHT, MARGIN_X, MARGIN_Y,
    GRAVITY, SIM_HZ, PHYSICS_SUBSTEPS, SOLVER_ITERATIONS,
    ARENA_SEGMENT_RADIUS, CAR_RIDE_HEIGHT,
    BALL_RADIUS, CAR_SPAWN_X_DEFENSIVE, CAR_SPAWN_X_ATTACK,
    CAR_BALL_RESTITUTION,
    COLLISION_CAR_BODY, COLLISION_BALL, COLLISION_GOAL_SENSOR
)
from src.core.actions import CarAction
from src.core.arena import Arena
from src.core.ball import Ball
from src.core.car import Car


class Simulation:
    """Headless 2D physics simulation environment for Rocket League."""

    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        # No global damping: drag belongs to the entities that model it, otherwise it
        # silently bleeds energy out of every body including the player's car.
        self.space.damping = 1.0
        self.space.iterations = SOLVER_ITERATIONS

        # Score & Event tracking
        self.score_blue: int = 0
        self.score_orange: int = 0
        self.last_goal_team: Optional[str] = None
        self._spawn_index: int = 0
        self.time_elapsed: float = 0.0

        # Center coordinates
        self.center_x = MARGIN_X + FIELD_WIDTH / 2.0
        self.center_y = MARGIN_Y + FIELD_HEIGHT / 2.0

        # Instantiate entities
        self.arena = Arena(self.space)
        # 1. Ball spawns resting on the ground at kickoff
        self.ball = Ball(self.space, x=self.center_x, y=self.ball_spawn_y)
        # 2. Car spawns on its side at initial kickoff position
        init_x, init_y, init_ang, init_fac = self.get_spawn_position("blue")
        self.car = Car(self.space, x=init_x, y=init_y, angle=init_ang, team="blue")
        self.car.facing_x = init_fac

        self._setup_collision_handlers()

    @property
    def spawn_y(self) -> float:
        """Floor height that puts the car exactly at its settled ride height."""
        return MARGIN_Y + ARENA_SEGMENT_RADIUS + CAR_RIDE_HEIGHT

    @property
    def ball_spawn_y(self) -> float:
        """Floor elevation that places the ball resting on the arena floor."""
        return MARGIN_Y + ARENA_SEGMENT_RADIUS + BALL_RADIUS

    def get_spawn_position(self, team: Optional[str] = None) -> Tuple[float, float, float, int]:
        """Get the next alternating kickoff spawn (x, y, angle, facing_x) on the team's side."""
        team = team or self.car.team
        positions = [CAR_SPAWN_X_DEFENSIVE, CAR_SPAWN_X_ATTACK]
        x_blue = positions[self._spawn_index % len(positions)]
        self._spawn_index += 1

        if team == "orange":
            # Symmetrical position on Orange side
            x_orange = self.arena.x_right - (x_blue - self.arena.x_left)
            return (x_orange, self.spawn_y, 0.0, -1)
        else:
            return (x_blue, self.spawn_y, 0.0, 1)

    def _setup_collision_handlers(self):
        """Configure contact listeners for car/ball strikes and goal detection.

        Wheel-vs-world contact is handled by the car's suspension raycasts, so no sensor
        bookkeeping is needed here.
        """
        # Car Body <-> Ball: a plain rigid-body strike. Momentum transfer already produces
        # a punchy hit because the car is ~6x the ball's mass; the previous post-solve
        # "punch" impulse fired once per sub-step and injected unbounded energy.
        h_car_ball = self.space.add_collision_handler(COLLISION_CAR_BODY, COLLISION_BALL)

        def _car_ball_pre_solve(arbiter, space, data):
            arbiter.restitution = CAR_BALL_RESTITUTION
            return True

        h_car_ball.pre_solve = _car_ball_pre_solve

        # Ball <-> Goal Sensor: passive pass-through (100% inside logic handled in _check_goal)
        h_ball_goal = self.space.add_collision_handler(COLLISION_BALL, COLLISION_GOAL_SENSOR)
        h_ball_goal.begin = lambda arbiter, space, data: False

    def _check_goal(self) -> Optional[str]:
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

    def step(self, action: CarAction, dt: float = 1.0 / SIM_HZ):
        """Advance the physics simulation by dt using sub-stepping for stability."""
        sub_dt = dt / PHYSICS_SUBSTEPS
        for _ in range(PHYSICS_SUBSTEPS):
            self.car.update(action, sub_dt)
            self.ball.apply_aerodynamics(sub_dt)
            self.space.step(sub_dt)

        self.ball.record_trail()
        self.time_elapsed += dt

        # Score logic: goal only counts when ball is 100% inside the goal pocket
        scoring_team = self._check_goal()
        if scoring_team is not None:
            if scoring_team == "blue":
                self.score_blue += 1
            elif scoring_team == "orange":
                self.score_orange += 1
            self.last_goal_team = scoring_team
            # Reset kickoff after goal
            self.reset(reset_scores=False)

    def reset(self, reset_scores: bool = False, spawn_pos: Optional[Tuple[float, float, float, int]] = None):
        """Reset the arena, ball, and car to initial kickoff conditions."""
        if reset_scores:
            self.score_blue = 0
            self.score_orange = 0
            self.last_goal_team = None

        # 1. Ball spawns resting on the ground at center
        self.ball.reset(self.center_x, self.ball_spawn_y)

        # 2. Car spawns at its next alternating position on its side
        if spawn_pos is None:
            x, y, angle, facing = self.get_spawn_position()
        else:
            x, y, angle, facing = spawn_pos

        self.car.reset(x, y, angle=angle, facing_x=facing)
        self.time_elapsed = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Query complete simulation state for headless evaluation or RL."""
        return {
            "time": self.time_elapsed,
            "ball": {
                "position": self.ball.position,
                "velocity": self.ball.velocity,
                "angle": self.ball.angle
            },
            "car": {
                "position": self.car.position,
                "nose_position": self.car.nose_position,
                "velocity": self.car.velocity,
                "angle": self.car.angle,
                "angular_velocity": self.car.body.angular_velocity,
                "is_grounded": self.car.is_grounded,
                "ground_normal": self.car.ground_normal,
                "wheel_contacts": self.car.wheel_contact_count,
                "boost": self.car.boost_amount,
                "is_boosting": self.car.is_boosting,
                "input_vector": self.car.last_input_vector,
                "facing_x": self.car.facing_x
            },
            "score": {
                "blue": self.score_blue,
                "orange": self.score_orange
            },
            "last_goal": self.last_goal_team
        }
