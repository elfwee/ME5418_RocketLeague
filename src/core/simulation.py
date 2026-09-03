"""Headless simulation manager orchestrating Pymunk space, entities, and collisions."""
from typing import Dict, Any, Optional
import pymunk
from src.config import (
    FIELD_WIDTH, FIELD_HEIGHT, MARGIN_X, MARGIN_Y,
    GRAVITY, SIM_HZ, PHYSICS_SUBSTEPS, SOLVER_ITERATIONS,
    ARENA_SEGMENT_RADIUS, CAR_RIDE_HEIGHT,
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

        # Center coordinates
        self.center_x = MARGIN_X + FIELD_WIDTH / 2.0
        self.center_y = MARGIN_Y + FIELD_HEIGHT / 2.0

        # Instantiate entities
        self.arena = Arena(self.space)
        self.ball = Ball(self.space, x=self.center_x, y=self.center_y)
        self.car = Car(self.space, x=MARGIN_X + FIELD_WIDTH * 0.25, y=self.spawn_y, angle=0.0, team="blue")

        # Metrics & event tracking
        self.time_elapsed: float = 0.0
        self.last_goal_team: Optional[str] = None

        self._setup_collision_handlers()

    @property
    def spawn_y(self) -> float:
        """Floor height that puts the car exactly at its settled ride height."""
        return MARGIN_Y + ARENA_SEGMENT_RADIUS + CAR_RIDE_HEIGHT

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

        # Ball <-> Goal Sensor
        h_ball_goal = self.space.add_collision_handler(COLLISION_BALL, COLLISION_GOAL_SENSOR)

        def _ball_goal_begin(arbiter, space, data):
            _, shape_sensor = arbiter.shapes
            self.last_goal_team = getattr(shape_sensor, "team", "unknown")
            return False

        h_ball_goal.begin = _ball_goal_begin

    def step(self, action: CarAction, dt: float = 1.0 / SIM_HZ):
        """Advance the physics simulation by dt using sub-stepping for stability."""
        sub_dt = dt / PHYSICS_SUBSTEPS
        for _ in range(PHYSICS_SUBSTEPS):
            self.car.update(action, sub_dt)
            self.ball.apply_aerodynamics(sub_dt)
            self.space.step(sub_dt)

        self.ball.record_trail()
        self.time_elapsed += dt

    def reset(self):
        """Reset the arena, ball, and car to initial kickoff conditions."""
        self.ball.reset(self.center_x, self.center_y)
        self.car.reset(MARGIN_X + FIELD_WIDTH * 0.25, self.spawn_y, angle=0.0)
        self.last_goal_team = None
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
            "last_goal": self.last_goal_team
        }
