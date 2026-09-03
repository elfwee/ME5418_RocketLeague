"""Headless simulation manager orchestrating Pymunk space, entities, and collisions."""
from typing import Dict, Any, Optional
import pymunk
from src.config import (
    FIELD_WIDTH, FIELD_HEIGHT, MARGIN_X, MARGIN_Y,
    GRAVITY, SIM_HZ, PHYSICS_SUBSTEPS,
    COLLISION_ARENA, COLLISION_CAR_BODY,
    COLLISION_CAR_WHEEL, COLLISION_BALL, COLLISION_GOAL_SENSOR
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
        self.space.damping = 0.999

        # Center coordinates
        self.center_x = MARGIN_X + FIELD_WIDTH / 2.0
        self.center_y = MARGIN_Y + FIELD_HEIGHT / 2.0

        # Instantiate entities
        self.arena = Arena(self.space)
        self.ball = Ball(self.space, x=self.center_x, y=self.center_y)
        self.car = Car(self.space, x=MARGIN_X + FIELD_WIDTH * 0.25, y=MARGIN_Y + 0.6, angle=0.0, team="blue")

        # Metrics & event tracking
        self.time_elapsed: float = 0.0
        self.last_goal_team: Optional[str] = None

        self._setup_collision_handlers()

    def _setup_collision_handlers(self):
        """Configure contact listeners for wheel sensors, ball collisions, and goals."""
        # 1. Car Wheels <-> Arena Surfaces
        h_wheel_arena = self.space.add_collision_handler(COLLISION_CAR_WHEEL, COLLISION_ARENA)

        def _wheel_arena_begin(arbiter, space, data):
            shape_wheel, _ = arbiter.shapes
            if hasattr(shape_wheel, "car"):
                shape_wheel.car.wheel_contact_count += 1
            return False

        def _wheel_arena_separate(arbiter, space, data):
            shape_wheel, _ = arbiter.shapes
            if hasattr(shape_wheel, "car"):
                shape_wheel.car.wheel_contact_count = max(0, shape_wheel.car.wheel_contact_count - 1)
            return False

        h_wheel_arena.begin = _wheel_arena_begin
        h_wheel_arena.separate = _wheel_arena_separate

        # 2. Car Wheels <-> Ball (Flip resets and underside pop)
        h_wheel_ball = self.space.add_collision_handler(COLLISION_CAR_WHEEL, COLLISION_BALL)

        def _wheel_ball_begin(arbiter, space, data):
            shape_wheel, _ = arbiter.shapes
            if hasattr(shape_wheel, "car"):
                shape_wheel.car.wheel_contact_count += 1
            return False

        def _wheel_ball_separate(arbiter, space, data):
            shape_wheel, _ = arbiter.shapes
            if hasattr(shape_wheel, "car"):
                shape_wheel.car.wheel_contact_count = max(0, shape_wheel.car.wheel_contact_count - 1)
            return False

        h_wheel_ball.begin = _wheel_ball_begin
        h_wheel_ball.separate = _wheel_ball_separate

        # 3. Car Body <-> Ball (High-energy physical strike with tapered scoop)
        h_car_ball = self.space.add_collision_handler(COLLISION_CAR_BODY, COLLISION_BALL)

        def _car_ball_pre_solve(arbiter, space, data):
            # Boost restitution during active car strikes to produce punchy momentum transfer
            arbiter.restitution = 0.88
            return True

        def _car_ball_post_solve(arbiter, space, data):
            shape_car, shape_ball = arbiter.shapes
            car = getattr(shape_car, "car", None)
            if car is not None:
                cvx, cvy = car.body.velocity
                fwd_x, fwd_y = car.forward_vector
                fwd_speed = cvx * fwd_x + cvy * fwd_y
                # If car is driving or boosting forward into the ball
                if fwd_speed > 0.5:
                    ball_mass = self.ball.body.mass
                    # Active forward punch impulse along driving direction (calibrated for authentic feel)
                    punch_mag = ball_mass * (fwd_speed * 0.55)
                    # Slight upward pop so ball rebounds cleanly downfield
                    pop_y = ball_mass * max(1.8, 0.22 * fwd_speed)
                    self.ball.body.apply_impulse_at_world_point(
                        (punch_mag * fwd_x, punch_mag * max(0.05, fwd_y) + pop_y),
                        self.ball.body.position
                    )

        h_car_ball.pre_solve = _car_ball_pre_solve
        h_car_ball.post_solve = _car_ball_post_solve

        # 4. Ball <-> Goal Sensor
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
            self.space.step(sub_dt)

        self.ball.update(dt)
        self.time_elapsed += dt

    def reset(self):
        """Reset the arena, ball, and car to initial kickoff conditions."""
        self.ball.reset(self.center_x, self.center_y)
        self.car.reset(MARGIN_X + FIELD_WIDTH * 0.25, MARGIN_Y + 0.6, angle=0.0)
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
                "is_grounded": self.car.is_grounded,
                "boost": self.car.boost_amount,
                "is_boosting": self.car.is_boosting,
                "input_vector": self.car.last_input_vector,
                "facing_x": self.car.facing_x
            },
            "last_goal": self.last_goal_team
        }
