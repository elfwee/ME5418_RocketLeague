"""Ball entity with physical bounciness, mass, and trajectory tracking."""
import math
from collections import deque
from typing import Deque, Tuple
import pymunk
from src.config import (
    BALL_RADIUS, BALL_MASS, BALL_RESTITUTION,
    BALL_FRICTION, BALL_AIR_DRAG, BALL_SPIN_DRAG,
    BALL_MAX_SPEED, BALL_MAX_SPIN,
    COLLISION_BALL
)


class Ball:
    """Physics representation of the 2D Rocket League ball."""

    def __init__(self, space: pymunk.Space, x: float, y: float):
        self.space = space
        self.radius = BALL_RADIUS

        moment = pymunk.moment_for_circle(BALL_MASS, 0, self.radius)
        self.body = pymunk.Body(BALL_MASS, moment, pymunk.Body.DYNAMIC)
        self.body.position = (x, y)

        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = BALL_RESTITUTION
        self.shape.friction = BALL_FRICTION
        self.shape.collision_type = COLLISION_BALL

        self.space.add(self.body, self.shape)

        # Trail history for visualization (stores recent (x, y) positions)
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=25)

    def apply_aerodynamics(self, dt: float):
        """Apply exponential linear/angular drag and clamp terminal speed.

        Exponential decay keeps the drag identical regardless of the sub-step size.
        Must run every physics sub-step so the speed cap can prevent tunnelling.
        """
        if BALL_AIR_DRAG > 0.0:
            self.body.velocity = self.body.velocity * math.exp(-BALL_AIR_DRAG * dt)
        if BALL_SPIN_DRAG > 0.0:
            self.body.angular_velocity *= math.exp(-BALL_SPIN_DRAG * dt)

        speed = self.body.velocity.length
        if speed > BALL_MAX_SPEED:
            self.body.velocity = self.body.velocity * (BALL_MAX_SPEED / speed)

        spin = self.body.angular_velocity
        if abs(spin) > BALL_MAX_SPIN:
            self.body.angular_velocity = math.copysign(BALL_MAX_SPIN, spin)

    def record_trail(self):
        """Append the current position to the visualization trail (once per rendered frame)."""
        self.trail.append((self.body.position.x, self.body.position.y))

    def update(self, dt: float):
        """Convenience wrapper applying aerodynamics and recording one trail sample."""
        self.apply_aerodynamics(dt)
        self.record_trail()

    def reset(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Reset ball to specified coordinates with optional initial velocity."""
        self.body.position = (x, y)
        self.body.velocity = (vx, vy)
        self.body.angular_velocity = 0.0
        self.body.angle = 0.0
        self.body.force = (0.0, 0.0)
        self.body.torque = 0.0
        self.trail.clear()

    @property
    def position(self) -> Tuple[float, float]:
        return (self.body.position.x, self.body.position.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.body.velocity.x, self.body.velocity.y)

    @property
    def angle(self) -> float:
        return self.body.angle
