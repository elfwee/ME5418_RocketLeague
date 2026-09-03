"""Car physics: raycast suspension, grip-limited traction and rate-limited attitude control.

The car is *not* a brick sliding on the floor. It is held up by two downward raycasts (one
per axle) that act as spring/damper suspension, and it is driven by a tyre force applied at
the rear contact patch and capped by a Coulomb friction circle. That gives correct behaviour
on flat ground, slopes, corner fillets and while balanced on the ball, without any of the
scripted downforce / pitch-cutoff hacks the physical model now makes unnecessary.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pymunk

from src.config import (
    CAR_WIDTH, CAR_HEIGHT, CAR_MASS,
    CHASSIS_FRICTION, CHASSIS_ELASTICITY,
    WHEEL_AXLE_Y, WHEEL_REAR_X, WHEEL_FRONT_X,
    SUSPENSION_STIFFNESS, SUSPENSION_DAMPER, SUSPENSION_REST_LEN,
    SUSPENSION_RAY_LEN, SUSPENSION_MAX_ACCEL,
    TIRE_GRIP, CAR_STICKY_ACCEL, CAR_DRIVE_ACCEL, CAR_COAST_DECEL,
    CAR_MAX_GROUND_SPEED, WHEEL_MIN_NORMAL_DOT, SURFACE_ALIGN_MIN_DOT,
    CAR_STEER_RATE_GAIN,
    CAR_GROUND_ANGULAR_SPEED, CAR_GROUND_ANGULAR_ACCEL,
    CAR_AIR_ANGULAR_SPEED, CAR_AIR_ANGULAR_ACCEL, CAR_AIR_SPIN_DECAY_TAU,
    CAR_PITCH_INPUT_THRESHOLD, INPUT_DEADZONE, THROTTLE_DEADZONE,
    FACING_FLIP_THRESHOLD,
    CAR_JUMP_SPEED, CAR_BOOST_ACCEL, CAR_MAX_AIR_SPEED, CAR_BOOST_SPEED_FADE,
    CAR_MAX_BOOST, CAR_BOOST_DRAIN, CAR_BOOST_REFILL,
    TURTLE_UP_THRESHOLD, TURTLE_PROBE_LEN, TURTLE_TRIGGER_DELAY,
    TURTLE_HOP_SPEED, TURTLE_FLIP_DURATION,
    CAR_FLIP_ANGULAR_SPEED, CAR_FLIP_ANGULAR_ACCEL,
    COLLISION_CAR_BODY,
)
from src.core.actions import CarAction


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar into [low, high]."""
    return low if value < low else (high if value > high else value)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class WheelContact:
    """Ground probe result for a single axle."""
    hit: bool = False
    point: Tuple[float, float] = (0.0, 0.0)
    normal: Tuple[float, float] = (0.0, 1.0)
    distance: float = SUSPENSION_RAY_LEN
    compression: float = 0.0
    normal_force: float = 0.0
    other_body: Optional[pymunk.Body] = None
    anchor: Tuple[float, float] = field(default=(0.0, 0.0))


class Car:
    """Physics representation of a 2D Rocket League car with bidirectional facing."""

    _next_filter_group = 0

    def __init__(self, space: pymunk.Space, x: float, y: float, angle: float = 0.0, team: str = "blue"):
        self.space = space
        self.team = team
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.mass = CAR_MASS

        # Horizontal facing state: +1 for facing Right, -1 for facing Left
        self.facing_x: int = 1

        # Base tapered convex polygon vertices (facing Right, counter-clockwise)
        self.base_chassis_vertices: List[Tuple[float, float]] = [
            (-1.00, -0.32),   # Rear bottom
            (0.95, -0.32),    # Front bottom
            (1.05, -0.15),    # Low tapered nose tip (wedge)
            (0.35, 0.36),     # Hood slope to cabin
            (-0.35, 0.38),    # Cabin roof
            (-1.00, 0.22)     # Rear spoiler deck
        ]

        # Flipped vertices for facing Left (reversing list maintains CCW winding)
        self.flipped_chassis_vertices: List[Tuple[float, float]] = [
            (vx, -vy) for vx, vy in reversed(self.base_chassis_vertices)
        ]

        moment = pymunk.moment_for_poly(self.mass, self.base_chassis_vertices)
        self.body = pymunk.Body(self.mass, moment, pymunk.Body.DYNAMIC)
        self.body.position = (x, y)
        self.body.angle = angle

        # Own shapes share a filter group so the suspension raycasts ignore the car itself
        Car._next_filter_group += 1
        self._filter_group = Car._next_filter_group
        self._query_filter = pymunk.ShapeFilter(group=self._filter_group)

        self.chassis_shape = pymunk.Poly(self.body, self.base_chassis_vertices, radius=0.04)
        self.chassis_shape.elasticity = CHASSIS_ELASTICITY
        self.chassis_shape.friction = CHASSIS_FRICTION
        self.chassis_shape.collision_type = COLLISION_CAR_BODY
        self.chassis_shape.filter = self._query_filter
        self.chassis_shape.car = self

        self.space.add(self.body, self.chassis_shape)

        # Internal state tracking
        self.boost_amount: float = CAR_MAX_BOOST
        self.is_boosting: bool = False
        self._prev_jump_action: bool = False

        # [rear, front] suspension probes, refreshed every physics sub-step
        self.wheels: List[WheelContact] = [WheelContact(), WheelContact()]
        self._ground_normal: Tuple[float, float] = (0.0, 1.0)
        self._grounded: bool = False

        # 2D direction input vector for purple arrow visualization
        self.last_input_vector: Tuple[float, float] = (0.0, 0.0)

        # Upside-down auto-righting state
        self.turtled_time: float = 0.0
        self._recovery_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Geometry / state queries
    # ------------------------------------------------------------------ #

    @property
    def is_grounded(self) -> bool:
        """True if at least one suspension probe found a drivable surface."""
        return self._grounded

    @property
    def wheel_contact_count(self) -> int:
        """Number of axles currently touching a surface."""
        return sum(1 for w in self.wheels if w.hit)

    @property
    def ground_normal(self) -> Tuple[float, float]:
        """Averaged surface normal under the car (world up when airborne)."""
        return self._ground_normal

    @property
    def rear_wheel_local(self) -> Tuple[float, float]:
        """Local coordinate of the rear wheel axle based on facing direction."""
        return (WHEEL_REAR_X, WHEEL_AXLE_Y * self.facing_x)

    @property
    def front_wheel_local(self) -> Tuple[float, float]:
        """Local coordinate of the front wheel axle based on facing direction."""
        return (WHEEL_FRONT_X, WHEEL_AXLE_Y * self.facing_x)

    @property
    def forward_vector(self) -> Tuple[float, float]:
        """Unit vector pointing along the car's nose in world space."""
        theta = self.body.angle
        return (math.cos(theta), math.sin(theta))

    @property
    def up_vector(self) -> Tuple[float, float]:
        """Unit vector pointing toward car's roof in world space."""
        theta = self.body.angle
        return (-self.facing_x * math.sin(theta), self.facing_x * math.cos(theta))

    @property
    def nose_position(self) -> Tuple[float, float]:
        """World position of the tapered front nose tip."""
        p = self.body.local_to_world((1.05, -0.15 * self.facing_x))
        return (p.x, p.y)

    @property
    def tail_position(self) -> Tuple[float, float]:
        """World position of the rear spoiler/tail."""
        p = self.body.local_to_world((-1.00, 0.22 * self.facing_x))
        return (p.x, p.y)

    @property
    def active_chassis_vertices(self) -> List[Tuple[float, float]]:
        """Current polygon vertices in local space based on facing direction."""
        return self.base_chassis_vertices if self.facing_x == 1 else self.flipped_chassis_vertices

    def _set_facing(self, new_facing: int, snap: bool = False):
        """Switch car facing direction between Right (+1) and Left (-1)."""
        if new_facing == self.facing_x:
            return
        self.facing_x = new_facing
        self.chassis_shape.unsafe_set_vertices(self.active_chassis_vertices)
        self.space.reindex_shapes_for_body(self.body)
        if snap:
            sign = math.copysign(math.pi, self.body.angle if self.body.angle != 0.0 else 1.0)
            self.body.angle = _wrap_angle(sign - self.body.angle)
            self.body.angular_velocity = -self.body.angular_velocity

    # ------------------------------------------------------------------ #
    # Ground sensing
    # ------------------------------------------------------------------ #

    def _sense_ground(self):
        """Cast one ray per axle along the car's -up axis and cache the contact data."""
        up_x, up_y = self.up_vector
        nx_sum = 0.0
        ny_sum = 0.0
        hits = 0

        for wheel, local in ((self.wheels[0], self.rear_wheel_local),
                             (self.wheels[1], self.front_wheel_local)):
            start = self.body.local_to_world(local)
            end = (start.x - up_x * SUSPENSION_RAY_LEN, start.y - up_y * SUSPENSION_RAY_LEN)
            wheel.anchor = (start.x, start.y)

            info = self.space.segment_query_first(start, end, 0.0, self._query_filter)

            # Reject grazing/backside hits whose normal is not roughly beneath the car
            if info is None or (info.normal.x * up_x + info.normal.y * up_y) < WHEEL_MIN_NORMAL_DOT:
                wheel.hit = False
                wheel.compression = 0.0
                wheel.normal_force = 0.0
                wheel.distance = SUSPENSION_RAY_LEN
                wheel.other_body = None
                continue

            wheel.hit = True
            wheel.point = (info.point.x, info.point.y)
            wheel.normal = (info.normal.x, info.normal.y)
            wheel.distance = info.alpha * SUSPENSION_RAY_LEN
            wheel.compression = max(0.0, SUSPENSION_REST_LEN - wheel.distance)
            wheel.other_body = info.shape.body
            nx_sum += info.normal.x
            ny_sum += info.normal.y
            hits += 1

        self._grounded = hits > 0
        if hits > 0:
            norm = math.hypot(nx_sum, ny_sum)
            self._ground_normal = (nx_sum / norm, ny_sum / norm) if norm > 1e-9 else (up_x, up_y)
        else:
            self._ground_normal = (0.0, 1.0)

    def _apply_suspension(self):
        """Spring/damper force per axle along the surface normal, with Newton's third law."""
        half_mass = self.mass * 0.5

        for wheel in self.wheels:
            if not wheel.hit:
                continue

            nx, ny = wheel.normal
            v = self.body.velocity_at_world_point(wheel.anchor)
            v_n = v.x * nx + v.y * ny

            accel = SUSPENSION_STIFFNESS * wheel.compression - SUSPENSION_DAMPER * v_n
            force = half_mass * _clamp(accel, 0.0, SUSPENSION_MAX_ACCEL)
            wheel.normal_force = force
            if force <= 0.0:
                continue

            self.body.apply_force_at_world_point((force * nx, force * ny), wheel.anchor)

            # Push back on whatever we are standing on (e.g. dribbling the ball)
            other = wheel.other_body
            if other is not None and other.body_type == pymunk.Body.DYNAMIC:
                other.apply_force_at_world_point((-force * nx, -force * ny), wheel.point)

    # ------------------------------------------------------------------ #
    # Attitude control
    # ------------------------------------------------------------------ #

    def _drive_angle_to(self, target_angle: float, dt: float,
                        omega_max: float, alpha_max: float):
        """Rate- and acceleration-limited attitude controller (cannot overshoot or ring)."""
        diff = _wrap_angle(target_angle - self.body.angle)
        omega_target = _clamp(diff * CAR_STEER_RATE_GAIN, -omega_max, omega_max)
        alpha = _clamp((omega_target - self.body.angular_velocity) / dt, -alpha_max, alpha_max)
        self.body.torque += self.body.moment * alpha

    def _surface_align_angle(self) -> Optional[float]:
        """Body angle that puts the car's roof along the ground normal, if drivable."""
        nx, ny = self._ground_normal
        if ny < SURFACE_ALIGN_MIN_DOT:
            return None
        return math.atan2(-self.facing_x * nx, self.facing_x * ny)

    def _target_angle(self, action: CarAction) -> Optional[float]:
        """Resolve the commanded body angle from the 2D input vector and ground contour."""
        if action.magnitude > INPUT_DEADZONE:
            # A meaningful vertical component means the player is aiming a heading
            # (wheelie, aerial launch), so the world-space input vector wins.
            if not self._grounded or abs(action.dir_y) >= CAR_PITCH_INPUT_THRESHOLD:
                return math.atan2(action.dir_y, action.dir_x)
            return self._surface_align_angle()

        return self._surface_align_angle() if self._grounded else None

    def _apply_attitude(self, action: CarAction, dt: float):
        """Steer the chassis toward the commanded heading, or damp spin when idle."""
        if self._recovery_time > 0.0:
            upright_target = 0.0 if self.facing_x == 1 else math.pi
            self._drive_angle_to(upright_target, dt, CAR_FLIP_ANGULAR_SPEED, CAR_FLIP_ANGULAR_ACCEL)
            return

        target = self._target_angle(action)
        if target is None:
            # No heading command and nothing to align to: bleed off spin smoothly.
            # Exponential decay keeps this independent of the sub-step size.
            self.body.angular_velocity *= math.exp(-dt / CAR_AIR_SPIN_DECAY_TAU)
            return

        if self._grounded:
            self._drive_angle_to(target, dt, CAR_GROUND_ANGULAR_SPEED, CAR_GROUND_ANGULAR_ACCEL)
        else:
            self._drive_angle_to(target, dt, CAR_AIR_ANGULAR_SPEED, CAR_AIR_ANGULAR_ACCEL)

    # ------------------------------------------------------------------ #
    # Traction
    # ------------------------------------------------------------------ #

    def _apply_traction(self, action: CarAction, dt: float):
        """Rear-wheel drive / engine braking along the surface tangent, limited by tyre grip."""
        if not self._grounded:
            return

        if self.boost_amount < CAR_MAX_BOOST:
            self.boost_amount = min(CAR_MAX_BOOST, self.boost_amount + CAR_BOOST_REFILL * dt)

        rear, front = self.wheels
        loaded = [w for w in self.wheels if w.normal_force > 0.0]
        if not loaded:
            # Tyres are unloaded (mid-jump, cresting a bump). Nothing can be transmitted
            # through the contact patch, so no downforce and no drive force.
            return

        nx, ny = self._ground_normal

        # Downforce keeps the tyres loaded so the car can hold slopes and corner fillets.
        # It is only real while a contact patch exists to react against, otherwise it would
        # act as a phantom rope dragging the car back down out of every jump.
        stick = self.mass * CAR_STICKY_ACCEL * (len(loaded) / len(self.wheels))
        self.body.apply_force_at_world_point((-stick * nx, -stick * ny), self.body.position)

        # Surface tangent oriented along the nose
        fwd_x, fwd_y = self.forward_vector
        t_x, t_y = -ny, nx
        if t_x * fwd_x + t_y * fwd_y < 0.0:
            t_x, t_y = -t_x, -t_y

        vel = self.body.velocity
        v_t = vel.x * t_x + vel.y * t_y
        throttle = abs(action.dir_x)

        if throttle > THROTTLE_DEADZONE:
            target_speed = CAR_MAX_GROUND_SPEED * throttle
            accel = _clamp((target_speed - v_t) / dt, -CAR_DRIVE_ACCEL, CAR_DRIVE_ACCEL)
            # Rear-wheel drive: only the driven axle's normal load provides grip, which is
            # what naturally kills traction during a wheelie or on a near-vertical surface.
            grip = TIRE_GRIP * rear.normal_force
            point = rear.point if rear.hit else self.body.position
        else:
            accel = _clamp(-v_t / dt, -CAR_COAST_DECEL, CAR_COAST_DECEL)
            grip = TIRE_GRIP * (rear.normal_force + front.normal_force)
            point = self.body.position

        force = _clamp(self.mass * accel, -grip, grip)
        if force != 0.0:
            self.body.apply_force_at_world_point((force * t_x, force * t_y), point)

    # ------------------------------------------------------------------ #
    # Jump, boost and recovery
    # ------------------------------------------------------------------ #

    def _apply_jump(self, action: CarAction):
        """Edge-triggered jump impulse along the car's roof axis."""
        jump_just_pressed = action.jump and not self._prev_jump_action
        self._prev_jump_action = action.jump

        if not (jump_just_pressed and self._grounded):
            return

        up_x, up_y = self.up_vector
        impulse = self.mass * CAR_JUMP_SPEED
        self.body.apply_impulse_at_world_point((impulse * up_x, impulse * up_y), self.body.position)

        # Unload the suspension for this sub-step so it cannot fight the impulse
        self._grounded = False
        for wheel in self.wheels:
            wheel.hit = False
            wheel.normal_force = 0.0
            wheel.compression = 0.0

    def _apply_boost(self, action: CarAction, dt: float):
        """Rocket thrust along the heading, faded out near top speed."""
        if not (action.boost and self.boost_amount > 0.0):
            return

        self.is_boosting = True
        self.boost_amount = max(0.0, self.boost_amount - CAR_BOOST_DRAIN * dt)

        fwd_x, fwd_y = self.forward_vector
        vel = self.body.velocity
        fwd_speed = vel.x * fwd_x + vel.y * fwd_y

        # Fading the thrust instead of clamping the velocity keeps the resultant vector
        # continuous, so gravity and thrust always sum cleanly.
        fade = _clamp((CAR_MAX_AIR_SPEED - fwd_speed) / CAR_BOOST_SPEED_FADE, 0.0, 1.0)
        if fade <= 0.0:
            return

        thrust = self.mass * CAR_BOOST_ACCEL * fade
        self.body.apply_force_at_world_point((thrust * fwd_x, thrust * fwd_y), self.body.position)

    def _is_turtled(self) -> bool:
        """True when resting inverted with a surface close beneath the centre of mass."""
        if self._grounded or self.up_vector[1] > TURTLE_UP_THRESHOLD:
            return False

        origin = self.body.position
        end = (origin.x, origin.y - TURTLE_PROBE_LEN)
        return self.space.segment_query_first(origin, end, 0.0, self._query_filter) is not None

    def _update_recovery(self, action: CarAction, dt: float):
        """Detect a turtled car and recover with a physical hop-and-flip."""
        if self._recovery_time > 0.0:
            self._recovery_time = max(0.0, self._recovery_time - dt)
            return

        if not self._is_turtled():
            self.turtled_time = 0.0
            return

        self.turtled_time += dt
        if self.turtled_time <= TURTLE_TRIGGER_DELAY and not action.jump:
            return

        # Hop straight up in world space, then let the high-authority attitude
        # controller sweep the chassis upright while airborne.
        self.body.apply_impulse_at_world_point(
            (0.0, self.mass * TURTLE_HOP_SPEED), self.body.position
        )
        self._recovery_time = TURTLE_FLIP_DURATION
        self.turtled_time = 0.0

    # ------------------------------------------------------------------ #
    # Main update
    # ------------------------------------------------------------------ #

    def update(self, action: CarAction, dt: float):
        """Advance the car controller by one physics sub-step."""
        action.clamp()
        self.is_boosting = False
        self.last_input_vector = (action.dir_x, action.dir_y)

        # Update facing based on heading / input (unless turtled on the ground or recovering)
        if not (self._is_turtled() or self._recovery_time > 0.0):
            if self._grounded and abs(action.dir_x) > FACING_FLIP_THRESHOLD:
                new_facing = 1 if action.dir_x > 0 else -1
                if new_facing != self.facing_x:
                    self._set_facing(new_facing, snap=True)
            else:
                # When airborne, animation flips according to blue heading vector side
                fwd_x = math.cos(self.body.angle)
                if fwd_x > 0.05 and self.facing_x != 1:
                    self._set_facing(1, snap=False)
                elif fwd_x < -0.05 and self.facing_x != -1:
                    self._set_facing(-1, snap=False)

        self._sense_ground()
        self._update_recovery(action, dt)
        self._apply_attitude(action, dt)
        self._apply_jump(action)
        self._apply_suspension()
        self._apply_traction(action, dt)
        self._apply_boost(action, dt)

    def reset(self, x: float, y: float, angle: float = 0.0, facing_x: Optional[int] = None):
        """Reset car state to starting position and orientation."""
        self.body.position = (x, y)
        if facing_x is None:
            facing_x = -1 if math.cos(angle) < 0.0 else 1
        elif facing_x == -1 and angle == 0.0:
            angle = math.pi
        self.body.angle = angle
        self.body.velocity = (0.0, 0.0)
        self.body.angular_velocity = 0.0
        self.body.force = (0.0, 0.0)
        self.body.torque = 0.0
        self.boost_amount = CAR_MAX_BOOST
        self.is_boosting = False
        self._prev_jump_action = False
        self.last_input_vector = (0.0, 0.0)
        self.turtled_time = 0.0
        self._recovery_time = 0.0
        self._grounded = False
        self._ground_normal = (0.0, 1.0)
        for wheel in self.wheels:
            wheel.hit = False
            wheel.compression = 0.0
            wheel.normal_force = 0.0
        self._set_facing(facing_x, snap=False)
        self.space.reindex_shapes_for_body(self.body)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.body.position.x, self.body.position.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.body.velocity.x, self.body.velocity.y)

    @property
    def angle(self) -> float:
        return self.body.angle
