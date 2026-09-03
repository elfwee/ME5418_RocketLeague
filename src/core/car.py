"""Car physics model with bidirectional facing, RWD traction, 2D steering, and turtle flip."""
import math
from typing import Tuple, List
import pymunk
from src.config import (
    CAR_WIDTH, CAR_HEIGHT, CAR_MASS,
    CAR_DRIVE_ACCEL, CAR_MAX_GROUND_SPEED,
    CAR_BOOST_ACCEL, CAR_MAX_AIR_SPEED,
    CAR_JUMP_SPEED, CAR_STEER_KP, CAR_STEER_KD,
    CAR_MAX_BOOST, CAR_BOOST_DRAIN, CAR_BOOST_REFILL,
    RWD_PITCH_CUTOFF_DEG, MARGIN_Y,
    COLLISION_CAR_BODY, COLLISION_CAR_WHEEL
)
from src.core.actions import CarAction


class Car:
    """Physics representation of a 2D Rocket League car with bidirectional facing."""

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
            (-vx, vy) for vx, vy in reversed(self.base_chassis_vertices)
        ]

        # Moment of inertia
        moment = pymunk.moment_for_poly(self.mass, self.base_chassis_vertices)
        self.body = pymunk.Body(self.mass, moment, pymunk.Body.DYNAMIC)
        self.body.position = (x, y)
        self.body.angle = angle

        # 1. Main Tapered Chassis Shape
        self.chassis_shape = pymunk.Poly(self.body, self.base_chassis_vertices, radius=0.04)
        self.chassis_shape.elasticity = 0.38
        self.chassis_shape.friction = 0.6
        self.chassis_shape.collision_type = COLLISION_CAR_BODY
        self.chassis_shape.car = self

        # 2. Underside Wheel Sensor Shape
        self.wheel_sensor = pymunk.Segment(
            self.body,
            (-0.75, -0.42),
            (0.65, -0.42),
            radius=0.06
        )
        self.wheel_sensor.sensor = True
        self.wheel_sensor.collision_type = COLLISION_CAR_WHEEL
        self.wheel_sensor.car = self

        self.space.add(self.body, self.chassis_shape, self.wheel_sensor)

        # Internal state tracking
        self.boost_amount: float = CAR_MAX_BOOST
        self.wheel_contact_count: int = 0
        self.is_boosting: bool = False
        self._prev_jump_action: bool = False

        # 2D direction input vector for purple arrow visualization
        self.last_input_vector: Tuple[float, float] = (0.0, 0.0)

        # Upside-down auto-righting timer
        self.turtled_time: float = 0.0

    @property
    def is_grounded(self) -> bool:
        """True if wheels are in contact with arena surfaces or ball."""
        return self.wheel_contact_count > 0

    @property
    def rear_wheel_local(self) -> Tuple[float, float]:
        """Local coordinate of the rear wheel axle based on facing direction."""
        return (-0.65 * self.facing_x, -0.38)

    @property
    def front_wheel_local(self) -> Tuple[float, float]:
        """Local coordinate of the front wheel axle based on facing direction."""
        return (0.55 * self.facing_x, -0.38)

    @property
    def forward_vector(self) -> Tuple[float, float]:
        """Unit vector pointing along the car's nose in world space."""
        theta = self.body.angle
        if self.facing_x == 1:
            return (math.cos(theta), math.sin(theta))
        else:
            return (-math.cos(theta), -math.sin(theta))

    @property
    def up_vector(self) -> Tuple[float, float]:
        """Unit vector pointing toward car's roof in world space."""
        theta = self.body.angle
        return (-math.sin(theta), math.cos(theta))

    @property
    def nose_position(self) -> Tuple[float, float]:
        """World position of the tapered front nose tip."""
        local_x = 1.05 * self.facing_x
        p = self.body.local_to_world((local_x, -0.15))
        return (p.x, p.y)

    @property
    def active_chassis_vertices(self) -> List[Tuple[float, float]]:
        """Current polygon vertices in local space based on facing direction."""
        return self.base_chassis_vertices if self.facing_x == 1 else self.flipped_chassis_vertices

    def _set_facing(self, new_facing: int):
        """Switch car facing direction between Right (+1) and Left (-1)."""
        if new_facing == self.facing_x:
            return
        self.facing_x = new_facing
        verts = self.active_chassis_vertices
        self.chassis_shape.unsafe_set_vertices(verts)
        self.space.reindex_shapes_for_body(self.body)

    def update(self, action: CarAction, dt: float):
        """Apply physics forces based on 2D direction vector, RWD traction, boost, and auto-right."""
        action.clamp()
        self.is_boosting = False
        self.last_input_vector = (action.dir_x, action.dir_y)

        # Update facing direction when horizontal input is commanded
        if action.dir_x > 0.15:
            self._set_facing(1)
        elif action.dir_x < -0.15:
            self._set_facing(-1)

        fwd_x, fwd_y = self.forward_vector
        up_x, up_y = self.up_vector
        vx, vy = self.body.velocity
        mag = action.magnitude

        # --- 1. 2D Direction Steering Control ---
        if mag > 0.15:
            # Target angle relative to car's horizontal facing direction
            if self.facing_x == 1:
                target_angle = math.atan2(action.dir_y, action.dir_x)
            else:
                target_angle = math.atan2(-action.dir_y, -action.dir_x)

            curr_angle = self.body.angle
            diff = (target_angle - curr_angle + math.pi) % (2.0 * math.pi) - math.pi

            max_torque = self.body.moment * 600.0
            torque = self.body.moment * (diff * CAR_STEER_KP - self.body.angular_velocity * CAR_STEER_KD)
            self.body.torque = max(-max_torque, min(max_torque, torque))
        else:
            if self.is_grounded:
                # Return nose to horizontal ground when no directional steering is commanded
                diff = (0.0 - self.body.angle + math.pi) % (2.0 * math.pi) - math.pi
                self.body.torque = self.body.moment * (diff * 70.0 - self.body.angular_velocity * 14.0)
            else:
                self.body.angular_velocity *= max(0.0, 1.0 - 6.0 * dt)

        # --- 2. Rear-Wheel Drive (RWD) Ground Locomotion with Pitch Cutoff ---
        if self.is_grounded:
            if self.boost_amount < CAR_MAX_BOOST:
                self.boost_amount = min(CAR_MAX_BOOST, self.boost_amount + CAR_BOOST_REFILL * dt)

            # Calculate pitch angle relative to horizontal ground
            pitch_deg = abs(math.degrees(math.atan2(fwd_y, abs(fwd_x))))
            cutoff_deg = RWD_PITCH_CUTOFF_DEG

            if pitch_deg < cutoff_deg:
                pitch_factor = max(0.0, math.cos(math.radians(pitch_deg)))
            else:
                pitch_factor = 0.0

            rw_world = self.body.local_to_world(self.rear_wheel_local)

            # Ground wheelie handling & Rocket Launch:
            # When boosting with an upward command, disable sticky downforce and provide front
            # lift assist so the rocket launcher takes off diagonally toward the target vector!
            if action.boost and action.dir_y > 0.08 and self.boost_amount > 0.0:
                front_pt = self.body.local_to_world((0.85 * self.facing_x, 0.0))
                self.body.apply_force_at_world_point((0.0, self.mass * 20.0 * action.dir_y), front_pt)
            elif action.dir_y > 0.08:
                # Normal non-boosted wheelie: sticky downforce keeps rear axle planted
                sticky_force = -self.mass * 12.0
                self.body.apply_force_at_world_point((sticky_force * up_x, sticky_force * up_y), rw_world)
            else:
                sticky_force = -self.mass * 8.0
                self.body.apply_force_at_world_point((sticky_force * up_x, sticky_force * up_y), self.body.position)

            # Horizontal drive force at rear wheel
            if abs(action.dir_x) > 0.1:
                target_fwd_speed = CAR_MAX_GROUND_SPEED * abs(action.dir_x)
                curr_fwd_speed = vx * fwd_x + vy * fwd_y
                speed_err = target_fwd_speed - curr_fwd_speed

                max_dv = CAR_DRIVE_ACCEL * dt
                clamped_dv = max(-max_dv, min(max_dv, speed_err))
                drive_force_mag = self.mass * (clamped_dv / dt) * pitch_factor

                self.body.apply_force_at_world_point(
                    (drive_force_mag * fwd_x, drive_force_mag * fwd_y),
                    rw_world
                )
            else:
                curr_fwd_speed = vx * fwd_x + vy * fwd_y
                brake_dv = -curr_fwd_speed * min(1.0, 10.0 * dt)
                self.body.apply_force_at_world_point(
                    (self.mass * (brake_dv / dt) * fwd_x, self.mass * (brake_dv / dt) * fwd_y),
                    self.body.position
                )

            # Lateral anti-skid friction
            curr_lat_speed = vx * up_x + vy * up_y
            if abs(curr_lat_speed) > 0.01:
                lat_damp = -self.mass * curr_lat_speed * 14.0
                self.body.apply_force_at_world_point((lat_damp * up_x, lat_damp * up_y), self.body.position)

        # --- 3. Jump Impulse ---
        jump_just_pressed = action.jump and not self._prev_jump_action
        if jump_just_pressed and self.is_grounded:
            impulse_mag = self.mass * CAR_JUMP_SPEED
            self.body.apply_impulse_at_world_point((impulse_mag * up_x, impulse_mag * up_y), self.body.position)
            self.wheel_contact_count = 0

        self._prev_jump_action = action.jump

        # --- 4. Rocket Boost (Only booster propels car into the air) ---
        if action.boost and self.boost_amount > 0.0:
            self.is_boosting = True
            self.boost_amount = max(0.0, self.boost_amount - CAR_BOOST_DRAIN * dt)

            boost_force = self.mass * CAR_BOOST_ACCEL
            self.body.apply_force_at_world_point((boost_force * fwd_x, boost_force * fwd_y), self.body.position)

            # Clamp engine forward boost speed along heading (preserves free downward gravity descent)
            fwd_speed = self.body.velocity.x * fwd_x + self.body.velocity.y * fwd_y
            if fwd_speed > CAR_MAX_AIR_SPEED:
                excess = fwd_speed - CAR_MAX_AIR_SPEED
                self.body.velocity = (self.body.velocity.x - excess * fwd_x, self.body.velocity.y - excess * fwd_y)

        # --- 5. Upside-Down Auto-Righting / Turtle Flip (Preserves Facing Direction) ---
        # A car is turtled only when genuinely inverted on its roof (roof normal pointing down)
        # and wheels are off the ground. During a ground wheelie (pitch ~ 90 deg), up_vector[1] ~ 0.0,
        # so it will never trigger a false turtle reset.
        is_roof_down = self.up_vector[1] < -0.70
        is_near_floor = self.body.position.y < (MARGIN_Y + 1.25)
        wheels_off_ground = self.wheel_contact_count == 0

        if is_roof_down and is_near_floor and wheels_off_ground:
            self.turtled_time += dt
            # Snaps upright in the current facing direction with a clean hop
            if self.turtled_time > 0.10 or action.jump:
                self.body.angle = 0.0
                self.body.angular_velocity = 0.0
                self.body.position = (self.body.position.x, MARGIN_Y + 0.65)
                self.body.velocity = (self.body.velocity.x * 0.75, 3.2)
                self.wheel_contact_count = 1
                self.turtled_time = 0.0
        else:
            self.turtled_time = 0.0

    def reset(self, x: float, y: float, angle: float = 0.0, facing_x: int = 1):
        """Reset car state to starting position and orientation."""
        self.body.position = (x, y)
        self.body.angle = angle
        self.body.velocity = (0.0, 0.0)
        self.body.angular_velocity = 0.0
        self.boost_amount = CAR_MAX_BOOST
        self.wheel_contact_count = 0
        self.is_boosting = False
        self._prev_jump_action = False
        self.last_input_vector = (0.0, 0.0)
        self.turtled_time = 0.0
        self._set_facing(facing_x)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.body.position.x, self.body.position.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.body.velocity.x, self.body.velocity.y)

    @property
    def angle(self) -> float:
        return self.body.angle
