"""Model Predictive Control State Machine (HZISM) Heuristic Bot for 2D Rocket League."""
import math
from typing import Tuple, Optional
from src.core.actions import CarAction
from src.config import (
    CAR_MAX_GROUND_SPEED, CAR_MAX_AIR_SPEED, GRAVITY_MAG,
    BOT_DODGE_STRIKE_DIST, BOT_AERIAL_MIN_BOOST, BOT_DEFENSE_ZONE_X
)

from src.config import (FIXED_DT, CAR_DRIVE_ACCEL)

from dataclasses import dataclass
import math
import numpy as np
from enum import Enum

class State(Enum):
    ATTACK = "attack"
    DEFEND = "defend"

try:
    import cvxpy as cp
except ImportError:
    cp = None


@dataclass
class MPCConfig:
    horizon: int = 20
    dt: float = FIXED_DT

    max_accel: float = CAR_DRIVE_ACCEL
    max_angular_accel: float = CAR_GROUND_ANGULAR_ACCEL

    # Cost weights
    q_pos: float = 20.0
    q_vel: float = 2.0
    q_angle: float = 8.0
    q_omega: float = 1.0

    r_accel: float = 0.10
    r_angular: float = 0.10

    rd_accel: float = 0.50
    rd_angular: float = 0.50


class CarMPCController:
    """
    MPC controller sitting ABOVE Car.

    State:
        x = [px, py, vx, vy, theta, omega]

    Control:
        u = [longitudinal_accel, angular_accel]
    """

    NX = 6
    NU = 2

    def __init__(self, config: MPCConfig = None):
        if cp is None:
            raise ImportError(
                "CarMPCController requires cvxpy. "
                "Install with: pip install cvxpy"
            )

        self.cfg = config or MPCConfig()

        self.previous_u = np.zeros(self.NU)

    # -------------------------------------------------------------- #
    # State extraction
    # -------------------------------------------------------------- #
    def get_state(self, car):
        return np.array([
            car.body.position.x,
            car.body.position.y,
            car.body.velocity.x,
            car.body.velocity.y,
            car.body.angle,
            car.body.angular_velocity,
        ], dtype=float)

    # -------------------------------------------------------------- #
    # Linearised vehicle model
    # -------------------------------------------------------------- #

    def linear_model(self, theta):
        """
        Local linearisation around the current heading.

        px[k+1] = px + vx*dt
        py[k+1] = py + vy*dt

        vx[k+1] = vx + cos(theta)*a*dt
        vy[k+1] = vy + sin(theta)*a*dt

        theta[k+1] = theta + omega*dt
        omega[k+1] = omega + alpha*dt
        """

        dt = self.cfg.dt

        A = np.eye(self.NX)

        A[0, 2] = dt
        A[1, 3] = dt
        A[4, 5] = dt

        B = np.zeros((self.NX, self.NU))

        B[2, 0] = math.cos(theta) * dt
        B[3, 0] = math.sin(theta) * dt

        B[5, 1] = dt

        return A, B

    # -------------------------------------------------------------- #
    # MPC optimization
    # -------------------------------------------------------------- #

    def solve(self, car, reference):
        """
        reference:
            ndarray shape (N + 1, 6)

        Each row:
            [px, py, vx, vy, theta, omega]
        """

        N = self.cfg.horizon
        cfg = self.cfg

        x0 = self.get_state(car)

        if reference.shape != (N + 1, self.NX):
            raise ValueError(
                f"reference must have shape {(N + 1, self.NX)}"
            )

        # Linearise around current orientation
        A, B = self.linear_model(x0[4])

        X = cp.Variable((self.NX, N + 1))
        U = cp.Variable((self.NU, N))

        constraints = [
            X[:, 0] == x0
        ]

        cost = 0

        Q = np.diag([
            cfg.q_pos,
            cfg.q_pos,
            cfg.q_vel,
            cfg.q_vel,
            cfg.q_angle,
            cfg.q_omega,
        ])

        R = np.diag([
            cfg.r_accel,
            cfg.r_angular,
        ])

        Rd = np.diag([
            cfg.rd_accel,
            cfg.rd_angular,
        ])

        for k in range(N):

            # Prediction model
            constraints += [
                X[:, k + 1] == A @ X[:, k] + B @ U[:, k]
            ]

            # Actuator limits
            constraints += [
                cp.abs(U[0, k]) <= cfg.max_accel,
                cp.abs(U[1, k]) <= cfg.max_angular_accel,
            ]

            error = X[:, k] - reference[k]

            cost += cp.quad_form(error, Q)
            cost += cp.quad_form(U[:, k], R)

            # Penalise abrupt control changes
            if k == 0:
                du = U[:, k] - self.previous_u
            else:
                du = U[:, k] - U[:, k - 1]

            cost += cp.quad_form(du, Rd)

        # Terminal cost
        terminal_error = X[:, N] - reference[N]

        cost += cp.quad_form(
            terminal_error,
            2.0 * Q
        )

        problem = cp.Problem(
            cp.Minimize(cost),
            constraints
        )

        try:
            problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False
            )
        except Exception:
            return self._fallback_action(car, reference[0])

        if U.value is None:
            return self._fallback_action(car, reference[0])

        # Receding horizon:
        # ONLY execute first MPC command.
        u = U.value[:, 0]

        self.previous_u = u.copy()

        return self.control_to_action(car, u)

    # -------------------------------------------------------------- #
    # Convert MPC command -> existing CarAction
    # -------------------------------------------------------------- #

    def control_to_action(self, car, u):

        accel = float(u[0])
        angular_accel = float(u[1])

        # Convert acceleration into your existing [-1, 1]
        # throttle representation.
        throttle = np.clip(
            accel / self.cfg.max_accel,
            -1.0,
            1.0
        )

        # Predict desired angular velocity
        omega_des = (
            car.body.angular_velocity
            + angular_accel * self.cfg.dt
        )

        # Predict desired heading
        target_angle = (
            car.body.angle
            + omega_des * self.cfg.dt
        )

        # Your Car._target_angle() interprets dir_x/dir_y
        # as a world-space desired heading.
        dir_x = math.cos(target_angle)
        dir_y = math.sin(target_angle)

        # Keep magnitude related to throttle.
        magnitude = max(abs(throttle), 0.15)

        dir_x *= magnitude
        dir_y *= magnitude

        return CarAction(
            dir_x=float(dir_x),
            dir_y=float(dir_y),
            jump=False,
            boost=False,
        )

    # -------------------------------------------------------------- #
    # Emergency fallback
    # -------------------------------------------------------------- #

    def _fallback_action(self, car, target):

        dx = target[0] - car.body.position.x
        dy = target[1] - car.body.position.y

        norm = math.hypot(dx, dy)

        if norm < 1e-6:
            return CarAction(
                dir_x=0.0,
                dir_y=0.0,
                jump=False,
                boost=False,
            )

        return CarAction(
            dir_x=dx / norm,
            dir_y=dy / norm,
            jump=False,
            boost=False,
        )

    def determine_state(self, sim, car):

        def_zone_depth = 8.5
        att_zone_depth = 7.5
        aerial_def_depth = 6.5

        bx, by = sim.ball.position
        b_vel = sim.ball.velocity

        if self.team = "orange":
            if bx > (sim.arena.x_right - def_zone_depth):
                return "DEFEND"
