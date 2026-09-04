"""Hierarchical Zone & Intercept State Machine (HZISM) Heuristic Bot for 2D Rocket League."""
import math
from typing import Tuple, Optional
from src.core.actions import CarAction
from src.config import (
    CAR_MAX_GROUND_SPEED, CAR_MAX_AIR_SPEED, GRAVITY_MAG,
    BOT_DODGE_STRIKE_DIST, BOT_AERIAL_MIN_BOOST, BOT_DEFENSE_ZONE_X
)


class HeuristicBot:
    """Rule-based AI bot with trajectory lookahead, shadow defense, and dynamic aerial/ground mechanics."""

    def __init__(self, team: str = "orange", difficulty: str = "pro"):
        self.team = team
        self.difficulty = difficulty
        self.current_state = "KICKOFF"
        self._jump_cooldown = 0.0
        self._jump_press_timer = 0
        self._last_dodge_time = 0.0

    def predict_ball_position(self, sim, dt_ahead: float) -> Tuple[float, float]:
        """Predict ball position dt_ahead seconds into the future assuming ballistic trajectory and floor bounce."""
        bx, by = sim.ball.position
        vx, vy = sim.ball.velocity
        floor_y = sim.ball_spawn_y

        pred_x = bx + vx * dt_ahead
        pred_y = by + vy * dt_ahead - 0.5 * GRAVITY_MAG * (dt_ahead ** 2)

        # Floor bounce approximation
        if pred_y < floor_y:
            pred_y = floor_y + (floor_y - pred_y) * 0.65

        # Arena bounds clamping
        pred_x = max(sim.arena.x_left + 1.0, min(sim.arena.x_right - 1.0, pred_x))
        pred_y = max(floor_y, min(sim.arena.y_ceil - 1.0, pred_y))

        return (pred_x, pred_y)

    def determine_state(self, sim, car) -> str:
        """Evaluate match geometry to select the active behavioral state."""
        bx, by = sim.ball.position
        b_vel = sim.ball.velocity
        b_speed = math.hypot(b_vel[0], b_vel[1])
        cx, cy = car.position

        # 1. KICKOFF: Ball is stationary near center field
        if abs(bx - sim.center_x) < 0.5 and b_speed < 1.0 and abs(by - sim.ball_spawn_y) < 0.2:
            return "KICKOFF"

        if self.team == "orange":
            # 1. In defensive half, DEFEND takes top priority!
            if bx > BOT_DEFENSE_ZONE_X:
                return "DEFEND"
            # 2. In opponent half (attacking third), never rotate back; stay on attack to score
            if bx < 12.0:
                if by > 4.0 and car.boost_amount > BOT_AERIAL_MIN_BOOST and car.both_wheels_grounded:
                    return "AERIAL"
                return "ATTACK"
            # 3. In midfield, rotate back if caught on wrong side
            if cx <= bx - 0.2:
                return "ROTATE_BACK"
        else:
            # Blue defends Left goal
            if bx < (sim.arena.x_left + 9.5):
                return "DEFEND"
            if bx > (sim.arena.x_right - 12.0):
                if by > 4.0 and car.boost_amount > BOT_AERIAL_MIN_BOOST and car.both_wheels_grounded:
                    return "AERIAL"
                return "ATTACK"
            if cx >= bx + 0.2:
                return "ROTATE_BACK"

        # 4. AERIAL vs ATTACK:
        if by > 4.0 and car.boost_amount > BOT_AERIAL_MIN_BOOST and car.both_wheels_grounded:
            return "AERIAL"

        return "ATTACK"

    def compute_action(self, sim) -> CarAction:
        """Generate the control action (steering, throttle, jump, boost) for the bot car."""
        car = sim.car_orange if self.team == "orange" else sim.car
        if car is None:
            return CarAction()

        action = CarAction()
        self.current_state = self.determine_state(sim, car)

        cx, cy = car.position
        bx, by = sim.ball.position
        b_vx, b_vy = sim.ball.velocity
        self._jump_cooldown = max(0.0, self._jump_cooldown - (1.0 / 60.0))

        goal_target_x = sim.arena.x_left if self.team == "orange" else sim.arena.x_right
        own_goal_x = sim.arena.x_right if self.team == "orange" else sim.arena.x_left
        goal_target_y = (sim.arena.goal_y_bot + sim.arena.goal_y_top) * 0.5

        # -------------------------------------------------------------
        # STATE 1: KICKOFF
        # -------------------------------------------------------------
        if self.current_state == "KICKOFF":
            dx = bx - cx
            dist_to_ball = abs(dx)
            # Full throttle toward center
            action.dir_x = -1.0 if self.team == "orange" else 1.0
            action.dir_y = 0.0
            action.boost = True

            # When closing in on kickoff ball, execute a power dodge flip
            if dist_to_ball < 3.2 and car.has_jump2:
                if car.both_wheels_grounded and self._jump_cooldown <= 0.0:
                    action.jump = True
                    self._jump_cooldown = 0.6
                elif not car.both_wheels_grounded and car.has_jump2:
                    action.jump = True
                    action.dir_x = -1.0 if self.team == "orange" else 1.0
            return action

        # -------------------------------------------------------------
        # STATE 2: ROTATE_BACK (Shadow Defense / Anti-Own-Goal)
        # -------------------------------------------------------------
        if self.current_state == "ROTATE_BACK":
            # Dynamic safe target behind the ball: at least 3.0m behind ball or at defensive post
            if self.team == "orange":
                safe_target_x = min(sim.arena.x_right - 2.0, max(bx + 3.0, own_goal_x - 3.5))
            else:
                safe_target_x = max(sim.arena.x_left + 2.0, min(bx - 3.0, own_goal_x + 3.5))

            dx = safe_target_x - cx
            if abs(dx) > 0.4:
                action.dir_x = 1.0 if dx > 0 else -1.0
            else:
                # Reached defensive position: immediately face toward the ball / opponent net!
                action.dir_x = -1.0 if self.team == "orange" else 1.0

            action.dir_y = 0.0

            # Boost to recover quickly if far out of position
            if abs(dx) > 5.0 and car.boost_amount > 25.0:
                action.boost = True

            return action

        # -------------------------------------------------------------
        # STATE 3: DEFEND (Goalie Clearance in Defensive Zone)
        # -------------------------------------------------------------
        if self.current_state == "DEFEND":
            # Intercept ball before it enters goal
            lead_time = min(0.5, max(0.1, abs(cx - bx) / (CAR_MAX_GROUND_SPEED + 0.1)))
            pred_bx, pred_by = self.predict_ball_position(sim, lead_time)

            dx = pred_bx - cx
            dy = pred_by - cy
            dist = math.hypot(dx, dy)

            if self.team == "orange":
                # Orange defends Right goal (x = 30.5).
                # If Orange is to the left of the ball (cx < pred_bx - 0.2), drive RIGHT (+1.0) toward ball.
                # If Orange is to the right of the ball (cx >= pred_bx - 0.2), drive LEFT (-1.0) to clear.
                clear_dir_x = 1.0 if dx > 0.2 else -1.0
            else:
                # Blue defends Left goal (x = 4.5).
                # If Blue is to the right of the ball (cx > pred_bx + 0.2), drive LEFT (-1.0) toward ball.
                # If Blue is to the left of the ball (cx <= pred_bx + 0.2), drive RIGHT (+1.0) to clear.
                clear_dir_x = -1.0 if dx < -0.2 else 1.0

            # High shot: execute aerial save
            if pred_by > 3.0 and car.boost_amount > 10.0:
                action.dir_x = clear_dir_x
                action.dir_y = 0.8
                action.boost = True
                if car.both_wheels_grounded and self._jump_cooldown <= 0.0:
                    action.jump = True
                    self._jump_cooldown = 0.6
                elif car.has_jump2 and dist < 2.0:
                    action.jump = True  # Dodge clear
            else:
                # Ground clearance toward opponent net
                action.dir_x = clear_dir_x
                action.dir_y = 0.0
                if dist < 1.5 and pred_by > 2.7 and car.both_wheels_grounded and self._jump_cooldown <= 0.0:
                    action.jump = True
                    self._jump_cooldown = 0.6

            return action

        # -------------------------------------------------------------
        # STATE 4: AERIAL (High Ball Challenge)
        # -------------------------------------------------------------
        if self.current_state == "AERIAL":
            pred_bx, pred_by = self.predict_ball_position(sim, 0.4)
            dx = pred_bx - cx
            dy = pred_by - cy
            angle_to_ball = math.atan2(dy, dx)

            action.dir_x = math.cos(angle_to_ball)
            action.dir_y = math.sin(angle_to_ball)

            if car.both_wheels_grounded and self._jump_cooldown <= 0.0:
                action.jump = True
                self._jump_cooldown = 0.6
            else:
                action.boost = True
                if math.hypot(dx, dy) < 2.0 and car.has_jump2:
                    action.jump = True  # Dodge strike

            return action

        # -------------------------------------------------------------
        # STATE 5: ATTACK (Offensive Strike & Power Shot)
        # -------------------------------------------------------------
        lead_time = 0.15
        pred_bx, pred_by = self.predict_ball_position(sim, lead_time)
        dx_to_ball = pred_bx - cx
        dist_to_ball = math.hypot(dx_to_ball, pred_by - cy)

        if self.team == "orange":
            # Orange shoots toward Blue net (Left, -X)
            if cx >= pred_bx - 0.25:
                # Behind the ball, drive left directly into the ball toward the net
                action.dir_x = -1.0
            else:
                # Trapped between ball and Blue net, loop around behind to the right
                action.dir_x = 1.0
        else:
            # Blue shoots toward Orange net (Right, +X)
            if cx <= pred_bx + 0.25:
                action.dir_x = 1.0
            else:
                action.dir_x = -1.0

        action.dir_y = 0.0

        # Elevated ball jump strike or fast power dodge
        if pred_by > 2.8 and dist_to_ball < 2.2 and car.both_wheels_grounded and self._jump_cooldown <= 0.0:
            action.dir_y = 0.3
            action.jump = True
            self._jump_cooldown = 0.8
        elif dist_to_ball < 1.4 and abs(car.velocity[0]) > 6.0 and car.both_wheels_grounded and self._jump_cooldown <= 0.0:
            action.jump = True
            self._jump_cooldown = 0.8

        # Boost when closing in from distance
        if dist_to_ball > 4.0 and abs(cx - pred_bx) > 3.0 and car.boost_amount > 40.0:
            action.boost = True

        return action
