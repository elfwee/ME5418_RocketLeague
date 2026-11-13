"""Hierarchical Zone & Intercept State Machine (HZISM) Heuristic Bot for 2D Rocket League."""

import math

from src.config.sim_config import BOT_AERIAL_MIN_BOOST, CAR_MAX_GROUND_SPEED, GRAVITY_MAG, SIM_HZ
from src.sim.core.actions import CarAction


class HeuristicBot:
    """Rule-based AI bot with trajectory lookahead, shadow defense, and dynamic aerial/ground mechanics."""

    def __init__(self, team: str = "orange"):
        self.team = team
        self.current_state = "KICKOFF"
        self._is_kickoff = True
        self._jump_cooldown = 0.0
        self._jump_press_timer = 0
        self._last_dodge_time = 0.0
        self._stuck_timer = 0.0
        self._unstick_active = 0.0
        self._prev_jump_sent = False

    def reset(self):
        """Reset internal timers and behavioral state upon kickoff."""
        self.current_state = "KICKOFF"
        self._is_kickoff = True
        self._jump_cooldown = 0.0
        self._jump_press_timer = 0
        self._last_dodge_time = 0.0
        self._stuck_timer = 0.0
        self._unstick_active = 0.0
        self._prev_jump_sent = False

    @property
    def _attack_dir(self) -> float:
        """World +X direction toward the opponent's goal (+1 for Blue, -1 for Orange)."""
        return 1.0 if self.team == "blue" else -1.0

    def _to_team_d(self, sim, world_x: float) -> float:
        """Convert world X coordinate to distance from own goal line (0 = own goal, field_w = opp goal)."""
        return (world_x - sim.arena.x_left) if self.team == "blue" else (sim.arena.x_right - world_x)

    def _finalize(self, action: CarAction) -> CarAction:
        """Record jump edge-state and return action."""
        self._prev_jump_sent = bool(action.jump)
        return action

    def _request_jump(self, action: CarAction, car, fwd_cmd: float, dir_y_air: float = 0.0) -> bool:
        """Trigger an edge-detected jump while keeping grounded pitch flat to avoid wheelie slowdowns.

        Returns True when Jump 1 or Jump 2 is freshly pressed on this frame.
        """
        action.dir_x = fwd_cmd * self._attack_dir
        # Only pitch in midair; setting |dir_y| > 0.35 while grounded induces a wheelie (10% speed cap).
        action.dir_y = 0.0 if car.both_wheels_grounded else dir_y_air
        if self._prev_jump_sent:
            action.jump = False
            return False
        action.jump = True
        return True

    def predict_ball_position(self, sim, dt_ahead: float) -> tuple[float, float]:
        """Predict ball position dt_ahead seconds into the future assuming ballistic trajectory and floor bounce."""
        bx, by = sim.ball.position
        vx, vy = sim.ball.velocity
        floor_y = sim.ball_spawn_y

        pred_x = bx + vx * dt_ahead
        pred_y = by + vy * dt_ahead - 0.5 * GRAVITY_MAG * (dt_ahead**2)

        # Floor bounce approximation: inverts penetration below floor scaled by restitution factor e = 0.65
        if pred_y < floor_y:
            pred_y = floor_y + (floor_y - pred_y) * 0.65

        # Arena bounds clamping
        pred_x = max(sim.arena.x_left + 1.0, min(sim.arena.x_right - 1.0, pred_x))
        pred_y = max(floor_y, min(sim.arena.y_ceil - 1.0, pred_y))

        return (pred_x, pred_y)

    def determine_state(self, sim, car) -> str:
        """Evaluate match geometry to select the active behavioral state."""
        bx, by = sim.ball.position
        b_vx, b_vy = sim.ball.velocity
        b_speed = math.hypot(b_vx, b_vy)
        cx, _ = car.position

        car_d = self._to_team_d(sim, cx)
        ball_d = self._to_team_d(sim, bx)
        field_w = sim.arena.x_right - sim.arena.x_left

        # Kickoff tracking: true kickoff ends once ball moves from center, car overshoots, or mirror-stalls
        if self._is_kickoff:
            if (
                abs(bx - sim.center_x) > 0.6
                or b_speed > 1.2
                or (by - sim.ball_spawn_y) > 0.4
                or sim.time_elapsed > 1.8
                or car_d > ball_d + 0.5
            ):
                self._is_kickoff = False

        if self._is_kickoff:
            return "KICKOFF"

        def_zone_depth = 8.5
        att_zone_depth = 7.5
        aerial_def_depth = 6.5

        look_bx, _ = self.predict_ball_position(sim, 0.15)
        look_d = self._to_team_d(sim, look_bx)
        v_fwd_ball = b_vx * self._attack_dir

        # AERIAL Persistence: continue aerial approach while ball stays high in front of car
        if self.current_state == "AERIAL" and not car.both_wheels_grounded:
            if car.boost_amount > 5.0 and by > 3.5 and car_d < ball_d + 0.2:
                if ball_d < aerial_def_depth:
                    return "DEFEND"
                return "AERIAL"

        # 1. Defensive third or fast shot rushing own goal
        if ball_d < def_zone_depth or (look_d < 7.0 and v_fwd_ball < -3.5):
            return "DEFEND"

        # 2. Attacking pocket: keep pressure unless far out of position
        if ball_d > (field_w - att_zone_depth):
            if car_d >= ball_d + 1.5:
                return "ROTATE_BACK"
            if by > 3.5 and car.boost_amount > BOT_AERIAL_MIN_BOOST and car.both_wheels_grounded:
                return "AERIAL"
            return "ATTACK"

        # 3. Midfield: recover goal-side when caught ahead of current or immediate predicted ball position
        if car_d >= ball_d + 0.2 or (v_fwd_ball < -4.0 and car_d >= look_d + 0.2):
            return "ROTATE_BACK"

        # 4. Midfield aerial vs ground attack
        if by > 3.5 and car.boost_amount > BOT_AERIAL_MIN_BOOST and car.both_wheels_grounded:
            return "AERIAL"

        return "ATTACK"

    def _recover_behind_ball(
        self,
        action: CarAction,
        car,
        car_d: float,
        ball_d: float,
        pred_d: float,
        by: float,
        pred_by: float,
        v_fwd_ball: float,
        in_def_corner: bool = False,
    ) -> CarAction:
        """Return to the goal-side of the ball without driving it toward our own corner/goal."""
        _, cy = car.position
        _, cvy = car.velocity
        # Use min(actual_gap, pred_gap) so fast retreating balls don't mask immediate rear-end collisions.
        actual_gap = max(0.0, car_d - ball_d)
        pred_gap = max(0.0, car_d - pred_d)
        gap = min(actual_gap, pred_gap)

        # Ball is high enough that a flat grounded car clears cleanly underneath it
        if by >= 3.55 and pred_by >= 3.45:
            action.dir_x = -1.0 * self._attack_dir
            action.dir_y = 0.0
            action.jump = False
            action.boost = (pred_gap > 3.2 or v_fwd_ball < -3.0) and car.boost_amount > 15.0
            return self._finalize(action)

        # Deep own corner: never drive the ball along the floor up the curved corner ramp into own net
        if in_def_corner or pred_d < 3.4:
            if v_fwd_ball > 1.5:
                # Rebound already clearing toward midfield: hold outside to strike next touch
                action.dir_x = (-1.0 if car_d > 5.0 else 1.0) * self._attack_dir
                action.dir_y = 0.0
                action.jump = False
                action.boost = False
                return self._finalize(action)

            if gap > 3.8:
                action.dir_x = -1.0 * self._attack_dir
                action.dir_y = 0.0
                action.boost = gap > 6.5 and car.boost_amount > 25.0
                return self._finalize(action)

            # Vault cleanly over the corner ball without skyrocketing past the goal mouth
            if car.can_ground_jump:
                if self._jump_cooldown <= 0.0:
                    self._request_jump(action, car, -0.75, 0.0)
                    self._jump_cooldown = 0.35
                else:
                    action.dir_x = -0.5 * self._attack_dir
                    action.dir_y = 0.0
            else:
                if cy < by + 0.9 and cvy < 3.0 and car.has_jump2 and not self._prev_jump_sent:
                    # Type 1 neutral double-jump launches straight upward along up_vector
                    action.dir_x = 0.0
                    action.dir_y = 0.0
                    action.jump = True
                    action.boost = False
                elif cy > by + 1.6:
                    # Already high enough to clear the ball: drop onto corner goal-side
                    action.dir_x = -0.5 * self._attack_dir
                    action.dir_y = -0.35
                    action.boost = False
                else:
                    action.dir_x = -0.75 * self._attack_dir
                    action.dir_y = 0.25
                    action.boost = car.boost_amount > 5.0 and cy < by + 1.2 and cvy < 6.0
            return self._finalize(action)

        # Open field / mid-defensive recovery: sprint back, then jump cleanly over low ball when close
        fast_retreat = v_fwd_ball < -5.0
        jump_trigger_gap = 4.6 if fast_retreat else 3.4

        if gap <= jump_trigger_gap:
            if car.can_ground_jump:
                if self._jump_cooldown <= 0.0:
                    self._request_jump(action, car, -0.85 if fast_retreat else -1.0, 0.0)
                    self._jump_cooldown = 0.40
                else:
                    action.dir_x = -0.85 * self._attack_dir
                    action.dir_y = 0.0
            else:
                if cy < by + 1.1 and cvy < 4.0 and car.has_jump2 and not self._prev_jump_sent:
                    action.dir_x = 0.0
                    action.dir_y = 0.0
                    action.jump = True
                else:
                    action.dir_x = -1.0 * self._attack_dir
                    action.dir_y = 0.20 if cy < by + 1.4 else -0.20
                    action.boost = fast_retreat and car.boost_amount > 10.0 and cy < by + 1.3 and cvy < 7.0
            return self._finalize(action)

        action.dir_x = -1.0 * self._attack_dir
        action.dir_y = 0.0
        action.boost = gap > 5.0 and car.boost_amount > 25.0
        return self._finalize(action)

    def _compute_unstick(self, sim, car, car_d: float, ball_d: float, field_w: float) -> CarAction:
        """Recover when blocked by a wall or stationary opponent while avoiding own-goal hits."""
        action = CarAction()
        if car_d < 3.0:
            fwd = 1.0
        elif car_d > field_w - 3.0:
            fwd = -1.0
        else:
            fwd = 1.0 if ball_d >= car_d else -1.0

        self._request_jump(action, car, fwd, 0.45)
        action.boost = car.boost_amount > 5.0
        return self._finalize(action)

    def _compute_kickoff(self, sim, car, car_d: float, ball_d: float) -> CarAction:
        """Fast kickoff approach with flat Jump 1 followed by directional dodge strike."""
        action = CarAction()
        dist_to_ball = abs(ball_d - car_d)

        # Break 1e-10 Bot-vs-Bot center mirror deadlocks cleanly after initial contact if ball never moved
        if sim.time_elapsed >= 1.15 and sim.time_elapsed < 1.35:
            total_goals = getattr(sim, "score_blue", 0) + getattr(sim, "score_orange", 0)
            hesitate_team = "blue" if (total_goals % 2 == 0) else "orange"
            if self.team == hesitate_team:
                action.dir_x = 0.35 * self._attack_dir
                action.dir_y = 0.0
                return self._finalize(action)

        fwd = 1.0 if ball_d >= car_d else -1.0
        action.dir_x = fwd * self._attack_dir
        action.dir_y = 0.0
        action.boost = True

        if dist_to_ball < 2.5 and car.has_jump2:
            if car.both_wheels_grounded and self._jump_cooldown <= 0.0:
                if self._request_jump(action, car, fwd, 0.0):
                    self._jump_cooldown = 0.50
                    self._jump_press_timer = 1
            elif not car.both_wheels_grounded and dist_to_ball < 2.0:
                if self._jump_press_timer == 1 or self._prev_jump_sent:
                    action.jump = False
                    self._jump_press_timer = 2
                else:
                    action.jump = True
                    action.dir_x = fwd * self._attack_dir
                    action.dir_y = 0.25
                    self._jump_press_timer = 0
        return self._finalize(action)

    def _compute_defend(
        self,
        sim,
        car,
        car_d: float,
        ball_d: float,
        by: float,
        v_fwd_ball: float,
        v_fwd_car: float,
    ) -> CarAction:
        """Defend own goal mouth: clear downfield, jump high for crossbar saves, or safely vault corner balls."""
        action = CarAction()
        _, cy = car.position
        lead_time = min(0.45, max(0.08, abs(car_d - ball_d) / (CAR_MAX_GROUND_SPEED + 0.1)))
        pred_bx, pred_by = self.predict_ball_position(sim, lead_time)
        pred_d = self._to_team_d(sim, pred_bx)
        dist = math.hypot(pred_d - car_d, pred_by - cy)

        # Wrong side of the ball in defense
        if car_d > pred_d + 0.4:
            # High shot threatening elevated own goal (goal_y_bot ~ 5.86): attempt emergency save/sprint under
            if pred_d < 4.5 and pred_by >= 3.8 and car_d < 8.5 and cy < pred_by + 0.8:
                if car_d > pred_d + 1.5 and by >= 3.65:
                    # Enough gap to sprint flat underneath onto the goal line
                    action.dir_x = -1.0 * self._attack_dir
                    action.dir_y = 0.0
                    action.boost = car.boost_amount > 5.0
                    return self._finalize(action)
                # Near goal mouth: hold near goal line (d ~ 0.6..1.0) and jump/boost vertically to block
                out_fwd = -0.30 if (v_fwd_ball < -2.0 and car_d > 0.9) else (0.20 if pred_d < 2.2 else -0.25)
                if car.can_ground_jump and self._jump_cooldown <= 0.0:
                    if self._request_jump(action, car, out_fwd, 0.0):
                        self._jump_cooldown = 0.35
                else:
                    action.dir_x = out_fwd * self._attack_dir
                    action.dir_y = 0.85
                    action.boost = car.boost_amount > 5.0
                    if car.has_jump2 and dist < 2.5 and not self._prev_jump_sent:
                        action.jump = True
                return self._finalize(action)

            return self._recover_behind_ball(
                action, car, car_d, ball_d, pred_d, by, pred_by, v_fwd_ball, in_def_corner=(pred_d < 3.4)
            )

        # Floating well above a low corner ball: drop onto ball instead of boosting over it into midfield
        if not car.is_grounded and by < 3.6 and pred_d < 4.0 and cy > by + 1.4:
            action.dir_x = (0.35 if car_d < ball_d - 0.5 else -0.35) * self._attack_dir
            action.dir_y = -0.65
            action.boost = False
            return self._finalize(action)

        # Goal-side clearance toward opponent net (+fwd)
        if pred_by > 3.2 and car.boost_amount > 8.0 and (cy < pred_by or not car.both_wheels_grounded):
            # Aerial save / high clearance
            dx = max(0.3, pred_d - car_d)
            dy = pred_by - cy
            angle = math.atan2(dy, dx)
            if car.can_ground_jump:
                if self._jump_cooldown <= 0.0 and self._request_jump(action, car, 1.0, 0.0):
                    self._jump_cooldown = 0.45
                else:
                    action.dir_x = 1.0 * self._attack_dir
                    action.dir_y = 0.0
                    action.boost = abs(pred_d - car_d) > 2.5
            else:
                action.dir_x = max(0.35, math.cos(angle)) * self._attack_dir
                action.dir_y = math.sin(angle)
                action.boost = True
                if car.has_jump2 and dist < 2.3 and not self._prev_jump_sent:
                    action.dir_x = 1.0 * self._attack_dir
                    action.dir_y = 0.25
                    action.jump = True
        else:
            # Ground / corner clearance downfield
            action.dir_x = 1.0 * self._attack_dir
            action.dir_y = 0.0
            action.boost = (pred_d - car_d) > 2.0 and car.boost_amount > 8.0
            if dist < 2.2 and (pred_by > 2.6 or v_fwd_car > 5.0):
                if car.both_wheels_grounded and self._jump_cooldown <= 0.0:
                    if self._request_jump(action, car, 1.0, 0.0):
                        self._jump_cooldown = 0.40
                elif not car.both_wheels_grounded and car.has_jump2 and not self._prev_jump_sent:
                    action.dir_y = 0.25
                    action.jump = True
        return self._finalize(action)

    def _compute_aerial(self, sim, car) -> CarAction:
        """Launch and boost onto the goal-scoring side of an airborne ball."""
        action = CarAction()
        cx, cy = car.position
        car_d = self._to_team_d(sim, cx)

        pred_bx, pred_by = self.predict_ball_position(sim, 0.35)
        pred_d = self._to_team_d(sim, pred_bx)
        # Target just behind ball center, and aim toward goal elevation if ball is above crossbar
        target_d = pred_d - 0.55
        target_y = (
            (sim.arena.goal_y_bot + sim.arena.goal_y_top) * 0.5
            if pred_by > sim.arena.goal_y_top - 0.4
            else pred_by - 0.1
        )

        dx = target_d - car_d
        dy = target_y - cy
        if dx < -0.3:
            # Overshot slightly behind ball: reverse cleanly in air without boosting backward into ball
            action.dir_x = -1.0 * self._attack_dir
            action.dir_y = 0.0 if car.both_wheels_grounded else 0.15
            action.boost = False
            return self._finalize(action)

        if car.both_wheels_grounded:
            if self._jump_cooldown <= 0.0 and self._request_jump(action, car, 1.0, 0.0):
                self._jump_cooldown = 0.50
            else:
                action.dir_x = 1.0 * self._attack_dir
                action.dir_y = 0.0
                action.boost = dx > 2.5 and car.boost_amount > 15.0
        else:
            angle = math.atan2(dy, max(0.2, dx))
            action.dir_x = math.cos(angle) * self._attack_dir
            action.dir_y = math.sin(angle)
            action.boost = car.boost_amount > 3.0
            if math.hypot(pred_d - car_d, pred_by - cy) < 2.4 and car.has_jump2 and not self._prev_jump_sent:
                action.dir_x = 1.0 * self._attack_dir
                action.dir_y = 0.25 if pred_by < sim.arena.goal_y_bot + 1.0 else -0.15
                action.jump = True
        return self._finalize(action)

    def _compute_attack(
        self,
        sim,
        car,
        car_d: float,
        ball_d: float,
        by: float,
        v_fwd_ball: float,
        v_fwd_car: float,
        field_w: float,
    ) -> CarAction:
        """Attack toward opponent's elevated goal, chipping ground balls upward and dodging cleanly."""
        action = CarAction()
        _, cy = car.position
        pred_bx, pred_by = self.predict_ball_position(sim, 0.15)
        pred_d = self._to_team_d(sim, pred_bx)
        dist_to_ball = math.hypot(pred_d - car_d, pred_by - cy)

        # If ahead of the ball (unless wedging a ball right on opponent goal mouth from above), recover behind it
        in_goal_pocket = pred_d > field_w - 2.0 and car_d > field_w - 2.2 and cy > by
        if car_d > pred_d + 0.5 and not in_goal_pocket:
            return self._recover_behind_ball(
                action, car, car_d, ball_d, pred_d, by, pred_by, v_fwd_ball, in_def_corner=False
            )

        action.dir_x = 1.0 * self._attack_dir
        action.dir_y = 0.0

        gap_fwd = pred_d - car_d
        in_att_zone = ball_d > (field_w - 8.5)
        closing_toward_car = v_fwd_ball < -1.0

        # Pop / chip shots into the elevated goal mouth (y = 5.86..11.14)
        should_pop = (
            (in_att_zone and dist_to_ball < (3.4 if closing_toward_car else 2.5))
            or (ball_d > 14.0 and 0.4 <= gap_fwd <= 2.3 and pred_by < 3.4 and v_fwd_car > 5.0)
            or (pred_by > 2.7 and dist_to_ball < 2.2)
            or (dist_to_ball < 1.5 and v_fwd_car > 6.5)
        )

        if should_pop:
            if car.can_ground_jump and self._jump_cooldown <= 0.0:
                if self._request_jump(action, car, 1.0, 0.0):
                    self._jump_cooldown = 0.35
            elif (
                not car.can_ground_jump
                and car.has_jump2
                and gap_fwd > -0.2
                and dist_to_ball < 2.5
                and not self._prev_jump_sent
            ):
                action.dir_x = 1.0 * self._attack_dir
                action.dir_y = 0.35
                action.jump = True
                action.boost = car.boost_amount > 5.0

        if gap_fwd > 2.2 and car.boost_amount > 8.0 and car.both_wheels_grounded:
            action.boost = True

        return self._finalize(action)

    def compute_action(self, sim) -> CarAction:
        """Generate the control action (steering, throttle, jump, boost) for the bot car."""
        car = sim.orange_car if self.team == "orange" else sim.blue_car
        if car is None:
            return CarAction()

        if sim.time_elapsed == 0.0:
            self.reset()

        self.current_state = self.determine_state(sim, car)

        cx, _ = car.position
        bx, by = sim.ball.position
        b_vx, _ = sim.ball.velocity
        self._jump_cooldown = max(0.0, self._jump_cooldown - (1.0 / SIM_HZ))

        car_d = self._to_team_d(sim, cx)
        ball_d = self._to_team_d(sim, bx)
        v_fwd_ball = b_vx * self._attack_dir
        v_fwd_car = car.velocity[0] * self._attack_dir
        field_w = sim.arena.x_right - sim.arena.x_left

        # Obstacle stagnation detector
        if self._unstick_active > 0.0:
            self._unstick_active = max(0.0, self._unstick_active - (1.0 / SIM_HZ))
            return self._compute_unstick(sim, car, car_d, ball_d, field_w)
        if (car.is_grounded or car.can_ground_jump or car.wheel_contact_count >= 1) and abs(car.velocity[0]) < 0.20:
            self._stuck_timer += 1.0 / SIM_HZ
            if self._stuck_timer > 0.5:
                self._stuck_timer = 0.0
                self._unstick_active = 0.35
                return self._compute_unstick(sim, car, car_d, ball_d, field_w)
        else:
            self._stuck_timer = max(0.0, self._stuck_timer - (2.0 / SIM_HZ))

        if self.current_state == "KICKOFF":
            return self._compute_kickoff(sim, car, car_d, ball_d)

        if self.current_state == "ROTATE_BACK":
            pred_bx, pred_by = self.predict_ball_position(sim, 0.25)
            pred_d = self._to_team_d(sim, pred_bx)
            action = CarAction()
            return self._recover_behind_ball(
                action, car, car_d, ball_d, pred_d, by, pred_by, v_fwd_ball, in_def_corner=(pred_d < 3.4)
            )

        if self.current_state == "DEFEND":
            return self._compute_defend(sim, car, car_d, ball_d, by, v_fwd_ball, v_fwd_car)

        if self.current_state == "AERIAL":
            return self._compute_aerial(sim, car)

        return self._compute_attack(sim, car, car_d, ball_d, by, v_fwd_ball, v_fwd_car, field_w)
