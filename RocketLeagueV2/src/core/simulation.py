"""Headless simulation manager orchestrating Pymunk space, entities, and collisions."""
import math
from typing import Dict, Any, Optional, Tuple
import pymunk
from src.config import (
    FIELD_WIDTH, FIELD_HEIGHT, MARGIN_X, MARGIN_Y,
    GRAVITY, SIM_HZ, PHYSICS_SUBSTEPS, SOLVER_ITERATIONS,
    ARENA_SEGMENT_RADIUS, CAR_RIDE_HEIGHT,
    BALL_RADIUS, CAR_SPAWN_X_DEFENSIVE, CAR_SPAWN_X_ATTACK,
    CAR_BALL_RESTITUTION,
    COLLISION_CAR_BODY, COLLISION_BALL, COLLISION_GOAL_SENSOR,
    ENABLE_ORANGE_CAR, ORANGE_IS_BOT
)
from src.core.actions import CarAction
from src.core.arena import Arena
from src.core.ball import Ball
from src.core.car import Car
from src.ai.heuristic_bot import HeuristicBot


class Simulation:
    """Headless 2D physics simulation environment for Rocket League."""

    def __init__(self, enable_orange: bool = ENABLE_ORANGE_CAR, orange_is_bot: bool = ORANGE_IS_BOT):
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
        self.goal_scored_this_step: Optional[str] = None
        self._spawn_index: int = 0
        self.time_elapsed: float = 0.0
        self.match_time: float = 0.0

        # Center coordinates
        self.center_x = MARGIN_X + FIELD_WIDTH / 2.0
        self.center_y = MARGIN_Y + FIELD_HEIGHT / 2.0

        # Instantiate entities
        self.arena = Arena(self.space)
        # 1. Ball spawns resting on the ground at kickoff
        self.ball = Ball(self.space, x=self.center_x, y=self.ball_spawn_y)
        # 2. Blue Car spawns on its side at initial kickoff position
        init_x, init_y, init_ang, init_fac = self.get_spawn_position("blue")
        self.car = Car(self.space, x=init_x, y=init_y, angle=init_ang, team="blue")
        self.car.reset(init_x, init_y, angle=init_ang, facing_x=init_fac)

        # 3. Orange Car and AI Bot (optional, toggleable)
        self.enable_orange: bool = enable_orange
        self.orange_is_bot: bool = orange_is_bot
        self.car_orange: Optional[Car] = None
        self.orange_bot: Optional[HeuristicBot] = None

        if self.enable_orange:
            ox, oy, oang, ofac = self.get_spawn_position("orange")
            self.car_orange = Car(self.space, x=ox, y=oy, angle=oang, team="orange")
            self.car_orange.reset(ox, oy, angle=oang, facing_x=ofac)
            if self.orange_is_bot:
                self.orange_bot = HeuristicBot(team="orange")

        self._spawn_index += 1

        self._setup_collision_handlers()

    def set_orange_enabled(self, enabled: bool, is_bot: bool = True):
        """Enable or disable the Orange opponent car and AI bot dynamically."""
        if enabled == self.enable_orange:
            return

        self.enable_orange = enabled
        self.orange_is_bot = is_bot

        if enabled:
            if self.car_orange is None:
                ox = 2.0 * self.center_x - self.car.position[0]
                oy = self.spawn_y
                self.car_orange = Car(self.space, x=ox, y=oy, angle=math.pi, team="orange")
                self.car_orange.reset(ox, oy, angle=math.pi, facing_x=-1)
            if self.orange_is_bot and self.orange_bot is None:
                self.orange_bot = HeuristicBot(team="orange")
        else:
            if self.car_orange is not None:
                self.space.remove(self.car_orange.chassis_shape, self.car_orange.body)
                self.car_orange = None
                self.orange_bot = None

    @property
    def spawn_y(self) -> float:
        """Floor height that puts the car exactly at its settled ride height."""
        return MARGIN_Y + ARENA_SEGMENT_RADIUS + CAR_RIDE_HEIGHT

    @property
    def ball_spawn_y(self) -> float:
        """Floor elevation that places the ball resting on the arena floor."""
        return MARGIN_Y + ARENA_SEGMENT_RADIUS + BALL_RADIUS

    def get_spawn_position(self, team: Optional[str] = None, spawn_index: Optional[int] = None) -> Tuple[float, float, float, int]:
        """Get the kickoff spawn (x, y, angle, facing_x) on the team's side for the kickoff round."""
        team = team or self.car.team
        positions = [CAR_SPAWN_X_DEFENSIVE, CAR_SPAWN_X_ATTACK]
        idx = self._spawn_index if spawn_index is None else spawn_index
        x_blue = positions[idx % len(positions)]

        if team == "orange":
            # Exact symmetrical position on Orange side, facing Left (angle = pi)
            x_orange = 2.0 * self.center_x - x_blue
            return (x_orange, self.spawn_y, math.pi, -1)
        else:
            return (x_blue, self.spawn_y, 0.0, 1)

    def _setup_collision_handlers(self):
        """Configure contact listeners for car/ball strikes and goal detection."""
        # Car Body <-> Ball: momentum transfer
        h_car_ball = self.space.add_collision_handler(COLLISION_CAR_BODY, COLLISION_BALL)

        def _car_ball_pre_solve(arbiter, space, data):
            arbiter.restitution = CAR_BALL_RESTITUTION
            return True

        h_car_ball.pre_solve = _car_ball_pre_solve

        # Car Body <-> Car Body: two cars collide and bounce
        h_car_car = self.space.add_collision_handler(COLLISION_CAR_BODY, COLLISION_CAR_BODY)

        def _car_car_pre_solve(arbiter, space, data):
            arbiter.restitution = 0.70
            return True

        h_car_car.pre_solve = _car_car_pre_solve

        # Ball <-> Goal Sensor: passive pass-through
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

    def step(self, action: CarAction, dt: float = 1.0 / SIM_HZ, action_orange: Optional[CarAction] = None):
        """Advance the physics simulation by dt using sub-stepping for stability."""
        self.goal_scored_this_step = None

        if self.car_orange is not None and action_orange is None and self.orange_bot is not None:
            action_orange = self.orange_bot.compute_action(self)

        sub_dt = dt / PHYSICS_SUBSTEPS
        for _ in range(PHYSICS_SUBSTEPS):
            self.car.update(action, sub_dt)
            if self.car_orange is not None:
                self.car_orange.update(action_orange or CarAction(), sub_dt)
            self.ball.apply_aerodynamics(sub_dt)
            self.space.step(sub_dt)

        self.ball.record_trail()
        self.time_elapsed += dt
        self.match_time += dt

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

    def reset(self, reset_scores: bool = False, spawn_pos: Optional[Tuple[float, float, float, int]] = None):
        """Reset the arena, ball, and car to initial kickoff conditions."""
        if reset_scores:
            self.score_blue = 0
            self.score_orange = 0
            self.last_goal_team = None
            self.goal_scored_this_step = None
            self.match_time = 0.0

        # 1. Ball spawns resting on the ground at center
        self.ball.reset(self.center_x, self.ball_spawn_y)

        # 2. Blue Car spawns at kickoff position for current round
        if spawn_pos is None:
            x, y, angle, facing = self.get_spawn_position("blue")
        else:
            x, y, angle, facing = spawn_pos

        self.car.reset(x, y, angle=angle, facing_x=facing)

        # 3. Orange Car spawns at the exact symmetrical position mirrored across center_x
        if self.car_orange is not None:
            if spawn_pos is None:
                ox, oy, oang, ofacing = self.get_spawn_position("orange")
            else:
                ox = 2.0 * self.center_x - x
                oy = y
                oang = math.pi
                ofacing = -1
            self.car_orange.reset(ox, oy, angle=oang, facing_x=ofacing)

        # Reset bot if active
        if self.orange_bot is not None:
            self.orange_bot.reset()

        # Advance kickoff spawn index for the next kickoff round
        self._spawn_index += 1

        self.time_elapsed = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Query complete simulation state for headless evaluation or RL."""
        state = {
            "time": self.time_elapsed,
            "match_time": self.match_time,
            "goal_scored_step": self.goal_scored_this_step,
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
                "both_wheels_grounded": self.car.both_wheels_grounded,
                "has_jump2": self.car.has_jump2,
                "is_flipping": self.car._flip_active,
                "boost": self.car.boost_amount,
                "is_boosting": self.car.is_boosting,
                "input_vector": self.car.last_input_vector,
                "facing_x": self.car.facing_x
            },
            "score": {
                "blue": self.score_blue,
                "orange": self.score_orange
            },
            "last_goal": self.last_goal_team,
            "orange_enabled": self.enable_orange
        }

        if self.car_orange is not None:
            state["car_orange"] = {
                "position": self.car_orange.position,
                "nose_position": self.car_orange.nose_position,
                "velocity": self.car_orange.velocity,
                "angle": self.car_orange.angle,
                "angular_velocity": self.car_orange.body.angular_velocity,
                "is_grounded": self.car_orange.is_grounded,
                "ground_normal": self.car_orange.ground_normal,
                "wheel_contacts": self.car_orange.wheel_contact_count,
                "both_wheels_grounded": self.car_orange.both_wheels_grounded,
                "has_jump2": self.car_orange.has_jump2,
                "is_flipping": self.car_orange._flip_active,
                "boost": self.car_orange.boost_amount,
                "is_boosting": self.car_orange.is_boosting,
                "input_vector": self.car_orange.last_input_vector,
                "facing_x": self.car_orange.facing_x,
                "bot_state": self.orange_bot.current_state if self.orange_bot else None
            }

        return state
