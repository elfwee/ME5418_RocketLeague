"""Gymnasium wrapper around the 2D Rocket League simulation.

The env always learns from the *ego* car's point of view. When the ego car is Orange, the
observation and the x component of the action are mirrored, so one policy can play either side.

`control_car` selects who the policy drives:
    "blue"   - policy drives Blue; Orange is absent or the heuristic bot (`bot_type`).
    "orange" - policy drives Orange (mirrored); Blue takes keyboard input.
    "both"   - policy drives both cars (self-play); the reward is Blue's.
"""

import math

import gymnasium as gym
import numpy as np
import pygame

from src.config.gym_config import GymConfig
from src.config.sim_config import ARENA_DIAGONAL, ENABLE_ORANGE_CAR, SCREEN_HEIGHT, SCREEN_WIDTH, SIM_HZ
from src.gym_env import action_codec
from src.sim.core.actions import CarAction
from src.sim.core.control import CarControl
from src.sim.core.observation import normalize_position
from src.sim.core.simulation import Simulation
from src.sim.visualization.renderer import Renderer

# register env as gym environment
if GymConfig.gym_id not in gym.envs.registry:
    gym.register(
        id=GymConfig.gym_id,
        entry_point="src.gym_env.rocket_league_env:RocketLeagueEnv",
        max_episode_steps=GymConfig.episode_max_steps,
    )

BOT_TYPES = ("none", "heuristic_bot")
CONTROL_CARS = ("blue", "orange", "both")
CONTROL_CAR_ALIASES = {"bot_vs_bot": "both", "bots": "both"}


class RocketLeagueEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": SIM_HZ}

    def __init__(
        self,
        render_mode=None,
        bot_type="none",
        discrete_action=True,
        control_car="blue",
        random_spawn: bool = True,
        config: GymConfig | None = None,
    ):
        if bot_type not in BOT_TYPES:
            raise ValueError(f"Invalid bot_type: {bot_type}. Must be 'none' or 'heuristic_bot'.")
        control_car = CONTROL_CAR_ALIASES.get(control_car, control_car)
        if control_car not in CONTROL_CARS:
            raise ValueError(f"Invalid control_car: {control_car}. Must be 'blue', 'orange', or 'both'.")

        self.cfg = config or GymConfig()
        self.render_mode = render_mode
        self.bot_type = bot_type
        self.control_car = control_car
        self.discrete_action = discrete_action
        # Team whose perspective defines the default observation, the reward and `is_success`
        self._ego_team = "orange" if control_car == "orange" else "blue"

        enable_orange = (bot_type == "heuristic_bot") or (control_car in ("orange", "both"))
        self.sim = Simulation(enable_orange=enable_orange, random_spawn=random_spawn)

        # Scripted controllers for cars the policy does not drive (None = policy-driven or idle)
        self.blue_control = None
        self.orange_control = None
        if control_car == "blue" and bot_type == "heuristic_bot":
            self.orange_control = CarControl(control_type="heuristic_bot", team="orange", sim=self.sim)
        elif control_car == "orange":
            self.blue_control = CarControl(control_type="human", team="blue", sim=self.sim)

        if self.discrete_action:
            self.action_space = gym.spaces.Discrete(action_codec.NUM_DISCRETE_ACTIONS)
        else:
            self.action_space = gym.spaces.MultiDiscrete(list(action_codec.FACTORED_DIMS))
        obs_dim = len(self.sim.get_state_norm(as_flat_array=True))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # When `_defer_obs` is set (by frame-skipping callers that only need the last frame's
        # observation), step() skips building the observation and returns a placeholder instead.
        self._defer_obs = False
        self._dummy_obs = (np.zeros(obs_dim, dtype=np.float32), np.zeros(obs_dim, dtype=np.float32))
        self._dummy_idx = 0

        self._episode_count = 0
        self.isopen = True
        self._reset_shaping_state()
        self._init_rendering()

    # ------------------------------------------------------------------ setup helpers

    def _init_rendering(self):
        self.renderer = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Rocket League Sideswipe 2D")
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        elif self.render_mode == "rgb_array":
            pygame.init()
            screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            return
        self.renderer = Renderer(screen)
        self.clock = pygame.time.Clock()

    def _reset_shaping_state(self):
        """Forget the previous-step values used by the shaping rewards."""
        self.ball_prev = None
        self.dist_prev = None

    # ------------------------------------------------------------------ observations and actions

    def _get_obs(self, team=None):
        """Flat normalised observation from `team`'s perspective (default: the ego team)."""
        if (team or self._ego_team) == "orange":
            state = self.sim.get_state_norm_mirror(as_flat_array=True)
        else:
            state = self.sim.get_state_norm(as_flat_array=True)
        return np.asarray(state, dtype=np.float32)

    def _resolve_car_actions(self, action, action_orange=None):
        """Return the (blue, orange) CarActions for this step given the policy action(s)."""
        if self.control_car == "both":
            if action_orange is None and isinstance(action, tuple) and len(action) == 2:
                action, action_orange = action
            blue_action = action_codec.decode(action)
            orange_action = (
                action_codec.decode(action_orange, mirror_x=True) if action_orange is not None else CarAction()
            )
        elif self.control_car == "orange":
            blue_action = self.blue_control.get_action() if self.blue_control is not None else CarAction()
            orange_action = action_codec.decode(action, mirror_x=True)
        else:
            blue_action = action_codec.decode(action)
            bot_active = self.orange_control is not None and self.sim.orange_car is not None
            orange_action = self.orange_control.get_action() if bot_active else CarAction()
        return blue_action, orange_action

    # ------------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng.seed(seed)
            np.random.seed(seed)

        options = options or {}
        reset_scores = (seed is not None) or options.get("reset_scores", False)
        if reset_scores:
            self._episode_count = 0
        spawn_index = self._episode_count
        self._episode_count += 1
        self.sim.reset(
            reset_scores=reset_scores,
            spawn_index=spawn_index,
            random_spawn=options.get("random_spawn"),
            clear_events=True,
        )
        for control in (self.blue_control, self.orange_control):
            if control is not None:
                control.reset()
        self._reset_shaping_state()
        self.render()

        info = {}
        if self.control_car == "both":
            info["obs_orange"] = self._get_obs("orange")
        return self._get_obs(), info

    def step(self, action, action_orange=None):
        """Advance one frame. In "both" mode pass Orange's action as `action_orange` or as `(blue, orange)`."""
        blue_action, orange_action = self._resolve_car_actions(action, action_orange)
        self.sim.step(blue_action, 1.0 / SIM_HZ, action_orange=orange_action)
        reward, terminated = self._calculate_reward()
        self.render()

        goal_scored = self.sim.goal_scored_this_step
        info = {
            "goal_scored": goal_scored,
            "is_success": bool(goal_scored == self._ego_team),
            "score": {"blue": self.sim.score_blue, "orange": self.sim.score_orange},
        }
        truncated = not self.isopen  # window closed by the user
        if self._defer_obs and not terminated and self.isopen:
            self._dummy_idx ^= 1
            return self._dummy_obs[self._dummy_idx], reward, terminated, truncated, info
        if self.control_car == "both":
            info["obs_orange"] = self._get_obs("orange")
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ reward

    def _relative_team(self, team):
        """Map a sim team name to "ego" / "opponent" (None stays None)."""
        if team is None:
            return None
        return "ego" if team == self._ego_team else "opponent"

    def _calculate_reward(self):
        """Return (reward, terminated) for the ego team after the latest sim step."""
        last_touch = self._relative_team(self.sim.last_touch)
        goal = self._relative_team(self.sim.goal_scored_this_step)

        if goal is not None:
            self._reset_shaping_state()
            if goal == "ego":
                return (self.cfg.reward_opponent_own_goal if last_touch == "opponent" else self.cfg.reward_goal), True
            return self.cfg.reward_concede, True

        reward = 0.0
        reward += self._ball_progress_reward(last_touch)

        # Potential-based car-to-ball approach shaping + distance and time penalty
        dist = self._ego_ball_distance()
        if self.dist_prev is None:
            self.dist_prev = dist
        reward += (self.dist_prev - dist) * self.cfg.approach_reward_scale
        self.dist_prev = dist
        reward -= dist * self.cfg.distance_penalty_scale
        reward -= self.cfg.time_penalty
        return reward, False

    def _ball_progress_reward(self, last_touch):
        """Reward ball movement towards the opponent goal (+x in the ego frame)."""
        ball_x, ball_y = normalize_position(self.sim.ball.position)
        if self._ego_team == "orange":
            ball_x = -ball_x
        if self.ball_prev is None:
            self.ball_prev = ball_x
        delta_x = ball_x - self.ball_prev
        self.ball_prev = ball_x
        if abs(delta_x) <= self.cfg.ball_progress_deadband:
            return 0.0

        # Non-linear weighting: x_factor exponentially amplifies rewards near the goal mouth (|ball_x| -> 1);
        # y_factor retains a 0.25 baseline for ground-rolling balls (ball_y <= -0.55) while rewarding elevated trajectories.
        x_factor = np.exp(1 / (1.2 - ball_x**2))
        y_factor = (1 - abs(ball_y)) ** 2 if ball_y > self.cfg.ground_ball_y else self.cfg.ground_ball_y_factor
        progress = delta_x * x_factor * y_factor

        # Only credit what the last toucher caused: no penalty for the ego car when its own touch
        # sends the ball backwards, and no reward when the opponent's touch sends it forwards.
        if (progress <= 0.0 and last_touch == "ego") or (progress >= 0.0 and last_touch == "opponent"):
            return 0.0
        return progress

    def _ego_ball_distance(self):
        """Distance from the ego car to the ball as a fraction of the arena diagonal, capped at 1."""
        ego_car = self.sim.blue_car
        if self._ego_team == "orange" and self.sim.orange_car is not None:
            ego_car = self.sim.orange_car
        ball_x, ball_y = self.sim.ball.position
        car_x, car_y = ego_car.position
        return min(1.0, math.hypot(ball_x - car_x, ball_y - car_y) / ARENA_DIAGONAL)

    # ------------------------------------------------------------------ rendering

    def render(self):
        if not self.isopen:
            return None
        if self.render_mode == "human":
            if not self._handle_window_events():
                return None
            self.renderer.render(self.sim)
            pygame.display.flip()
            self.clock.tick(SIM_HZ)
            return None
        if self.render_mode == "rgb_array":
            return self.get_frame()
        return None

    def _handle_window_events(self) -> bool:
        """Process pygame events for the human window. Returns False once the user closes it."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.isopen = False
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_b:
                    self.sim.set_orange_enabled(not self.sim.enable_orange)
                elif event.key == pygame.K_l:
                    self.renderer.show_raycasts = not self.renderer.show_raycasts
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.renderer.handle_click(event.pos)
        return True

    def get_frame(self):
        """Capture the current visual frame as an RGB numpy array (H, W, 3)."""
        if self.renderer is not None:
            self.renderer.render(self.sim)
            return np.transpose(pygame.surfarray.array3d(self.renderer.screen), (1, 0, 2))
        return None

    def close(self):
        self.isopen = False
        if self.render_mode in ("human", "rgb_array") and pygame.get_init():
            pygame.quit()


def my_check_env():
    from gymnasium.utils.env_checker import check_env

    env = gym.make(GymConfig.gym_id, render_mode=None, bot_type="none", disable_env_checker=True)
    check_env(env.unwrapped)
    print("Environment check passed successfully!")


def demo_gym():
    """Check the env, then play one episode of random actions in a window."""
    my_check_env()
    bot_type = "heuristic_bot" if ENABLE_ORANGE_CAR else "none"
    env = gym.make(GymConfig.gym_id, render_mode="human", bot_type=bot_type)

    obs, info = env.reset()
    total_reward = 0.0
    step = 0
    while env.unwrapped.isopen:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward

        with np.printoptions(precision=2, suppress=True):
            print(f"step: {step}")
            print("Observed State: ", obs)
        step += 1

        if terminated or truncated:
            print(f"Episode reward: {total_reward:.4f}, terminated: {terminated}, truncated: {truncated}")
            break

    env.close()
