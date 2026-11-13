"""Gymnasium wrapper around the 2D Rocket League simulation.

The env always learns from the *ego* car's point of view. When the ego car is Orange, the
observation and the x component of the action are mirrored, so one policy can play either side.

`control_car` selects who the policy drives:
    "blue"   - policy drives Blue; Orange is absent or the heuristic bot (`bot_type`).
    "orange" - policy drives Orange (mirrored); Blue takes keyboard input.
    "both"   - policy drives both cars (self-play); the reward is Blue's.

`step()` advances one sim frame. Wrap the env in `wrappers.ActionRepeat` to hold each decision for
several frames (the agents use 4). Rewards are defined in `rewards.py`.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pygame

from src.config.gym_config import GymConfig
from src.config.render_config import SCREEN_HEIGHT, SCREEN_WIDTH
from src.config.sim_config import SIM_HZ
from src.gym_env import action_codec
from src.gym_env.rewards import ball_progress_reward, car_ball_distance, goal_reward, relative_team, shaping_reward
from src.sim.control import HeuristicController, HumanController
from src.sim.core.actions import CarAction
from src.sim.core.observation import OBS_DIM, build_observation, normalize_position
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


@dataclass(frozen=True)
class Frame:
    """One sim frame's input for `RocketLeagueEnv.step` when the caller needs more than an action.

    `build_obs=False` skips building the observation (a zero placeholder is returned instead);
    frame-skipping wrappers use it for every frame but the last.
    """

    action: object
    action_orange: object = None
    build_obs: bool = True


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
            self.orange_control = HeuristicController("orange", self.sim)
        elif control_car == "orange":
            self.blue_control = HumanController()

        if self.discrete_action:
            self.action_space = gym.spaces.Discrete(action_codec.NUM_DISCRETE_ACTIONS)
        else:
            self.action_space = gym.spaces.MultiDiscrete(list(action_codec.FACTORED_DIMS))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

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

    def get_observation(self, team: str | None = None) -> np.ndarray:
        """The 52-float observation from `team`'s point of view (default: the ego car's)."""
        return np.asarray(build_observation(self.sim, team or self._ego_team), dtype=np.float32)

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
            self.sim.rng.seed(seed)  # spawn sampling; no global RNG is touched

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
            info["obs_orange"] = self.get_observation("orange")
        return self.get_observation(), info

    def step(self, action, action_orange=None):
        """Advance one frame. In "both" mode pass Orange's action as `action_orange` or as `(blue, orange)`.

        `action` may also be a `Frame`, which carries both actions and whether to build the observation.
        """
        build_obs = True
        if isinstance(action, Frame):
            action, action_orange, build_obs = action.action, action.action_orange, action.build_obs
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
        if not build_obs:
            return np.zeros(OBS_DIM, dtype=np.float32), reward, terminated, truncated, info
        if self.control_car == "both":
            info["obs_orange"] = self.get_observation("orange")
        return self.get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ reward

    def _ego_car(self):
        if self._ego_team == "orange" and self.sim.orange_car is not None:
            return self.sim.orange_car
        return self.sim.blue_car

    def _calculate_reward(self):
        """Return (reward, terminated) for the ego team after the latest sim step (see `rewards.py`)."""
        last_touch = relative_team(self.sim.last_touch, self._ego_team)
        goal = relative_team(self.sim.goal_scored_this_step, self._ego_team)
        if goal is not None:
            self._reset_shaping_state()
            return goal_reward(goal, last_touch, self.cfg), True

        # Ball progress towards the opponent goal, in the ego frame (+x attacks)
        ball_x, ball_y = normalize_position(self.sim.ball.position)
        if self._ego_team == "orange":
            ball_x = -ball_x
        if self.ball_prev is None:
            self.ball_prev = ball_x
        delta_x = ball_x - self.ball_prev
        self.ball_prev = ball_x
        progress = ball_progress_reward(delta_x, ball_x, ball_y, last_touch, self.cfg)

        dist = car_ball_distance(self._ego_car().position, self.sim.ball.position)
        if self.dist_prev is None:
            self.dist_prev = dist
        reward = shaping_reward(progress, dist, self.dist_prev, self.cfg)
        self.dist_prev = dist
        return reward, False

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
