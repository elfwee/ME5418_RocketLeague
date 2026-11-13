import os
import sys
import random

import gymnasium as gym
import numpy as np
import pygame

# Ensure repository root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.core.actions import CarAction
from src.sim.core.control import CarControl
from src.sim.core.simulation import Simulation
from src.sim.visualization.renderer import Renderer
from src.config.sim_config import SCREEN_WIDTH, SCREEN_HEIGHT, SIM_HZ, ENABLE_ORANGE_CAR
from src.config.gym_config import GymConfig

# register env as gym environment
if GymConfig.gym_id not in gym.envs.registry:
    gym.register(
        id=GymConfig.gym_id,
        entry_point="src.gym_env.rocket_league_env:RocketLeagueEnv",
        max_episode_steps=GymConfig.episode_max_steps,
    )

# (dir_x, dir_y): none, right, left, down, up, down-right, up-right, down-left, up-left


class RocketLeagueEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": SIM_HZ}

    def __init__(self, render_mode=None, bot_type="none", discrete_action=True, control_car="blue"):
        if bot_type not in ("none", "heuristic_bot"):
            raise ValueError(f"Invalid bot_type: {bot_type}. Must be 'none' or 'heuristic_bot'.")
        if control_car in ("bot_vs_bot", "bots", "both"):
            control_car = "both"
        if control_car not in ("blue", "orange", "both"):
            raise ValueError(f"Invalid control_car: {control_car}. Must be 'blue', 'orange', or 'both'.")

        self.render_mode = render_mode
        self.bot_type = bot_type
        self.control_car = control_car

        enable_orange = (bot_type == "heuristic_bot") or (self.control_car in ("orange", "both"))
        self.sim = Simulation(enable_orange=enable_orange, random_spawn=True)

        if self.control_car == "blue":
            self.blue_control = None
            self.orange_control = (
                CarControl(control_type="heuristic_bot", team="orange", sim=self.sim)
                if bot_type == "heuristic_bot"
                else None
            )
        elif self.control_car == "orange":
            self.blue_control = CarControl(control_type="human", team="blue", sim=self.sim)
            self.orange_control = None
        else:
            # control_car == "both": both Blue and Orange are controlled by policy
            self.blue_control = None
            self.orange_control = None

        self.ball_prev = None
        self.dist_prev = None
        self.input_prev = None
        self._episode_count = 0
        self.isopen = True
        self.discrete_action = discrete_action

        # direction (9), jump (2), boost (2) = 36 discrete actions
        if self.discrete_action:
            self.action_space = gym.spaces.Discrete(36)
        else:
            self.action_space = gym.spaces.MultiDiscrete([9, 2, 2])
        obs_dim = len(self.sim.get_state_norm(as_flat_array=True))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.renderer = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Rocket League Sideswipe 2D")
            self.renderer = Renderer(pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)))
            self.clock = pygame.time.Clock()
        elif self.render_mode == "rgb_array":
            pygame.init()
            self.renderer = Renderer(pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)))
            self.clock = pygame.time.Clock()

    def _get_obs(self, team=None):
        target_team = (
            ("orange" if self.control_car == "orange" else "blue")
            if team is None
            else team
        )
        if target_team == "orange":
            return np.asarray(self.sim.get_state_norm_mirror(as_flat_array=True), dtype=np.float32)
        return np.asarray(self.sim.get_state_norm(as_flat_array=True), dtype=np.float32)

    def _decode_action(self, action, mirror_x: bool = False) -> CarAction:
        if isinstance(action, (list, tuple, np.ndarray)):
            direction, jump, boost = int(action[0]), int(action[1]), int(action[2])
        else:
            act_int = int(action)
            direction = act_int // 4
            jump = (act_int % 4) // 2
            boost = act_int % 2
        dir_x, dir_y = GymConfig.directions[direction]
        if mirror_x:
            dir_x = -dir_x
        return CarAction(dir_x=float(dir_x), dir_y=float(dir_y), jump=jump == 1, boost=boost == 1).clamp()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        random_spawn = options.get("random_spawn", None) if options else None
        reset_scores = (seed is not None) or (options.get("reset_scores", False) if options else False)
        if reset_scores:
            self._episode_count = 0
        self.sim.goal_scored_this_step = None
        self.sim.last_touch = None
        self.sim._spawn_index = self._episode_count
        self._episode_count += 1
        self.sim.reset(reset_scores=reset_scores, random_spawn=random_spawn)
        if self.blue_control is not None:
            self.blue_control.reset()
        if self.orange_control is not None:
            self.orange_control.reset()
        self.ball_prev = None
        self.dist_prev = None
        self.input_prev = None
        self.render()
        info = {}
        if self.control_car == "both":
            info["obs_orange"] = self._get_obs("orange")
        return self._get_obs(), info

    def step(self, action, action_orange=None):
        if self.control_car == "both":
            if action_orange is None and isinstance(action, tuple) and len(action) == 2:
                action, action_orange = action
            car_action = self._decode_action(action, mirror_x=False)
            act_orange = (
                self._decode_action(action_orange, mirror_x=True)
                if action_orange is not None
                else CarAction()
            )
        elif self.control_car == "orange":
            car_action = self.blue_control.get_action() if self.blue_control is not None else CarAction()
            act_orange = self._decode_action(action, mirror_x=True)
        else:
            car_action = self._decode_action(action, mirror_x=False)
            act_orange = (
                self.orange_control.get_action()
                if (self.orange_control is not None and self.sim.orange_car is not None)
                else CarAction()
            )

        self.sim.step(car_action, 1.0 / SIM_HZ, action_orange=act_orange)
        reward, terminated = self._calculate_reward()
        self.render()
        info = {
            "goal_scored": self.sim.goal_scored_this_step,
            "score": {"blue": self.sim.score_blue, "orange": self.sim.score_orange},
        }
        if self.control_car == "both":
            info["obs_orange"] = self._get_obs("orange")
        return self._get_obs(), reward, terminated, not self.isopen, info

    def _calculate_reward(self):
        state = self.sim.get_state_norm_mirror() if self.control_car == "orange" else self.sim.get_state_norm()
        last_touch = state["last_touch"]
        goal = state["goal_scored_step"]

        if goal is not None:
            self.ball_prev = None
            self.dist_prev = None
            self.input_prev = None
            if goal == "blue":
                return (2.0 if last_touch == "orange" else 20.0), True
            return -20.0, True

        reward = 0.0

        # ball progress towards opponent goal
        ball_x, ball_y = state["ball"]["position"][:2]
        if self.ball_prev is None:
            self.ball_prev = ball_x
        delta_x = ball_x - self.ball_prev
        if abs(delta_x) > 0.0001:
            # Non-linear weighting: x_factor exponentially amplifies rewards near the goal mouth (|ball_x| -> 1);
            # y_factor retains a 0.25 baseline for ground-rolling balls (ball_y <= -0.55) while rewarding elevated trajectories.
            x_factor = np.exp(1 / (1.2 - ball_x**2))
            y_factor = (1 - abs(ball_y)) ** 2 if ball_y > -0.55 else 0.25
            progress = delta_x * x_factor * y_factor
            if not ((progress <= 0.0 and last_touch == "blue") or (progress >= 0.0 and last_touch == "orange")):
                reward += progress
        self.ball_prev = ball_x

        # Potential-based car-to-ball approach shaping + distance and time penalty
        dist = abs(state["relations"]["blue_car_to_ball"]["magnitude"])
        if self.dist_prev is None:
            self.dist_prev = dist
        reward += (self.dist_prev - dist) * 1.5
        self.dist_prev = dist
        reward -= dist * 0.005
        reward -= 0.001

        return reward, False

    def render(self):
        if not self.isopen:
            return None
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.isopen = False
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_b:
                        self.sim.set_orange_enabled(not self.sim.enable_orange, is_bot=True)
                    elif event.key == pygame.K_l:
                        self.renderer.show_raycasts = not self.renderer.show_raycasts
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.renderer.handle_click(event.pos)
            self.renderer.render(self.sim)
            pygame.display.flip()
            self.clock.tick(SIM_HZ)
            return None
        elif self.render_mode == "rgb_array":
            return self.get_frame()
        return None

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
    my_check_env()
    bot_type = "heuristic_bot" if ENABLE_ORANGE_CAR else "none"
    env = gym.make(GymConfig.gym_id, render_mode="human", bot_type=bot_type)

    obs, info = env.reset()
    total_reward = 0.0

    step = 0
    while env.unwrapped.isopen:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        with np.printoptions(precision=2, suppress=True):
            print(f"step: {step}")
            step+=1
            print("Observed State: ", obs)

        if terminated or truncated:
            print(f"Episode reward: {total_reward:.4f}, terminated: {terminated}, truncated: {truncated}")
            obs, info = env.reset()
            total_reward = 0.0
            break

    env.close()
