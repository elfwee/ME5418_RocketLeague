import os
import sys
import random

import gymnasium as gym
import numpy as np
import pygame

# Ensure repository root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.core.actions import CarAction
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
    metadata = {"render_modes": ["human"], "render_fps": SIM_HZ}

    def __init__(self, render_mode=None, bot_type="none", discrete_action=True):
        if bot_type not in ("none", "heuristic_bot"):
            raise ValueError(f"Invalid bot_type: {bot_type}. Must be 'none' or 'heuristic_bot'.")

        self.render_mode = render_mode
        self.sim = Simulation(enable_orange=(bot_type == "heuristic_bot"), random_spawn=True)
        self.ball_prev = None
        self.dist_prev = None
        self.input_prev = None
        self.isopen = True
        self.discrete_action = discrete_action

        # direction (9), jump (2), boost (2) = 36 discrete actions
        if self.discrete_action:
            self.action_space = gym.spaces.Discrete(36)
        else:
            self.action_space = gym.spaces.MultiDiscrete([9, 2, 2])
        obs_dim = len(self.sim.get_state_norm(as_flat_array=True))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Rocket League Sideswipe 2D")
            self.renderer = Renderer(pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)))
            self.clock = pygame.time.Clock()

    def _get_obs(self):
        return np.asarray(self.sim.get_state_norm(as_flat_array=True), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        random_spawn = options.get("random_spawn", None) if options else None
        self.sim.reset(reset_scores=True, random_spawn=random_spawn)
        self.ball_prev = None
        self.dist_prev = None
        self.input_prev = None
        self.render()
        return self._get_obs(), {}

    def step(self, action):
        if self.discrete_action:
            direction = action // 4
            jump = (action % 4) // 2
            boost = action % 2
        else:
            direction, jump, boost = action
        dir_x, dir_y = GymConfig.directions[direction]
        car_action = CarAction(dir_x=float(dir_x), dir_y=float(dir_y), jump=jump == 1, boost=boost == 1).clamp()

        self.sim.step(car_action, 1.0 / SIM_HZ)
        reward, terminated = self._calculate_reward()
        self.render()
        return self._get_obs(), reward, terminated, not self.isopen, {}

    def _calculate_reward(self):
        state = self.sim.get_state_norm()
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
        dist = abs(state["relations"]["agent_to_ball"]["magnitude"])
        if self.dist_prev is None:
            self.dist_prev = dist
        reward += (self.dist_prev - dist) * 1.5
        self.dist_prev = dist
        reward -= dist * 0.005
        reward -= 0.001

        return reward, False

    def render(self):
        if self.render_mode != "human" or not self.isopen:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.isopen = False
                return
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

    def close(self):
        self.isopen = False
        if self.render_mode == "human" and pygame.get_init():
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
