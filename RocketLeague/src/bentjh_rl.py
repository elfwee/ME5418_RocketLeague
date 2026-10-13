# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
#
# import panda_gym
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import pygame
from src.config import (
    SIM_HZ,
    FIELD_WIDTH,
    FIELD_HEIGHT,
    CAR_MAX_SPEED,
    CAR_MAX_ANGULAR_SPEED,
    BALL_MAX_SPEED,
    BALL_MAX_SPIN,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SIM_HZ,
    ENABLE_ORANGE_CAR,
)
from src.core.actions import CarAction
from src.core.simulation import Simulation
from src.visualization.renderer import Renderer

import gymnasium as gym

# from gymnasium as spaces
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

local_config = {"rl_id": "bentjh01_2d_rocketleague_v1", "max_steps": 10000}

if local_config["rl_id"] not in gym.envs.registry:
    gym.register(
        id=local_config["rl_id"],
        entry_point="src.bentjh_rl:GymEnv",
        max_episode_steps=local_config["max_steps"],
        render_mode="human",
        render_fps=SIM_HZ, 
        bot_type="zeros", 
    )

# Define the wrapper to swap 'truncated' for 'terminated'
class TerminateOnTimeout(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If the 6000 step limit is hit, force terminated to True
        if truncated:
            terminated = True

        return obs, reward, terminated, truncated, info

def get_observation(state: dict) -> np.ndarray:
    """
    Extract pose and twist for:
        car, car_orange, ball

    Returns:
        np.ndarray with shape (14,)
    """
    obs = []

    for name in ("car", "car_orange", "ball"):
        obj = state[name]

        # Pose: x, y, theta
        obs.extend(
            [
                obj["position"][0] / FIELD_WIDTH,  # [m]
                obj["position"][1] / FIELD_HEIGHT,  # [m]
                obj["angle"] / (np.pi),  # [rad]
            ]
        )

        # Twist: vx, vy, omega
        if "car" in name:
            obs.extend(
                [
                    obj["velocity"][0] / CAR_MAX_SPEED,
                    obj["velocity"][1] / CAR_MAX_SPEED,
                    obj["angular_velocity"] / CAR_MAX_ANGULAR_SPEED,
                ]
            )
        elif "ball" in name:
            obs.extend(
                [
                    obj["velocity"][0] / BALL_MAX_SPEED,
                    obj["velocity"][1] / BALL_MAX_SPEED,
                    obj["angular_velocity"] / BALL_MAX_SPIN,
                ]
            )

    # Score
    # obs.extend([state["score"]["blue"], state["score"]["orange"]])

    return np.asarray(obs, dtype=np.float32)


def training_viz(sim: Simulation):
    """Launch the Pygame interactive sandbox with 2D vector controls and Octane physics."""
    pygame.init()

    pygame.display.set_caption("Rocket League Sideswipe 2D - Suspension Physics Sandbox")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    renderer = Renderer(screen)
    renderer.render(sim)
    pygame.display.flip()
    clock.tick(SIM_HZ)

    pygame.quit()


class GymEnv(gym.Env):
    meta_data = {"render_mode": ["human", "terminal"], "render_fps": SIM_HZ}

    def __init__(self, render_mode: str = "terminal", bot_type:str="zeros"):
        super().__init__()

        # Setting up simulation
        self.sim = Simulation(enable_orange=True)
        self.bot_type = bot_type
        self.render_mode = render_mode
        self.previous_score = {"blue": 0, "orange": 0}
        self.step_count = 0
        self.is_open = True
        if render_mode == "human":
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(screen)

        # Setting up Environment
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        # select x-direction, y-direction, jump, and boost
        self.action_space = gym.spaces.MultiDiscrete([3,3,2,2]) # size = 36
        # select one of 9 possible directions, jump, and boost
        # self.action_space = gym.spaces.MultiDiscrete([9,2,2]) # size = 36
        # press any of 6 buttons (left, right, up, down, jump, boost)
        # self.action_space = gym.spaces.MultiBinary(6) # size = 64
        # has redundant actions e.g. up + down. This makes the probability of choosing a valid action lower
        self.reward_range = (-1,1)

    def calculate_reward(self, state):
        # score reward = [-1, 1]
        score_reward = (state["score"]["blue"] - self.previous_score["blue"]) - (
            state["score"]["orange"] - self.previous_score["orange"]
        )
        # time_reward = -1 * self.episode_count/100
        time_reward = 0
        return score_reward + time_reward

    def check_terminated(self, state):
        terminated = False
        if (
            state["score"]["blue"] != self.previous_score["blue"]
            or state["score"]["orange"] != self.previous_score["orange"]
        ):
            terminated = True
        self.previous_score["blue"] = state["score"]["blue"]
        self.previous_score["orange"] = state["score"]["orange"]
        return terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_score = {"blue": 0, "orange": 0}
        self.step_count = 0
        self.sim.reset(reset_scores=True)
        self.observed_state = get_observation(self.sim.get_state())
        self.render()
        return self.observed_state, {} #

    def step(self, action):
        """
        """
        if not self.is_open:
            return self.observed_state, 0.0, False, True, {"is_success": False}

        # Apply Action
        x, y, jump, boost = action
        # x,y = [0,1,2] where 2 is going left or going down
        # jump, boost = [0,1]
        if x == 2:
            x = -1 #left
        if y == 2:
            y = -1 #down
        car_action = CarAction(dir_x=float(x), dir_y=float(y), jump=bool(jump), boost=bool(boost)).clamp()

        # Updating State
        self.sim.step(car_action, 1.0 / SIM_HZ)
        self.step_count+=1
        state = self.sim.get_state()
        self.observed_state = get_observation(state)

        reward = self.calculate_reward(state)
        terminated = self.check_terminated(state)
        info = {"is_success": True}
        if terminated:
            info["goal_scored":self.step_count]
        truncated = not self.is_open

        if self.render_mode == "human":
            self.render()

        return self.observed_state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == None:
            return
        elif self.render_mode == "terminal":
            print(f"Step: {self.step_count}")
            print("Observed State:")
            print(self.observed_state)
        elif self.render_mode == "human":
            if not self.is_open:
                return
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.is_open = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.sim.reset()
                    elif event.key == pygame.K_b:
                        # Toggle bot to stop TODO
                        self.bot_type = "zeros"
                    elif event.key == pygame.K_l:
                        # Toggle 45-degree boundary raycasts on/off
                        self.renderer.show_raycasts = not self.renderer.show_raycasts
                elif event.type == pygame.MOUSEBUTTON:
                    if event.button == 1:
                        self.renderer.handle_click(event.pos)
            self.renderer.render(self.sim)
            pygame.display.flip()
            self.clock.tick(self.render_fps)

    def close(self):
        if self.is_open:
            self.is_open = False
        if self.render_mode == "human" and pygame.get_init():
            pygame.display.quit()
            pygame.quit()

def check_environment():
    from gymnasium.utils.env_checker import check_env
    gym_env = gym.make(id=local_config["rl_id"], render_mode=None, disable_env_checker = True)
    check_env(gym_env.unwrapped)
    print("Environment check passed. gym.make successful")

def train_headless():
    """Run headless simulation loop for automated benchmark or verification."""
    if ENABLE_ORANGE_CAR:
        bot_type = "bot_level_3"
    else:
        bot_type = "zeros"
    # Create Gym Environment
    gym_env = gym.make(id=local_config["rl_id"])
    print("OBSERVATION SPACE:")
    observed_state, info = gym_env.reset()
    # print(observed_state)
    print(gym_env.observation_space.sample())
    print("ACTION SPACE:")
    print(gym_env.action_space.sample())

    # Using Stable Baseline PPO model
    model = PPO(policy="MultiInputPolicy", env=gym_env, device="cuda", verbose=1)
    gym_env.reset()
    total_reward = 0
    while gym_env.unwrapped.isopen:
        action = gym_env.action_space.sample()
        # print(action)
        observed_state, reward, terminated, truncated, info = gym_env.step(action)
        total_reward+=reward
        # print(observed_state)
        if not gym_env.unwrapped.isopen():
            break
        if terminated or truncated:
            print(f"Last Action:\n{action}\nRollout Reward:{total_reward}")


    model.save(f"PPO-{local_config['rl_id']}")
    print("Headless simulation benchmark completed successfully!")


def main():
    max_episode = int(sys.argv[1])
    train_headless(max_episode)


if __name__ == "__main__":
    main()
