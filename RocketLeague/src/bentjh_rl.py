# Virtual display
# from pyvirtualdisplay import Display
# 
# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()
# 
# import os
# import gymnasium
# import panda_gym
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import SIM_HZ
from src.core.actions import CarAction
from src.core.simulation import Simulation
import gym

# sys.modules["gym"] = gymnasium
# import gymnasium as gym
# from gymnasium as spaces

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

def get_observation(state: dict) -> np.ndarray:
    """
    Extract pose and twist for:
        car, car_orange, ball

    Returns:
        np.ndarray with shape (20,)
    """
    obs = []

    for name in ("car", "car_orange", "ball"):
        obj = state[name]

        # Pose: x, y, theta
        obs.extend([
            obj["position"][0],
            obj["position"][1],
            obj["angle"],
        ])

        # Twist: vx, vy, omega
        obs.extend([
            obj["velocity"][0],
            obj["velocity"][1],
            obj["angular_velocity"],
        ])

    # Score
    obs.extend([
        state["score"]["blue"],
        state["score"]["orange"]
    ])

    return np.asarray(obs, dtype=np.float32)

def gym_action_to_car_action(np_action: np.array) -> CarAction:
    """Convert a Gymnasium Discrete(5) action to a CarAction."""
    return CarAction(dir_x=np_action[0], 
                     dir_y=np_action[1], 
                     jump=np_action[2],
                     boost=np_action[3])

class GymEnv(gym.Env):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.previous_score = {"blue":0, "orange":0}
        self.episode_count = 0
        # self.observation_space = spaces.Box(low=1, high=1,shape=(4,),dtype=np.float32)
        # self.action_space = spaces.Discrete(2)
        # self.reward_range = (-1,1)

    def reset(self, seed=None, options=None):
        self.previous_score = {"blue":0, "orange":0}
        self.episode_count = 0
        self.sim.reset(reset_scores = True)

    def step(self, action):
        """
        state: 19 elements
        """
        self.sim.step(gym_action_to_car_action(action), 1.0 / SIM_HZ)
        state = self.sim.get_state()
        self.observed_state = get_observation(state)

        terminated = True
        truncated = False
        info = {}

        score_reward = (state["score"]["blue"]-self.previous_score["blue"]) - (state["score"]["orange"]-self.previous_score["orange"])
        time_reward = -1 * self.episode_count/100
        reward = score_reward + time_reward

        self.previous_score["blue"] = state["score"]["blue"]
        self.previous_score["orange"] = state["score"]["orange"]

        return (self.observed_state, reward, terminated, truncated, info)

    def render(self):
        pass

def train_headless(max_steps: int = 1):
    
    """Run headless simulation loop for automated benchmark or verification."""
    print(f"Running headless simulation for {max_steps} steps...")
    sim = Simulation(enable_orange=True)
    gym_env = GymEnv(sim)
    action = np.array([1.0, 0.2, False, True])
    gym_env.reset()

    for step in range(max_steps):
        gym_env.step(action)

    print("Headless simulation benchmark completed successfully!")


def main():
    max_episode = int(sys.argv[1])
    train_headless(max_episode)

if __name__ == "__main__":
    main()