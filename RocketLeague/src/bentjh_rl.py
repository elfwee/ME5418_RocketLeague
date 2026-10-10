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
    np_action[0] #do nothing
    
    np_action[1] #left
    np_action[2] #right
    np_action[3] #up
    np_action[4] #down
    np_action[5] #jump
    np_action[6] #boost

    np_action[7] #left + up 
    np_action[8] #left + down 
    np_action[9] #right + up 
    np_action[10] #right + down 
    
    np_action[11] #left + jump
    np_action[12] #up + jump
    np_action[13] #right + jump
    np_action[14] #down + jump

    np_action[15] #left + boost 
    np_action[16] #right + boost 
    np_action[17] #up + boost 
    np_action[18] #down + boost 

    np_action[19] #left + up + jump
    np_action[20] #left + up + boost
    np_action[21] #left + down + jump
    np_action[22] #left + down + boost

    np_action[23] #right + up + jump
    np_action[24] #right + up + boost
    np_action[25] #right + down + jump
    np_action[26] #right + down + boost

    np_action[27] #left + boost + jump
    np_action[28] #right + boost + jump
    np_action[29] #up + boost + jump
    np_action[30] #down + boost + jump

    np_action[31] #left + up + boost + jump
    np_action[32] #left + down + boost + jump
    np_action[33] #right + up + boost + jump
    np_action[34] #right + down + boost + jump

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
        self.observation_space = gym.spaces.Box(low=np.inf, high=np.inf,shape=(20,),dtype=np.float32)
        self.action_space = gym.spaces.Discrete(35)
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
        print(self.observed_state)
        print(action)

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