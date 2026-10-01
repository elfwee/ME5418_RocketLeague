import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import sys
import os
import pygame

# Ensure repository root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.actions import CarAction
from src.core.simulation import Simulation
from src.visualization.renderer import Renderer
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SIM_HZ, ENABLE_ORANGE_CAR
)

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
# When running this code, you can ignore this warning: "UserWarning: WARN: Overriding environment rocket-league-play-v0 already in registry."
register(
    id='rocket-league-play-v0',
    entry_point='rocketLeaguePlay:RocketLeagueEnv', # module_name:class_name
    max_episode_steps=6000,
)

class RocketLeagueEnv(gym.Env):
    metadata = {'render.modes': ['human', 'terminal'], 'render_fps': 1}

    def __init__(self, render_mode=None, render_fps=SIM_HZ, bot_type='none'):
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.bot_type = bot_type

        if self.bot_type == 'none':
            self.enable_orange = False
        elif self.bot_type == 'bot_level_3':
            self.enable_orange = True
        else:
            raise ValueError(f"Invalid bot_type: {self.bot_type}. Must be 'none' or 'orange'.")
        # Reset the environment
        self.reset()

        self.isopen = True
        if self.render_mode == 'human':
            pygame.init()
            pygame.display.set_caption("Rocket League Sideswipe 2D - Suspension Physics Sandbox")
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(screen)

        # Define action and observation space
        # Example: Discrete action space with 5 actions (e.g., move forward, move backward, jump, boost, etc.)
        # Actions space: 9 discrete actions for throttle/brake/steer, 2 for jump, 2 for boost
        self.action_space = spaces.MultiDiscrete([9, 2, 2])
        
        # Example: Observation space could be a vector of 10 continuous values (e.g., position, velocity, etc.)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment to an initial state and return an initial observation.
        """
        super().reset(seed=seed)
        # Reset the game state here
        self.sim = Simulation(enable_orange=self.enable_orange)
        self.state = np.asarray(self.sim.get_state_norm(as_flat_array=True), dtype=np.float32)  # Initialize state
        self.obs_dim = self.state.shape[0]  # Observation dimension
        # initial_observation = [0.0] * 10  # Example initial observation
        return self.state, {}

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Parameters:
            action (int): The action to take in the environment.
        
        Returns:
            observation (list): The next observation after taking the action.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """
        # Implement the logic to update the game state based on the action
        """
        observation = [0.0] * 10  # Example next observation
        reward = 0.0  # Example reward
        done = False  # Example done flag
        info = {}  # Example info dictionary
        
        return observation, reward, done, info
        """
        pass

    def render(self):
        """
        Render the environment to the screen.
        
        Parameters:
            mode (str): The mode in which to render the environment. Default is 'human'.
        """
        # Implement rendering logic here
        pass

    def close(self):
        """
        Perform any necessary cleanup when closing the environment.
        """
        # Implement cleanup logic here
        pass

# Check validity of the environment
def my_check_env():
    """
    Check the validity of the Rocket League environment using Gymnasium's check_env utility.
    This function will raise an error if the environment does not conform to Gymnasium's API.
    """
    from gymnasium.utils.env_checker import check_env
    env = gym.make('rocket-league-play-v0', render_mode=None)
    check_env(env.unwrapped)

if __name__ == "__main__":
    my_check_env()