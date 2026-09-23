import gymnasium as gym
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

# register env as gym environment
if "rocket-league-v0" not in gym.envs.registry:
    gym.register(
        id="rocket-league-v0",
        entry_point="src.rocket_league_env:RocketLeagueEnv",
        max_episode_steps=6000,
    )

# Define the wrapper to swap 'truncated' for 'terminated'
class TerminateOnTimeout(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # If the 6000 step limit is hit, force terminated to True
        if truncated:
            terminated = True
            
        return obs, reward, terminated, truncated, info

class RocketLeagueEnv(gym.Env):
    metadata = {'render_modes': ['human', 'terminal'], 'render_fps': SIM_HZ}

    def __init__(self, render_mode=None, render_fps=SIM_HZ, bot_type='none'):

        self.bot_type = bot_type
        self.render_mode = render_mode
        self.enable_orange = None
        self.render_fps = render_fps
        self.ball_prev = None

        if self.bot_type == 'none':
            self.enable_orange = False
        elif self.bot_type == 'bot_level_3':
            self.enable_orange = True
        else:
            raise ValueError(f"Invalid bot_type: {self.bot_type}. Must be 'none' or 'orange'.")

        self.sim = Simulation(enable_orange=self.enable_orange)
        self.state = np.asarray(self.sim.get_state_norm(as_flat_array=True), dtype=np.float32)  # Initialize state
        self.obs_dim = self.state.shape[0]  # Observation dimension
        # print(f"Observation dimension: {self.obs_dim}")

        self.isopen = True
        if self.render_mode == 'human':
            pygame.init()
            pygame.display.set_caption("Rocket League Sideswipe 2D - Suspension Physics Sandbox")
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(screen)

        # action space: 9 discrete actions for throttle/brake/steer, 2 for jump, 2 for boost
        self.action_space = gym.spaces.MultiDiscrete([9, 2, 2])

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        spawn_pos = options.get("spawn_pos") if options else None
        spawn_index = options.get("spawn_index") if options else None
        if options and options.get("random_spawn"):
            spawn_index = int(self.np_random.integers(0, 2))

        self.sim.reset(reset_scores=True, spawn_pos=spawn_pos, spawn_index=spawn_index)
        self.ball_prev = None
        self.state = np.asarray(self.sim.get_state_norm(as_flat_array=True), dtype=np.float32)
        self.render()

        return self.state, {} # observation, info

    def step(self, input_action):
        sub_action1, sub_action2, sub_action3 = input_action
        match sub_action1:
            case 0:  # No movement
                input_dir_x=0.0
                input_dir_y=0.0
            case 1:  # Right
                input_dir_x=1.0
                input_dir_y=0.0
            case 2:  # Left
                input_dir_x=-1.0
                input_dir_y=0.0
            case 3:  # Down
                input_dir_x=0.0
                input_dir_y=-1.0
            case 4:  # Up
                input_dir_x=0.0
                input_dir_y=1.0
            case 5:  # Down + Right
                input_dir_x=1.0
                input_dir_y=-1.0
            case 6:  # Up + Right
                input_dir_x=1.0
                input_dir_y=1.0
            case 7:  # Down + Left
                input_dir_x=-1.0
                input_dir_y=-1.0
            case 8:  # Up + Left
                input_dir_x=-1.0
                input_dir_y=1.0
            case _:
                input_dir_x=0.0
                input_dir_y=0.0

        action = CarAction(dir_x=input_dir_x, dir_y=input_dir_y, boost=sub_action3==1, jump=sub_action2==1).clamp()

        if not self.isopen:
            return self.state, 0.0, False, True, {}

        self.sim.step(action, 1.0 / self.render_fps)
        reward = self._calculate_reward()
        if self.render_mode == 'human':
            self.render()
        self.state = np.asarray(self.sim.get_state_norm(as_flat_array=True), dtype=np.float32)
        truncated = not self.isopen

        return self.state, reward, False, truncated, {}  # observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == None:
            return  # No rendering if render_mode is None

        if self.render_mode == 'terminal':
            self._render_terminal()

        if self.render_mode == 'human':
            self._render_human()

    def _render_terminal(self):
        # Print the state in a human-readable format
        print("Current State:")
        print(self.state)

    def _render_human(self):
        if not self.isopen:
            return

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.isopen = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.sim.reset()
                elif event.key == pygame.K_b:
                    # Toggle Orange bot on/off
                    self.sim.set_orange_enabled(not self.sim.enable_orange, is_bot=True)
                elif event.key == pygame.K_l:
                    # Toggle 45-degree boundary raycasts on/off
                    self.renderer.show_raycasts = not self.renderer.show_raycasts
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.renderer.handle_click(event.pos)

        self.renderer.render(self.sim)
        pygame.display.flip()
        self.clock.tick(self.render_fps)

    def close(self):
        if self.isopen:
            self.isopen = False
        if self.render_mode == 'human' and pygame.get_init():
            pygame.display.quit()
            pygame.quit()

    def _calculate_reward(self):
        reward = 0.0
        k_factor = 5.0  # Scaling factor for reward calculation
        state = self.sim.get_state_norm() # Get the current state of the simulation
        last_touch = state['last_touch']
        goal_scored_step = state['goal_scored_step']
        if goal_scored_step is not None:
            if goal_scored_step == 'blue':  # Blue team scored
                if last_touch == 'orange':
                    reward = 0.0
                else:
                    reward = 20.0  # Reward for scoring a goal
            elif goal_scored_step == 'orange':  # Orange team scored
                reward = -20.0  # Penalty for conceding a goal
            self.ball_prev = None
        else:
            ball_current = state['ball']['position'][0] # ball's x-position
            if self.ball_prev is None:
                self.ball_prev = ball_current
            delta_ball = ball_current - self.ball_prev
            if abs(delta_ball) > 0.0001:
                step_reward = delta_ball * abs(ball_current) * k_factor
                if (step_reward >= 0.0 and last_touch == 'blue') or (step_reward < 0.0 and last_touch == 'orange'):
                    reward += step_reward
            self.ball_prev = ball_current

            ball_proximity = state['relations']['agent_to_ball']['magnitude']
            reward += -abs(ball_proximity) * 0.1

        return reward

def my_check_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make('rocket-league-v0', render_mode=None, render_fps=SIM_HZ, bot_type='none', disable_env_checker=True)
    check_env(env.unwrapped)
    print("Environment check passed successfully!")

if __name__ == "__main__":
    # my_check_env()
    if ENABLE_ORANGE_CAR:
        _bot_type = 'bot_level_3'
    else:
        _bot_type = 'none'

    env = gym.make('rocket-league-v0', render_mode='human', render_fps=SIM_HZ, bot_type=_bot_type, max_episode_steps=6000)
    # env = TerminateOnTimeout(env) # Truncated == Terminated

    obs, info = env.reset()
    total_reward = 0.0

    while env.unwrapped.isopen:
        action = env.action_space.sample()  # Random action for demonstration
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if not env.unwrapped.isopen:
            break

        if terminated or truncated:
            print(f"Last action: {action}, Rollout Reward: {total_reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
