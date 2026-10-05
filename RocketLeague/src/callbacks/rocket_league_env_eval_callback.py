import gymnasium as gym
import numpy as np
import sys
import os
import pygame
import random

# Ensure repository root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.actions import CarAction
from src.core.simulation import Simulation
from src.visualization.renderer import Renderer
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SIM_HZ, ENABLE_ORANGE_CAR
)

# register env as gym environment
if "rocket-league-v1" not in gym.envs.registry:
    gym.register(
        id="rocket-league-v1",
        entry_point="src.rocket_league_env:RocketLeagueEnv",
        max_episode_steps=3600,
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
    _tb_writer = None
    _tb_writer_dir = None
    _total_steps = 0
    _total_episodes = 0

    def __init__(self, render_mode=None, render_fps=SIM_HZ, bot_type='none', enable_tb_logging=None, terminate_on_goal=True):

        self.bot_type = bot_type
        self.terminate_on_goal = terminate_on_goal
        self.render_mode = render_mode
        self.enable_orange = None
        self.orange_is_bot = False
        self.render_fps = render_fps
        self.ball_prev = None
        self.input_prev = None
        self.episode_reward = 0.0
        self.step1_reward = 0.0
        self.step2_reward = 0.0
        self.step3_reward = 0.0
        self.step4_reward = 0.0
        self.goal_reward = 0.0
        self.step_no = 0
        if enable_tb_logging is None:
            self.enable_tb_logging = (self.render_mode != 'human')
        else:
            self.enable_tb_logging = enable_tb_logging

        if self.bot_type in ('none', None, False):
            self.enable_orange = False
            self.orange_is_bot = False
        elif self.bot_type in ('bot_level_3', 'heuristic_bot', 'bot', True):
            self.enable_orange = True
            self.orange_is_bot = True
        elif self.bot_type == 'passive':
            self.enable_orange = True
            self.orange_is_bot = False
        else:
            raise ValueError(f"Invalid bot_type: {self.bot_type}. Must be 'none', 'bot_level_3', 'heuristic_bot', or 'passive'.")

        self.sim = Simulation(enable_orange=self.enable_orange, orange_is_bot=self.orange_is_bot)
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

    def set_opponent(self, bot_type='bot_level_3'):
        """Dynamically configure or switch the opponent without re-instantiating the environment."""
        self.bot_type = bot_type
        if self.bot_type in ('none', None, False):
            self.enable_orange = False
            self.orange_is_bot = False
            self.sim.set_orange_enabled(False, is_bot=False)
        elif self.bot_type in ('bot_level_3', 'heuristic_bot', 'bot', True):
            self.enable_orange = True
            self.orange_is_bot = True
            self.sim.set_orange_enabled(True, is_bot=True)
        elif self.bot_type == 'passive':
            self.enable_orange = True
            self.orange_is_bot = False
            self.sim.set_orange_enabled(True, is_bot=False)
        else:
            raise ValueError(f"Invalid bot_type: {self.bot_type}. Must be 'none', 'bot_level_3', 'heuristic_bot', or 'passive'.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        spawn_pos = options.get("spawn_pos") if options else None
        spawn_index = options.get("spawn_index") if options else None
        random_spawn = options.get("random_spawn") if options else None

        self.sim.reset(reset_scores=True, spawn_pos=spawn_pos, spawn_index=spawn_index, random_spawn=random_spawn)
        self.ball_prev = None
        self.input_prev = None
        self.episode_reward = 0.0
        self.step1_reward = 0.0
        self.step2_reward = 0.0
        self.step3_reward = 0.0
        self.step4_reward = 0.0
        self.goal_reward = 0.0
        self.step_no = 0
        self.state = np.asarray(self.sim.get_state_norm(as_flat_array=True), dtype=np.float32)
        self.render()
        initial_info = {
            "goals_for": int(self.sim.score_blue),
            "goals_against": int(self.sim.score_orange),
            "goal_difference": int(self.sim.score_blue - self.sim.score_orange),
            "match_result": "draw",
        }
        return self.state, initial_info # observation, info

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
            goals_for = int(self.sim.score_blue)
            goals_against = int(self.sim.score_orange)
            goal_diff = goals_for - goals_against
            match_result = "win" if goal_diff > 0 else ("loss" if goal_diff < 0 else "draw")
            return self.state, 0.0, False, True, {
                "is_success": False,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_difference": goal_diff,
                "match_result": match_result,
            }

        RocketLeagueEnv._total_steps += 1
        self.sim.step(action, 1.0 / self.render_fps)
        reward, terminated = self._calculate_reward()
        if self.render_mode == 'human':
            self.render()
        self.state = np.asarray(self.sim.get_state_norm(as_flat_array=True), dtype=np.float32)
        truncated = not self.isopen

        goals_for = int(self.sim.score_blue)
        goals_against = int(self.sim.score_orange)
        goal_diff = goals_for - goals_against
        match_result = "win" if goal_diff > 0 else ("loss" if goal_diff < 0 else "draw")

        goal_scored_step = self.sim.goal_scored_this_step
        info = {
            "is_success": bool(goal_scored_step == 'blue'),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_difference": goal_diff,
            "match_result": match_result,
        }
        if goal_scored_step is not None:
            info["goal_scored"] = goal_scored_step

        if terminated or truncated:
            RocketLeagueEnv._total_episodes += 1
            info["step_reward1"] = self.step1_reward
            info["step_reward2"] = self.step2_reward
            info["step_reward3"] = self.step3_reward
            info["step_reward4"] = self.step4_reward
            info["goal_reward"] = self.goal_reward
            info["episode_reward"] = self.episode_reward

            if self.enable_tb_logging:
                writer = self._get_tb_writer()
                if writer is not None:
                    step_idx = RocketLeagueEnv._total_steps
                    writer.add_scalar("rewards/step_reward1_field", self.step1_reward, step_idx)
                    writer.add_scalar("rewards/step_reward2_ball_dist", self.step2_reward, step_idx)
                    writer.add_scalar("rewards/step_reward3_step_penalty", self.step3_reward, step_idx)
                    writer.add_scalar("rewards/step_reward4_input_dir", self.step4_reward, step_idx)
                    writer.add_scalar("rewards/goal_reward", self.goal_reward, step_idx)
                    writer.add_scalar("rewards/episode_reward", self.episode_reward, step_idx)
                    writer.flush()

        return self.state, reward, terminated, truncated, info  # observation, reward, terminated, truncated, info

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
        if RocketLeagueEnv._tb_writer is not None:
            try:
                RocketLeagueEnv._tb_writer.flush()
                RocketLeagueEnv._tb_writer.close()
                RocketLeagueEnv._tb_writer = None
                RocketLeagueEnv._tb_writer_dir = None
            except Exception:
                pass

    @classmethod
    def _get_tb_writer(cls):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return None

        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(logs_dir):
            return None

        ppo_dirs = [
            os.path.join(logs_dir, d)
            for d in os.listdir(logs_dir)
            if os.path.isdir(os.path.join(logs_dir, d)) and d.startswith("PPO_")
        ]
        target_dir = max(ppo_dirs, key=os.path.getmtime) if ppo_dirs else logs_dir

        if cls._tb_writer is None or cls._tb_writer_dir != target_dir:
            if cls._tb_writer is not None:
                try:
                    cls._tb_writer.close()
                except Exception:
                    pass
            cls._tb_writer_dir = target_dir
            cls._tb_writer = SummaryWriter(log_dir=target_dir)

        return cls._tb_writer

    def _calculate_reward(self):
        reward = 0.0
        terminated = False
        state = self.sim.get_state_norm() # Get the current state of the simulation
        last_touch = state['last_touch']
        goal_scored_step = state['goal_scored_step']
        if goal_scored_step is not None:
            # print(f"step_no:{self.step_no}, episode reward:{self.episode_reward:.4f}, step1:{self.step1_reward:.4f}, step2:{self.step2_reward:.4f}, step3:{self.step3_reward:.4f}, step4:{self.step4_reward:.4f}")
            terminated = bool(self.terminate_on_goal)
            if goal_scored_step == 'blue':  # Blue team scored
                if last_touch == 'orange':
                    reward = 2.0
                else:
                    reward = 20.0  # Reward for scoring a goal
            elif goal_scored_step == 'orange':  # Orange team scored
                reward = -20.0  # Penalty for conceding a goal
            self.ball_prev = None
            self.goal_reward += reward
            self.episode_reward += reward
        else:
            # Field step reward
            step_reward1 = 0.0
            steepness_factor = 0.2
            ball_x = state['ball']['position'][0] # ball's x-position
            ball_y = state['ball']['position'][1] # ball's y-position
            if self.ball_prev is None:
                self.ball_prev = ball_x
            delta_x = ball_x - self.ball_prev
            if abs(delta_x) > 0.0001:
                x_factor = np.exp(1 / ((1 + steepness_factor) - ball_x**2))
                y_factor = (1 - abs(ball_y))**2 if ball_y > -0.55 else 0.0 # end of bottom curvature = -0.55
                step_reward1 = delta_x * x_factor * y_factor
                if (step_reward1 <= 0.0 and last_touch == 'blue') or (step_reward1 >= 0.0 and last_touch == 'orange'):
                    step_reward1 = 0.0
                reward += step_reward1
            self.ball_prev = ball_x

            # ball distance reward
            ball_proximity = state['relations']['agent_to_ball']['magnitude']
            step_reward2 = -abs(ball_proximity) * 0.01
            reward += step_reward2

            # step reward
            step_reward3 = -0.001
            reward += step_reward3

            # directional reward
            step_reward4 = 0.0
            last_input = state['car']['input_vector']
            if self.input_prev == None:
                self.input_prev = last_input
            elif last_input != (0.0, 0.0) and self.input_prev != last_input:
                self.input_prev = last_input
                step_reward4 = -0.001
            reward += step_reward4
                
            self.episode_reward += reward
            self.step1_reward += step_reward1
            self.step2_reward += step_reward2
            self.step3_reward += step_reward3
            self.step4_reward += step_reward4
            self.step_no += 1
            
            # print(f"reward: {reward:.4f}, step1: {step_reward1:.4f}, step2: {step_reward2:.4f}, step3: {step_reward3:.4f}, step4: {step_reward4:.4f}")
        return reward, terminated

def my_check_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make('rocket-league-v1', render_mode=None, render_fps=SIM_HZ, bot_type='none', disable_env_checker=True)
    check_env(env.unwrapped)
    print("Environment check passed successfully!")

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardLoggingCallback(BaseCallback):
        """
        Custom callback for Stable-Baselines3 that logs episode reward breakdown
        (step_reward1, step_reward2, step_reward3, step_reward4, goal_reward, episode_reward)
        directly into the SB3 Logger / TensorBoard.
        """
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "step_reward1" in info:
                    self.logger.record("rewards/step_reward1_field", info["step_reward1"])
                    self.logger.record("rewards/step_reward2_ball_dist", info["step_reward2"])
                    self.logger.record("rewards/step_reward3_step_penalty", info["step_reward3"])
                    self.logger.record("rewards/step_reward4_input_dir", info["step_reward4"])
                    self.logger.record("rewards/goal_reward", info["goal_reward"])
                    self.logger.record("rewards/episode_reward", info["episode_reward"])
            return True
except ImportError:
    class RewardLoggingCallback:
        pass

try:
    from src.callbacks.soccer_eval_callback import SoccerEvalCallback
except ImportError:
    try:
        from callbacks.soccer_eval_callback import SoccerEvalCallback
    except ImportError:
        SoccerEvalCallback = None

if __name__ == "__main__":
    # my_check_env()
    if ENABLE_ORANGE_CAR:
        _bot_type = 'bot_level_3'
    else:
        _bot_type = 'none'

    env = gym.make('rocket-league-v1', render_mode='human', render_fps=SIM_HZ, bot_type=_bot_type, max_episode_steps=3600)
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
