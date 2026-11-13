"""Gymnasium environment demo: run the env checker, then one episode of random actions in a window.

python -m src --demo --gym
"""

import gymnasium as gym
import numpy as np

from src.config.gym_config import GymConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv  # noqa: F401  (import registers the gym env id)


def my_check_env():
    from gymnasium.utils.env_checker import check_env

    env = gym.make(GymConfig.gym_id, render_mode=None, bot_type="none", disable_env_checker=True)
    check_env(env.unwrapped)
    print("Environment check passed successfully!")


def demo_gym():
    """Check the env, then play one episode of random actions in a window."""
    my_check_env()
    env = gym.make(GymConfig.gym_id, render_mode="human", bot_type="heuristic_bot")

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
