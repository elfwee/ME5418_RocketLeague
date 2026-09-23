import gymnasium as gym
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Ensure repository root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rocket_league_env import RocketLeagueEnv
from src.config import SIM_HZ


def train():
    # Base directories relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 6000 steps = 100 seconds (1 minute match) at 60 Hz
    max_episode_steps = 6000

    env_kwargs = {
        "bot_type": "bot_level_3",
        "render_mode": None,
        "max_episode_steps": max_episode_steps,  # Passed to gym.make -> wrapped in TimeLimit
    }

    # Training environment (4 parallel environments via DummyVecEnv for fast in-memory stepping)
    env = make_vec_env("rocket-league-v1", n_envs=4, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    # Separate evaluation environment (do not evaluate using the training environment)
    eval_env = make_vec_env("rocket-league-v1", n_envs=1, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        tensorboard_log=log_dir,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "PPO"),
        log_path=log_dir,
        eval_freq=4000,       # Evaluates every 2,500 calls * 4 envs = 10,000 timesteps
        n_eval_episodes=3,    # Evaluate across 3 full episodes
        deterministic=True,   # Evaluates the agent using greedy/best actions
        verbose=1,
    )

    TIMESTEPS = max_episode_steps * 1000  # 3,600,000 timesteps total

    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback)

if __name__ == "__main__":
    train()