import gymnasium as gym
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

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
    max_episode_steps = 3600

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

    # model = PPO.load(
    #     "/home/jensen/ME5418/ME5418_RocketLeague/RocketLeague/src/models/PPO/best_model_final2_nobot2.zip", 
    #     env=env,
    #     device="cpu",
    #     tensorboard_log=log_dir,
    #     )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "PPO"),
        log_path=log_dir,
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=os.path.join(model_dir, "PPO", "checkpoints"),
        name_prefix="ppo",
    )

    TIMESTEPS = 10_000_000

    model.learn(
        total_timesteps=TIMESTEPS,
        callback = CallbackList([eval_callback, checkpoint_callback]),
        # reset_num_timesteps=False,
    )

if __name__ == "__main__":
    train()