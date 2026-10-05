import gymnasium as gym
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# Ensure repository root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rocket_league_env import RocketLeagueEnv, RewardLoggingCallback, SoccerEvalCallback
from src.config import SIM_HZ


def train():
    # Base directories relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 3600 steps = 60 seconds (1 minute match) at 60 Hz
    max_episode_steps = 3600

    env_kwargs = {
        "bot_type": "bot_level_3",
        "render_mode": None,
        "max_episode_steps": max_episode_steps,  # Passed to gym.make -> wrapped in TimeLimit
    }

    # Training environment (4 parallel environments via DummyVecEnv for fast in-memory stepping)
    env = make_vec_env("rocket-league-v1", n_envs=4, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=1,
    #     device="cpu",
    #     tensorboard_log=log_dir,
    # )

    model = PPO.load(
        "/home/jensen/ME5418/ME5418_RocketLeague/RocketLeague/src/models/PPO/best_model_final4_nobot.zip", 
        env=env,
        device="cpu",
        tensorboard_log=log_dir,
        )

    # Independent soccer evaluation against frozen opponents
    # Selects best_model.zip based on match points (3W + D)/(3N) and goal differential,
    # completely independent of shaped training rewards.
    opponents = {
        "heuristic_bot": "bot_level_3",
    }

    soccer_eval_callback = SoccerEvalCallback(
        opponents=opponents,
        eval_freq=50_000,
        n_eval_matches=50,
        save_path=os.path.join(model_dir, "PPO"),
        deterministic=True,
        verbose=1,
    )

    # Logs episode reward breakdown (step1..step4, goal, total) to TensorBoard
    reward_logging_callback = RewardLoggingCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(model_dir, "PPO", "checkpoints"),
        name_prefix="ppo",
    )

    TIMESTEPS = 10_000_000

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=CallbackList([reward_logging_callback, soccer_eval_callback, checkpoint_callback]),
        # reset_num_timesteps=False,
    )

if __name__ == "__main__":
    train()