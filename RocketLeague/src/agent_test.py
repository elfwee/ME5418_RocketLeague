import gymnasium as gym
import sys
import os
from stable_baselines3 import PPO

# Ensure repository root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rocket_league_env import RocketLeagueEnv

def test_sb3(model_timesteps=None, model_path=None, bot_type="bot_level_3", render=True, num_episodes=5):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        if model_timesteps is not None:
            model_path = os.path.join(base_dir, "models", "PPO", f"ppo_{model_timesteps}")
        else:
            model_path = os.path.join(base_dir, "models", "PPO", "best_model.zip")

    env = gym.make("rocket-league-v0",render_mode="human" if render else None, bot_type=bot_type)

    model = PPO.load(model_path, env=env, device="cpu")

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if hasattr(env.unwrapped, "isopen") and not env.unwrapped.isopen:
                break

        print(f"Episode {episode} Finished | Total Reward: {episode_reward:.2f}")

        if hasattr(env.unwrapped, "isopen") and not env.unwrapped.isopen:
            break

    env.close()


if __name__ == "__main__":
    test_sb3(render=True, bot_type="bot_level_3", num_episodes=5)