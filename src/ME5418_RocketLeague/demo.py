"""
ME5418 Project Milestone 1 - Rocket League Gym Environment Demo
Runs sample episodes under a random policy with graphic visualization.
"""
# import os
# import sys

# Ensure src can be imported directly
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ME5418_RocketLeague.gym_environment.rocket_league_env import RocketLeagueEnv


def run_demo(episodes: int = 3, bot: bool = True):
    bot_type = "heuristic_bot" if bot else "none"
    print(f"Launching Rocket League Gym Demo with bot='{bot_type}'...")

    env = RocketLeagueEnv(render_mode="human", bot_type=bot_type)

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        total_reward = 0.0
        step_count = 0
        print(f"\n--- Episode {ep + 1}/{episodes} Started ---")

        while not terminated and env.isopen:
            # Sample random action (criteria: random/deterministic policy demo)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if truncated:
                break

        print(f"Episode {ep + 1} Finished: Steps={step_count}, Total Reward={total_reward:.2f}")

    env.close()
    print("\nDemo finished successfully!")


if __name__ == "__main__":
    run_demo(episodes=3, bot=True)