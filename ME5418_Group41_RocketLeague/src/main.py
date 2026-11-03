"""Main entry point for interactive 2D Rocket League simulation, Gym demo, and Neural Network demos."""

import os
import sys

# Ensure repository root is on sys.path when invoked directly (e.g., `python src/main.py`)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.sim.game import play_headless, play_interactive
from src.gym_env.rocket_league_env import demo_gym
from src.neural_nets.dueling_ddqn_torch import run_dqn_network
from src.neural_nets.a2c_torch import test_a2c_torch
from src.agent.agent import train_agent, play_agent
from src.gym_env.bentjh_gym import test_gym as demo_bentjh_gym
from src.config.gym_config import GymConfig, BentjhGymConfig

ENV_IDS = {"rocketleague": GymConfig.gym_id, "bentjh": BentjhGymConfig.gym_id}


def _arg_value(flag, default=None):
    """Return the value following ``flag`` in sys.argv, or ``default``."""
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


def main():
    env_id = ENV_IDS[_arg_value("--env", "rocketleague")]
    if "--train" in sys.argv:
        train_agent(
            number_of_episodes=int(_arg_value("--episodes", 100_000)),
            pretrain_model_filepath=_arg_value("--pretrain"),
            env_id=env_id,
        )
    elif "--play" in sys.argv:
        if "--agent" in sys.argv:
            play_agent(_arg_value("--agent"), env_id=env_id)
        elif "--headless" in sys.argv:
            play_headless(steps=300, enable_orange=True)
        else:
            play_interactive(enable_orange=True)
    elif "--demo" in sys.argv:
        if "--gym" in sys.argv:
            demo_gym()
        elif "--bentjh_gym" in sys.argv:
            demo_bentjh_gym(render_mode="human")
        elif "--dqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=False, enable_double_dqn=False)
        elif "--double_dqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=False, enable_double_dqn=True)
        elif "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=True, enable_double_dqn=True)
        elif "--a2c" in sys.argv or "-a2c" in sys.argv:
            test_a2c_torch()
        else:
            print(
                "Please specify a demo target:\n"
                "  python -m src --demo --gym           # Gymnasium environment demo\n"
                "  python -m src --demo --bentjh_gym    # bentjh env, keyboard-driven, prints reward\n"
                "  python -m src --demo --dqn           # Vanilla DQN (PyTorch)\n"
                "  python -m src --demo --double_dqn    # Double DQN (PyTorch)\n"
                "  python -m src --demo --dueling_ddqn  # Dueling Double DQN (PyTorch) [alias: --ddqn]\n"
                "  python -m src --demo --a2c           # Advantage Actor-Critic (PyTorch) [alias: -a2c]"
            )
    else:
        print(
            "Usage:\n"
            "  python -m src --train [--episodes N] [--pretrain PATH] [--env rocketleague|bentjh]  # Train A2C agent\n"
            "  python -m src --play --agent PATH [--env rocketleague|bentjh]  # Watch a trained agent play\n"
            "  python -m src --play                 # Play interactively\n"
            "  python -m src --play --headless      # Run headless physics benchmark\n"
            "  python -m src --demo --gym           # Demo Gymnasium environment\n"
            "  python -m src --demo --bentjh_gym    # Demo bentjh env (keyboard-driven)\n"
            "  python -m src --demo --dqn           # Demo Vanilla DQN (PyTorch)\n"
            "  python -m src --demo --double_dqn    # Demo Double DQN (PyTorch)\n"
            "  python -m src --demo --dueling_ddqn  # Demo Dueling Double DQN (PyTorch) [alias: --ddqn]"
        )


if __name__ == "__main__":
    main()
