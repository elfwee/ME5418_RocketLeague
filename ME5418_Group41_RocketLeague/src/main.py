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


def main():
    if "--play" in sys.argv:
        if "--headless" in sys.argv:
            play_headless(steps=300, enable_orange=True)
        else:
            play_interactive(enable_orange=True)
    elif "--demo" in sys.argv:
        if "--gym" in sys.argv:
            demo_gym()
        elif "--dqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=False, enable_double_dqn=False)
        elif "--double_dqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=False, enable_double_dqn=True)
        elif "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=True, enable_double_dqn=True)
        else:
            print(
                "Please specify a demo target:\n"
                "  python -m src --demo --gym           # Gymnasium environment demo\n"
                "  python -m src --demo --dqn           # Vanilla DQN (PyTorch)\n"
                "  python -m src --demo --double_dqn    # Double DQN (PyTorch)\n"
                "  python -m src --demo --dueling_ddqn  # Dueling Double DQN (PyTorch) [alias: --ddqn]"
            )
    else:
        print(
            "Usage:\n"
            "  python -m src --play                 # Play interactively\n"
            "  python -m src --play --headless      # Run headless physics benchmark\n"
            "  python -m src --demo --gym           # Demo Gymnasium environment\n"
            "  python -m src --demo --dqn           # Demo Vanilla DQN (PyTorch)\n"
            "  python -m src --demo --double_dqn    # Demo Double DQN (PyTorch)\n"
            "  python -m src --demo --dueling_ddqn  # Demo Dueling Double DQN (PyTorch) [alias: --ddqn]"
        )


if __name__ == "__main__":
    main()
