# src/my_app/main.py

"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""

import sys

# import os

# Ensure repository root is on sys.path when invoked directly
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.game import play_headless, play_interactive
from src.gym_env.rocket_league_env import demo_gym

def main():
    if "--play" in sys.argv:
        if "--headless" in sys.argv:
            play_headless(steps=300, enable_orange=True)
        play_interactive(enable_orange=True)
    elif "--demo" in sys.argv:
        if "--gym" in sys.argv:
            demo_gym()
        # Import lazily: loading TensorFlow and PyTorch in one process crashes (LLVM conflict)
        elif "--dqn" in sys.argv:
            from src.neural_nets.double_dqn_tf import test_nn
            test_nn()
        elif "--ddqn" in sys.argv:
            from ME5418_Group41_RocketLeague.src.neural_nets.dueling_ddqn_torch import run_network
            run_network()
