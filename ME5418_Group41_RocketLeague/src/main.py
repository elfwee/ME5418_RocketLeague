# src/my_app/main.py

"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""

import sys

# import os

# Ensure repository root is on sys.path when invoked directly
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.game import play_headless, play_interactive
from src.gym_env.rocket_league_env import demo_gym
from src.neural_nets.dueling_ddqn_torch import run_dqn_network

def main():
    if "--play" in sys.argv:
        if "--headless" in sys.argv:
            play_headless(steps=300, enable_orange=True)
        play_interactive(enable_orange=True)
    elif "--demo" in sys.argv:
        if "--gym" in sys.argv:
            demo_gym()
        elif "--double_dqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=False)
        elif "--dueling_ddqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=True)
            
