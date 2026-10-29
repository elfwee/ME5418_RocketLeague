# src/my_app/main.py

"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""
import sys
# import os

# Ensure repository root is on sys.path when invoked directly
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.game import play_headless, play_interactive
from src.neural_nets.dqn_tf import test_nn

def main():
    if "--play" in sys.argv:
        if "--headless" in sys.argv:
            play_headless(steps=300, enable_orange=True)
        play_interactive(enable_orange=True)
    elif "--demo_nn" in sys.argv:
        test_nn()

    # elif "--gym" in sys.argv:
    #     # test_gym(render_mode="human", adversary_type="random")
    #     test_gym(render_mode="human", adversary_type="none")
    #     # test_gym(render_mode="terminal", adversary_type="none")
    # 
    # elif "--net" in sys.argv:
    #     test_nn()
