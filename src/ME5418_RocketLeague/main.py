# src/my_app/main.py

"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""
import sys
# import os

# Ensure repository root is on sys.path when invoked directly
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ME5418_RocketLeague.simulation.core.game import Game
from ME5418_RocketLeague.gym_environment.bentjh_gym import test_gym
from ME5418_RocketLeague.gym_environment.bentjh_net import test_net

def main():
    game = Game()
    if "--headless" in sys.argv:
        game.run_headless()

    elif "--demo" in sys.argv:
        game.run_demo()

    elif "--gym" in sys.argv:
        # test_gym(render_mode="human", adversary_type="random")
        test_gym(render_mode="human", adversary_type="none")
        # test_gym(render_mode="terminal", adversary_type="none")
    
    elif "--net" in sys.argv:
        test_net()

    else:
        game.run_interactive()

if __name__ == '__main__':
    # if "--no-orange" in sys.argv:
        # enable_orange = False
    # elif "--orange" in sys.argv:
        # enable_orange = True
    pass