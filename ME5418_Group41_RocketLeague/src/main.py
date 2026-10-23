"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""
import sys
import os

# Ensure repository root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.game import Game

if __name__ == '__main__':
    game = Game()
    # if "--no-orange" in sys.argv:
        # enable_orange = False
    # elif "--orange" in sys.argv:
        # enable_orange = True

    if "--headless" in sys.argv:
        game.run_headless()
    else:
        game.run_interactive()
