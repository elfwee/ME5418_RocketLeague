import pygame
import numpy as np
from ME5418_RocketLeague.simulation.core.actions import CarAction
from ME5418_RocketLeague.ai.heuristic_bot import HeuristicBot
from ME5418_RocketLeague.simulation.core.simulation import Simulation
from typing import Dict, Any, Optional, Tuple

class CarControl():
    """
    - control_type:
        - none: returns empty action
        - human: enables keyboard control
        - level3: uses heuristics bot
    - team:
        - orange
        - blue
    - sim: Simulation needed for Heuristics Bot
    """
    def __init__(self, control_type:str = "none", team:str = "blue", sim:Optional[Simulation] = None):
        self.control_type = control_type
        self.sim = sim
        self.team = team

    def get_action(self):
        action = CarAction()
        if self.control_type == "none":
            pass

        elif self.control_type == "human":
            # gets keyboard input into CarAction
            keys = pygame.key.get_pressed()

            dir_x = 0.0
            dir_y = 0.0

            # Keyboard directional input: WASD or Arrow Keys
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                dir_x += 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                dir_x -= 1.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                dir_y += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                dir_y -= 1.0

            action.dir_x = dir_x
            action.dir_y = dir_y

            # Jump: Space bar or Gamepad Button A (button 0)
            action.jump = bool(keys[pygame.K_SPACE])

            # Rocket Boost: Shift, O, J, or Gamepad Button B / Trigger (buttons 1, 5)
            action.boost = bool(
                keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or
                keys[pygame.K_o] or keys[pygame.K_j]
            )

        elif self.control_type == "random":
            if self.sim.match_time % 0.5 < 0.001:
                self.random_action = CarAction(dir_x={0: 0.0, 1: 1.0, 2: -1.0}[np.random.randint(3)], 
                                   dir_y={0: 0.0, 1: 1.0, 2: -1.0}[np.random.randint(3)],
                                   jump={0: False, 1: True}[np.random.randint(2)],
                                   boost={0: False, 1: True}[np.random.randint(2)])
            action = self.random_action

        elif self.control_type == "level3":
            self.bot = HeuristicBot(team=self.team)
            action = self.bot.compute_action(self.sim)

        return action

    def reset(self):
        if self.control_type == "level3":
            self.bot.reset()