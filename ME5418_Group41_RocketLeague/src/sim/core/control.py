import pygame
import numpy as np
from src.sim.core.actions import CarAction
from src.sim.ai.heuristic_bot import HeuristicBot
from src.sim.core.simulation import Simulation
from typing import Dict, Any, Optional, Tuple


class CarControl:
    """Controller wrapper dispatching car actions by control source.

    - control_type:
        - passive: returns empty action
        - human: enables keyboard control
        - random: samples a random action every 0.5s
        - heuristic_bot: uses HeuristicBot (HZISM)
    - team:
        - orange
        - blue
    - sim: Simulation needed for HeuristicBot
    """

    def __init__(self, control_type: str = "none", team: str = "blue", sim: Optional[Simulation] = None):
        self.control_type = control_type
        self.sim = sim
        self.team = team
        self.bot = HeuristicBot(team=self.team) if self.control_type == "heuristic_bot" else None
        self.action = CarAction()

    def get_action(self):
        if self.control_type == "passive":
            self.action = CarAction(dir_x=0, dir_y=0, jump=False, boost=False)

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

            self.action.dir_x = dir_x
            self.action.dir_y = dir_y

            # Jump: Space bar or Gamepad Button A (button 0)
            self.action.jump = bool(keys[pygame.K_SPACE])

            # Rocket Boost: Shift, O, J, or Gamepad Button B / Trigger (buttons 1, 5)
            self.action.boost = bool(
                keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_o] or keys[pygame.K_j]
            )

        elif self.control_type == "random":
            if self.sim is not None and self.sim.match_time % 0.5 < 0.001:
                self.random_action = CarAction(
                    dir_x={0: 0.0, 1: 1.0, 2: -1.0}[np.random.randint(3)],
                    dir_y={0: 0.0, 1: 1.0, 2: -1.0}[np.random.randint(3)],
                    jump={0: False, 1: True}[np.random.randint(2)],
                    boost={0: False, 1: True}[np.random.randint(2)],
                )
            self.action = self.random_action

        elif self.control_type == "heuristic_bot":
            if self.bot is None:
                self.bot = HeuristicBot(team=self.team)
            self.action = self.bot.compute_action(self.sim)

        return self.action

    def reset(self):
        self.action = CarAction(dir_x=0, dir_y=0, jump=False, boost=False)
        if self.control_type == "heuristic_bot" and self.bot is not None:
            self.bot.reset()
