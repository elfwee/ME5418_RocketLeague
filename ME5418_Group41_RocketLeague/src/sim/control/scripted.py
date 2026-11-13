"""Controllers that need no learned policy: the heuristic bot, an idle car, and random inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.sim.ai.heuristic_bot import HeuristicBot
from src.sim.core.actions import CarAction

if TYPE_CHECKING:
    from src.sim.core.simulation import Simulation


class IdleController:
    """No input at all."""

    def get_action(self) -> CarAction:
        return CarAction(dir_x=0, dir_y=0, jump=False, boost=False)

    def reset(self):
        pass


class HeuristicController:
    """The rule-based `HeuristicBot` driving `team`'s car."""

    def __init__(self, team: str, sim: Simulation):
        self.team = team
        self.sim = sim
        self.bot = HeuristicBot(team=team)

    def get_action(self) -> CarAction:
        return self.bot.compute_action(self.sim)

    def reset(self):
        self.bot.reset()


class RandomController:
    """A random direction / jump / boost, re-drawn every 0.5 s of match time (from the global np.random)."""

    def __init__(self, sim: Simulation | None):
        self.sim = sim
        self.random_action = CarAction()

    def get_action(self) -> CarAction:
        if self.sim is not None and self.sim.match_time % 0.5 < 0.001:
            self.random_action = CarAction(
                dir_x={0: 0.0, 1: 1.0, 2: -1.0}[np.random.randint(3)],
                dir_y={0: 0.0, 1: 1.0, 2: -1.0}[np.random.randint(3)],
                jump={0: False, 1: True}[np.random.randint(2)],
                boost={0: False, 1: True}[np.random.randint(2)],
            )
        return self.random_action

    def reset(self):
        pass
