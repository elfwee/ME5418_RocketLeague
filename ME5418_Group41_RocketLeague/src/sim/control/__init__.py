"""Car controllers: every source of car input implements `Controller` (`get_action()`, `reset()`).

HumanController      keyboard
HeuristicController  rule-based HeuristicBot
IdleController       no input
RandomController     random inputs every 0.5 s
PolicyController     a trained checkpoint (in `src.sim.control.policy`, imported on demand
                     so the sim does not need torch unless a policy plays)
"""

from src.sim.control.base import Controller
from src.sim.control.human import HumanController
from src.sim.control.scripted import HeuristicController, IdleController, RandomController

CONTROL_TYPES = ("none", "passive", "human", "random", "heuristic_bot")


def make_controller(control_type: str = "none", team: str = "blue", sim=None) -> Controller:
    """A controller by name: 'none' / 'passive' (idle), 'human', 'random' or 'heuristic_bot'."""
    if control_type == "human":
        return HumanController()
    if control_type == "random":
        return RandomController(sim)
    if control_type == "heuristic_bot":
        return HeuristicController(team, sim)
    if control_type in ("none", "passive"):
        return IdleController()
    raise ValueError(f"Unknown control_type '{control_type}'; choose from {', '.join(CONTROL_TYPES)}.")


# Former name of the factory (it used to be a class dispatching on `control_type`)
CarControl = make_controller

__all__ = [
    "CONTROL_TYPES",
    "CarControl",
    "Controller",
    "HeuristicController",
    "HumanController",
    "IdleController",
    "RandomController",
    "make_controller",
]
