"""The interface every car controller implements."""

from typing import Protocol

from src.sim.core.actions import CarAction


class Controller(Protocol):
    """Produces one car's input every sim frame."""

    def get_action(self) -> CarAction:
        """The CarAction for the coming frame."""
        ...

    def reset(self):
        """Forget per-kickoff state (called after goals and on manual resets)."""
        ...
