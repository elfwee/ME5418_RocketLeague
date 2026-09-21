"""Action definitions for 2D direction vector and car control."""
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CarAction:
    """Car control action inputs for simulation steps.

    Attributes:
        dir_x: Horizontal direction input vector component (-1.0 to 1.0).
        dir_y: Vertical direction input vector component (-1.0 to 1.0).
        jump: Boolean trigger to initiate jump from surface or flip.
        boost: Boolean hold to fire directional rocket boosters.
    """
    dir_x: float = 0.0
    dir_y: float = 0.0
    jump: bool = False
    boost: bool = False

    def clamp(self) -> 'CarAction':
        """Clamp direction vector to unit magnitude if greater than 1."""
        mag = math.hypot(self.dir_x, self.dir_y)
        if mag > 1.0:
            self.dir_x /= mag
            self.dir_y /= mag
        self.jump = bool(self.jump)
        self.boost = bool(self.boost)
        return self

    @property
    def magnitude(self) -> float:
        """Magnitude of the 2D direction input vector (0.0 to 1.0)."""
        return min(1.0, math.hypot(self.dir_x, self.dir_y))

    @property
    def target_angle(self) -> float:
        """Target orientation angle in radians corresponding to (dir_x, dir_y)."""
        return math.atan2(self.dir_y, self.dir_x)

    # --- Backward compatibility accessors for older test cases ---
    @property
    def throttle(self) -> float:
        return self.dir_x

    @throttle.setter
    def throttle(self, val: float):
        self.dir_x = float(val)

    @property
    def pitch(self) -> float:
        return -self.dir_y

    @pitch.setter
    def pitch(self, val: float):
        self.dir_y = -float(val)
