"""Arena geometry generation with elevated goals and curved corner fillets."""
import math
from typing import List, Tuple
import pymunk
from src.config import (
    FIELD_WIDTH, FIELD_HEIGHT, CORNER_RADIUS,
    GOAL_DEPTH, GOAL_BOTTOM_Y, GOAL_TOP_Y,
    MARGIN_X, MARGIN_Y, ARENA_SEGMENT_RADIUS,
    COLLISION_ARENA, COLLISION_GOAL_SENSOR
)


class Arena:
    """Represents the bounded 2D arena including walls, ceiling, floor, and elevated goals."""

    def __init__(self, space: pymunk.Space):
        self.space = space
        self.segments: List[pymunk.Segment] = []
        self.goal_sensors: List[pymunk.Shape] = []

        # Key world coordinates
        self.x_left = MARGIN_X
        self.x_right = MARGIN_X + FIELD_WIDTH
        self.y_floor = MARGIN_Y
        self.y_ceil = MARGIN_Y + FIELD_HEIGHT

        self.goal_y_bot = MARGIN_Y + GOAL_BOTTOM_Y
        self.goal_y_top = MARGIN_Y + GOAL_TOP_Y

        self._build_arena()

    def _add_segment(self, p1: Tuple[float, float], p2: Tuple[float, float],
                     elasticity: float = 0.8, friction: float = 0.6,
                     radius: float = ARENA_SEGMENT_RADIUS) -> pymunk.Segment:
        """Helper to create and register a static collision segment."""
        seg = pymunk.Segment(self.space.static_body, p1, p2, radius)
        seg.elasticity = elasticity
        seg.friction = friction
        seg.collision_type = COLLISION_ARENA
        self.space.add(seg)
        self.segments.append(seg)
        return seg

    def _add_arc(self, center: Tuple[float, float], radius: float,
                 start_angle: float, end_angle: float, steps: int = 5,
                 elasticity: float = 0.8, friction: float = 0.6):
        """Create a rounded corner arc using a series of linear segments."""
        cx, cy = center
        angles = [start_angle + (end_angle - start_angle) * (i / steps) for i in range(steps + 1)]
        points = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
        for i in range(len(points) - 1):
            self._add_segment(points[i], points[i + 1], elasticity, friction)

    def _build_arena(self):
        """Construct the complete 2D arena with curved corners and elevated recessed goals."""
        xl = self.x_left
        xr = self.x_right
        yf = self.y_floor
        yc = self.y_ceil
        R = CORNER_RADIUS
        gd = GOAL_DEPTH
        g_bot = self.goal_y_bot
        g_top = self.goal_y_top

        # 1. Main Floor
        self._add_segment((xl + R, yf), (xr - R, yf), elasticity=0.7, friction=0.8)

        # 2. Main Ceiling
        self._add_segment((xl + R, yc), (xr - R, yc), elasticity=0.8, friction=0.5)

        # 3. Corner Fillets
        # Bottom-Left fillet: (xl + R, yf) -> (xl, yf + R)
        self._add_arc((xl + R, yf + R), R, math.pi * 1.5, math.pi, steps=5, elasticity=0.75, friction=0.7)
        # Top-Left fillet: (xl, yc - R) -> (xl + R, yc)
        self._add_arc((xl + R, yc - R), R, math.pi, math.pi * 0.5, steps=5, elasticity=0.8, friction=0.5)
        # Top-Right fillet: (xr - R, yc) -> (xr, yc - R)
        self._add_arc((xr - R, yc - R), R, math.pi * 0.5, 0.0, steps=5, elasticity=0.8, friction=0.5)
        # Bottom-Right fillet: (xr, yf + R) -> (xr - R, yf)
        self._add_arc((xr - R, yf + R), R, 0.0, -math.pi * 0.5, steps=5, elasticity=0.75, friction=0.7)

        # 4. Left Wall & Elevated Goal Structure
        # Lower wall beneath goal: (xl, yf + R) up to (xl, g_bot)
        # Balls rolling on the floor strike this vertical wall and bounce vertically in front of the goal
        self._add_segment((xl, yf + R), (xl, g_bot), elasticity=0.82, friction=0.4)
        # Left Goal Pocket:
        # Bottom crossbar/lip: (xl, g_bot) -> (xl - gd, g_bot)
        self._add_segment((xl, g_bot), (xl - gd, g_bot), elasticity=0.6, friction=0.5)
        # Back net: (xl - gd, g_bot) -> (xl - gd, g_top)
        self._add_segment((xl - gd, g_bot), (xl - gd, g_top), elasticity=0.4, friction=0.5)
        # Top crossbar: (xl - gd, g_top) -> (xl, g_top)
        self._add_segment((xl - gd, g_top), (xl, g_top), elasticity=0.6, friction=0.5)
        # Upper wall (backboard): (xl, g_top) -> (xl, yc - R)
        self._add_segment((xl, g_top), (xl, yc - R), elasticity=0.85, friction=0.4)

        # 5. Right Wall & Elevated Goal Structure
        # Lower wall beneath goal: (xr, yf + R) up to (xr, g_bot)
        self._add_segment((xr, yf + R), (xr, g_bot), elasticity=0.82, friction=0.4)
        # Right Goal Pocket:
        # Bottom crossbar/lip: (xr, g_bot) -> (xr + gd, g_bot)
        self._add_segment((xr, g_bot), (xr + gd, g_bot), elasticity=0.6, friction=0.5)
        # Back net: (xr + gd, g_bot) -> (xr + gd, g_top)
        self._add_segment((xr + gd, g_bot), (xr + gd, g_top), elasticity=0.4, friction=0.5)
        # Top crossbar: (xr + gd, g_top) -> (xr, g_top)
        self._add_segment((xr + gd, g_top), (xr, g_top), elasticity=0.6, friction=0.5)
        # Upper wall (backboard): (xr, g_top) -> (xr, yc - R)
        self._add_segment((xr, g_top), (xr, yc - R), elasticity=0.85, friction=0.4)

        # 6. Goal Sensors (Non-physical triggers inside the goal pocket)
        self._add_goal_sensor((xl - gd, g_bot), (xl, g_top), "left")
        self._add_goal_sensor((xr, g_bot), (xr + gd, g_top), "right")

    def _add_goal_sensor(self, p_min: Tuple[float, float], p_max: Tuple[float, float], team: str):
        """Invisible sensor box inside the goal pocket."""
        bb = pymunk.BB(p_min[0], p_min[1], p_max[0], p_max[1])
        sensor = pymunk.Poly(self.space.static_body, [
            (bb.left, bb.bottom), (bb.right, bb.bottom),
            (bb.right, bb.top), (bb.left, bb.top)
        ])
        sensor.sensor = True
        sensor.collision_type = COLLISION_GOAL_SENSOR
        sensor.team = team
        self.space.add(sensor)
        self.goal_sensors.append(sensor)
