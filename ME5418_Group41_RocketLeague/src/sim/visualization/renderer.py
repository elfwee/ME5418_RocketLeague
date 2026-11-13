"""Pygame renderer providing 2D graphics, vector indicators, and Octane car visuals."""

import math

import pygame

from src.config.render_config import (
    COLOR_BALL,
    COLOR_BALL_ACCENT,
    COLOR_BG,
    COLOR_GOAL_BLUE,
    COLOR_GOAL_ORANGE,
    COLOR_INPUT_VECTOR,
    COLOR_RAYCAST_BLUE,
    COLOR_RAYCAST_HIT,
    COLOR_RAYCAST_ORANGE,
    COLOR_VELOCITY_VECTOR,
    COLOR_WALL,
    PPM,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from src.config.sim_config import GOAL_CORNER_RADIUS, TOTAL_HEIGHT
from src.sim.core.car import Car
from src.sim.core.simulation import Simulation
from src.sim.visualization.car_sprite import draw_car
from src.sim.visualization.hud import draw_hud


class Renderer:
    """Renders the physical world state onto a Pygame surface."""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("monospace", 15, bold=True)
        self.large_font = pygame.font.SysFont("monospace", 22, bold=True)
        # Reused scratch layer for alpha-blended overlays; allocating one per trail
        # segment costs ~25 full-screen surfaces every frame.
        self._alpha_layer = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self._cached_goal_surfaces = {}
        self.show_raycasts: bool = False
        self.raycast_button_rect = pygame.Rect(0, 0, 0, 0)

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert SI coordinates (meters, +Y up) to screen pixels (+Y down)."""
        sx = int(x * PPM)
        sy = int((TOTAL_HEIGHT - y) * PPM)
        return (sx, sy)

    def render(self, sim: Simulation):
        """Render complete simulation frame: arena, raycasts, vectors, ball, cars, and HUD."""
        self.screen.fill(COLOR_BG)

        self._draw_arena(sim)
        self._draw_boundary_raycasts(sim)
        self._draw_ball(sim)
        self._draw_car(sim.blue_car)
        if sim.orange_car is not None:
            self._draw_car(sim.orange_car)
        self._draw_vector_indicators(sim)
        self._draw_hud(sim)

    def _draw_boundary_raycasts(self, sim: Simulation):
        """Draw 45-degree interval boundary raycasts from car center to field boundaries."""
        if not self.show_raycasts:
            return

        cars = [(sim.blue_car, COLOR_RAYCAST_BLUE)]
        if sim.enable_orange and sim.orange_car is not None:
            cars.append((sim.orange_car, COLOR_RAYCAST_ORANGE))

        for car, color in cars:
            raycasts = car.compute_boundary_raycasts()
            sp_start = self.world_to_screen(car.position[0], car.position[1])

            # Subtle origin circle at car center
            pygame.draw.circle(self.screen, color, sp_start, 4, width=1)

            for r in raycasts:
                hx, hy = r["hit_point"]
                sp_hit = self.world_to_screen(hx, hy)

                # Draw ray line from car center to field collision boundary
                pygame.draw.line(self.screen, color, sp_start, sp_hit, 1)

                # Draw hit point contact marker (white dot with colored halo)
                pygame.draw.circle(self.screen, COLOR_RAYCAST_HIT, sp_hit, 3)
                pygame.draw.circle(self.screen, color, sp_hit, 5, width=1)

    def _draw_arena(self, sim: Simulation):
        """Draw boundary segments, corner curves, center field marking, and elevated goal pockets."""
        arena = sim.arena

        # Subtle center court divider
        cx = (arena.x_left + arena.x_right) / 2.0
        sp_bot = self.world_to_screen(cx, arena.y_floor)
        sp_top = self.world_to_screen(cx, arena.y_ceil)
        pygame.draw.line(self.screen, (32, 40, 58), sp_bot, sp_top, 2)

        # Center kickoff circle
        center_screen = self.world_to_screen(cx, (arena.y_floor + arena.y_ceil) / 2.0)
        pygame.draw.circle(self.screen, (32, 40, 58), center_screen, int(3.5 * PPM), width=2)

        # Draw physical boundary segments (floor, ceiling, fillets, backboards, lower walls)
        for seg in arena.segments:
            p1 = seg.a
            p2 = seg.b
            sp1 = self.world_to_screen(p1.x, p1.y)
            sp2 = self.world_to_screen(p2.x, p2.y)
            thickness = max(3, int(seg.radius * 2 * PPM))
            pygame.draw.line(self.screen, COLOR_WALL, sp1, sp2, thickness)

        # Highlight elevated recessed goal pockets with netting and smooth inside corners
        r_px = int(GOAL_CORNER_RADIUS * PPM)
        for sensor in arena.goal_sensors:
            bb = sensor.bb
            sp_tl = self.world_to_screen(bb.left, bb.top)
            sp_br = self.world_to_screen(bb.right, bb.bottom)
            rect = pygame.Rect(sp_tl[0], sp_tl[1], sp_br[0] - sp_tl[0], sp_br[1] - sp_tl[1])
            is_left = getattr(sensor, "team", "") == "left"
            color = COLOR_GOAL_BLUE if is_left else COLOR_GOAL_ORANGE

            # Soft translucent goal pocket fill with smooth inside corners (cached to avoid per-frame allocations)
            cache_key = (is_left, rect.width, rect.height)
            radii = {
                "border_top_left_radius": r_px if is_left else 0,
                "border_bottom_left_radius": r_px if is_left else 0,
                "border_top_right_radius": 0 if is_left else r_px,
                "border_bottom_right_radius": 0 if is_left else r_px,
            }
            if cache_key not in self._cached_goal_surfaces:
                goal_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                pygame.draw.rect(goal_surf, (*color, 45), pygame.Rect(0, 0, rect.width, rect.height), **radii)

                # Netting grid lines masked to the rounded goal pocket
                net_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                for gy in range(0, rect.height, 16):
                    pygame.draw.line(net_surf, (*color, 75), (0, gy), (rect.width, gy), 1)
                for gx in range(0, rect.width, 16):
                    pygame.draw.line(net_surf, (*color, 55), (gx, 0), (gx, rect.height), 1)

                mask_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                pygame.draw.rect(mask_surf, (255, 255, 255, 255), pygame.Rect(0, 0, rect.width, rect.height), **radii)
                net_surf.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                goal_surf.blit(net_surf, (0, 0))
                self._cached_goal_surfaces[cache_key] = goal_surf

            self.screen.blit(self._cached_goal_surfaces[cache_key], (rect.x, rect.y))
            pygame.draw.rect(self.screen, color, rect, width=2, **radii)

    def _draw_ball(self, sim: Simulation):
        """Draw ball, rotation seam indicator, and trailing trajectory path."""
        ball = sim.ball
        bx, by = ball.position
        center_screen = self.world_to_screen(bx, by)
        r_screen = int(ball.radius * PPM)

        # Motion trail
        if len(ball.trail) > 1:
            points = [self.world_to_screen(tx, ty) for tx, ty in ball.trail]
            self._alpha_layer.fill((0, 0, 0, 0))
            width = max(1, int(r_screen * 0.35))
            for i in range(len(points) - 1):
                alpha = int(160 * (i / len(points)))
                pygame.draw.line(self._alpha_layer, (*COLOR_BALL_ACCENT, alpha), points[i], points[i + 1], width=width)
            self.screen.blit(self._alpha_layer, (0, 0))

        # Ball body
        pygame.draw.circle(self.screen, COLOR_BALL, center_screen, r_screen)
        pygame.draw.circle(self.screen, (100, 116, 139), center_screen, r_screen, width=3)

        # Ball rotation seam (cross pattern)
        angle = ball.angle
        for offset in [0.0, math.pi / 2.0]:
            a = angle + offset
            dx = math.cos(a) * (r_screen * 0.85)
            dy = -math.sin(a) * (r_screen * 0.85)
            p1 = (int(center_screen[0] - dx), int(center_screen[1] - dy))
            p2 = (int(center_screen[0] + dx), int(center_screen[1] + dy))
            pygame.draw.line(self.screen, COLOR_BALL_ACCENT, p1, p2, 3)

    def _draw_car(self, car: Car):
        draw_car(self.screen, car, self.world_to_screen)

    def _draw_vector_indicators(self, sim: Simulation):
        """Draw purple direction input vector and blue heading/velocity arrow for cars."""
        cars = [sim.blue_car]
        if sim.orange_car is not None:
            cars.append(sim.orange_car)

        for car in cars:
            cx, cy = car.position
            center_screen = self.world_to_screen(cx, cy)
            is_orange = car.team == "orange"

            # --- 1. Direction Input Vector ---
            dir_x, dir_y = car.last_input_vector
            input_mag = math.hypot(dir_x, dir_y)

            if input_mag > 0.08:
                vec_len = int(input_mag * 58.0)
                # Screen Y is inverted relative to physics Y
                target_screen = (
                    int(center_screen[0] + (dir_x / input_mag) * vec_len),
                    int(center_screen[1] - (dir_y / input_mag) * vec_len),
                )
                input_col = (251, 191, 36) if is_orange else COLOR_INPUT_VECTOR
                pygame.draw.line(self.screen, input_col, center_screen, target_screen, 3)
                angle_screen = math.atan2(-(dir_y / input_mag), (dir_x / input_mag))
                self._draw_arrowhead(target_screen, angle_screen, input_col, size=9)

            # --- 2. Heading / Velocity Arrow ---
            fwd_x, fwd_y = car.forward_vector
            arrow_angle_screen = math.atan2(-fwd_y, fwd_x)

            vx, vy = car.body.velocity
            speed = math.hypot(vx, vy)

            # Base length 54px extends beyond chassis (42px) so it is always visible; scales with speed
            arrow_len = min(110, max(54, int(54 + speed * 2.2)))

            tip_x = int(center_screen[0] + math.cos(arrow_angle_screen) * arrow_len)
            tip_y = int(center_screen[1] + math.sin(arrow_angle_screen) * arrow_len)

            vel_col = (237, 137, 54) if is_orange else COLOR_VELOCITY_VECTOR
            pygame.draw.line(self.screen, vel_col, center_screen, (tip_x, tip_y), 3)
            self._draw_arrowhead((tip_x, tip_y), arrow_angle_screen, vel_col, size=10)

    def _draw_arrowhead(self, tip: tuple[int, int], angle: float, color: tuple[int, int, int], size: int = 9):
        """Helper to draw a filled equilateral arrowhead polygon."""
        wing_angle = math.pi * 0.82
        p1 = (int(tip[0] + size * math.cos(angle + wing_angle)), int(tip[1] + size * math.sin(angle + wing_angle)))
        p2 = (int(tip[0] + size * math.cos(angle - wing_angle)), int(tip[1] + size * math.sin(angle - wing_angle)))
        pygame.draw.polygon(self.screen, color, [tip, p1, p2])

    def _draw_hud(self, sim: Simulation):
        self.raycast_button_rect = draw_hud(self.screen, self.font, self.large_font, sim, self.show_raycasts)

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Handle mouse click events on UI elements. Returns True if a button was clicked."""
        if hasattr(self, "raycast_button_rect") and self.raycast_button_rect.collidepoint(pos):
            self.show_raycasts = not self.show_raycasts
            return True
        return False
