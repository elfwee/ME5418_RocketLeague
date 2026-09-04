"""Pygame renderer providing 2D graphics, vector indicators, and Octane car visuals."""
import math
from typing import Tuple, List
import pygame
from src.config import (
    TOTAL_WIDTH, TOTAL_HEIGHT, PPM,
    SCREEN_WIDTH, SCREEN_HEIGHT, WHEEL_RADIUS, GOAL_CORNER_RADIUS,
    CAR_SCALE,
    COLOR_BG, COLOR_ARENA_BG, COLOR_FLOOR, COLOR_WALL, COLOR_CEILING,
    COLOR_GOAL_ORANGE, COLOR_GOAL_BLUE,
    COLOR_CAR_BLUE, COLOR_CAR_DARK, COLOR_CAR_ACCENT,
    COLOR_CAR_ORANGE, COLOR_CAR_ORANGE_DARK, COLOR_CAR_ORANGE_ACCENT,
    COLOR_WHEELS, COLOR_RIM, COLOR_SPOILER,
    COLOR_HEADLIGHT, COLOR_TAILLIGHT, COLOR_BOOST_FLAME,
    COLOR_BALL, COLOR_BALL_ACCENT,
    COLOR_INPUT_VECTOR, COLOR_VELOCITY_VECTOR,
    COLOR_TEXT, COLOR_UI_BAR_BG, COLOR_BOOST_BAR
)
from src.core.car import Car
from src.core.simulation import Simulation


class Renderer:
    """Renders the physical world state onto a Pygame surface."""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("monospace", 15, bold=True)
        self.large_font = pygame.font.SysFont("monospace", 22, bold=True)
        # Reused scratch layer for alpha-blended overlays; allocating one per trail
        # segment costs ~25 full-screen surfaces every frame.
        self._alpha_layer = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert SI coordinates (meters, +Y up) to screen pixels (+Y down)."""
        sx = int(x * PPM)
        sy = int((TOTAL_HEIGHT - y) * PPM)
        return (sx, sy)

    def render(self, sim: Simulation):
        """Render complete simulation frame: arena, vectors, ball, cars, and HUD."""
        self.screen.fill(COLOR_BG)

        self._draw_arena(sim)
        self._draw_ball(sim)
        self._draw_car(sim.car)
        if sim.car_orange is not None:
            self._draw_car(sim.car_orange)
        self._draw_vector_indicators(sim)
        self._draw_hud(sim)

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

            # Soft translucent goal pocket fill with smooth inside corners
            goal_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            radii = {
                "border_top_left_radius": r_px if is_left else 0,
                "border_bottom_left_radius": r_px if is_left else 0,
                "border_top_right_radius": 0 if is_left else r_px,
                "border_bottom_right_radius": 0 if is_left else r_px,
            }
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

            self.screen.blit(goal_surf, (rect.x, rect.y))
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
                pygame.draw.line(self._alpha_layer, (*COLOR_BALL_ACCENT, alpha),
                                 points[i], points[i + 1], width=width)
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
        """Draw stylized Octane-like car with tapered body, wheels, cabin glass, spoiler, lights."""
        body = car.body
        facing = car.facing_x
        is_orange = car.team == "orange"
        body_col = COLOR_CAR_ORANGE if is_orange else COLOR_CAR_BLUE
        accent_col = COLOR_CAR_ORANGE_ACCENT if is_orange else COLOR_CAR_ACCENT
        dark_col = COLOR_CAR_ORANGE_DARK if is_orange else COLOR_CAR_DARK

        # 1. Wheels with Rubber Tires and Inner Alloy Rims
        for local_pos in [car.rear_wheel_local, car.front_wheel_local]:
            w_pos = body.local_to_world(local_pos)
            sw = self.world_to_screen(w_pos.x, w_pos.y)
            tire_r = int(WHEEL_RADIUS * PPM)
            rim_r = int(WHEEL_RADIUS * 0.52 * PPM)

            # Outer tire
            pygame.draw.circle(self.screen, COLOR_WHEELS, sw, tire_r)
            pygame.draw.circle(self.screen, (15, 23, 42), sw, tire_r, width=2)
            # Inner rim
            pygame.draw.circle(self.screen, COLOR_RIM, sw, rim_r)
            pygame.draw.circle(self.screen, (241, 245, 249), sw, max(2, int(rim_r * 0.4)))

        # 2. Rear Spoiler / Wing (elevated on dual struts at rear of car)
        cs = CAR_SCALE
        strut1_base = body.local_to_world((-0.90 * cs, 0.20 * cs * facing))
        strut1_top = body.local_to_world((-0.90 * cs, 0.48 * cs * facing))
        strut2_base = body.local_to_world((-0.65 * cs, 0.20 * cs * facing))
        strut2_top = body.local_to_world((-0.65 * cs, 0.48 * cs * facing))

        s_s1_b = self.world_to_screen(strut1_base.x, strut1_base.y)
        s_s1_t = self.world_to_screen(strut1_top.x, strut1_top.y)
        s_s2_b = self.world_to_screen(strut2_base.x, strut2_base.y)
        s_s2_t = self.world_to_screen(strut2_top.x, strut2_top.y)

        pygame.draw.line(self.screen, (71, 85, 105), s_s1_b, s_s1_t, 2)
        pygame.draw.line(self.screen, (71, 85, 105), s_s2_b, s_s2_t, 2)

        # Wing blade
        wing_p1 = body.local_to_world((-1.05 * cs, 0.50 * cs * facing))
        wing_p2 = body.local_to_world((-0.55 * cs, 0.48 * cs * facing))
        s_w1 = self.world_to_screen(wing_p1.x, wing_p1.y)
        s_w2 = self.world_to_screen(wing_p2.x, wing_p2.y)
        pygame.draw.line(self.screen, COLOR_SPOILER, s_w1, s_w2, 4)

        # 3. Main Tapered Chassis Polygon
        screen_verts = []
        for v in car.active_chassis_vertices:
            w = body.local_to_world(v)
            screen_verts.append(self.world_to_screen(w.x, w.y))

        pygame.draw.polygon(self.screen, body_col, screen_verts)
        pygame.draw.polygon(self.screen, accent_col, screen_verts, width=2)

        # Lower body dark rocker panel
        dark_p1 = body.local_to_world((-0.95 * cs, -0.30 * cs * facing))
        dark_p2 = body.local_to_world((0.90 * cs, -0.30 * cs * facing))
        dark_p3 = body.local_to_world((0.85 * cs, -0.15 * cs * facing))
        dark_p4 = body.local_to_world((-0.95 * cs, -0.15 * cs * facing))
        s_dp = [self.world_to_screen(p.x, p.y) for p in [dark_p1, dark_p2, dark_p3, dark_p4]]
        pygame.draw.polygon(self.screen, dark_col, s_dp)

        # 4. Cockpit Cabin Window (Glass windshield with cyan tint)
        c1 = body.local_to_world((-0.30 * cs, 0.36 * cs * facing))
        c2 = body.local_to_world((0.30 * cs, 0.34 * cs * facing))
        c3 = body.local_to_world((0.40 * cs, 0.12 * cs * facing))
        c4 = body.local_to_world((-0.25 * cs, 0.12 * cs * facing))
        s_cabin = [self.world_to_screen(p.x, p.y) for p in [c1, c2, c3, c4]]
        pygame.draw.polygon(self.screen, (125, 211, 252), s_cabin)
        pygame.draw.polygon(self.screen, (224, 242, 254), s_cabin, width=2)

        # 5. Headlight & Forward Light Cone
        headlight_pos = body.local_to_world((1.02 * cs, -0.14 * cs * facing))
        s_hl = self.world_to_screen(headlight_pos.x, headlight_pos.y)
        pygame.draw.circle(self.screen, COLOR_HEADLIGHT, s_hl, 3)

        # Taillight
        tail_pos = body.local_to_world((-1.00 * cs, 0.05 * cs * facing))
        s_tl = self.world_to_screen(tail_pos.x, tail_pos.y)
        pygame.draw.circle(self.screen, COLOR_TAILLIGHT, s_tl, 3)

        # 6. Rocket Boost Exhaust Flame
        if car.is_boosting:
            tail_w = body.local_to_world((-1.02 * cs, 0.0))
            flame_tip_w = body.local_to_world(((-1.02 - 1.25) * cs, 0.0))
            flame_top_w = body.local_to_world((-1.02 * cs, 0.22 * cs * facing))
            flame_bot_w = body.local_to_world((-1.02 * cs, -0.22 * cs * facing))

            s_tail = self.world_to_screen(tail_w.x, tail_w.y)
            s_tip = self.world_to_screen(flame_tip_w.x, flame_tip_w.y)
            s_top = self.world_to_screen(flame_top_w.x, flame_top_w.y)
            s_bot = self.world_to_screen(flame_bot_w.x, flame_bot_w.y)

            pygame.draw.polygon(self.screen, COLOR_BOOST_FLAME, [s_top, s_tip, s_bot])
            # Inner white flame core
            core_tip_w = body.local_to_world(((-1.02 - 0.65) * cs, 0.0))
            s_core_tip = self.world_to_screen(core_tip_w.x, core_tip_w.y)
            pygame.draw.polygon(self.screen, (255, 255, 255), [s_top, s_core_tip, s_bot])

    def _draw_vector_indicators(self, sim: Simulation):
        """Draw purple direction input vector and blue heading/velocity arrow for cars."""
        cars = [sim.car]
        if sim.car_orange is not None:
            cars.append(sim.car_orange)

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
                    int(center_screen[1] - (dir_y / input_mag) * vec_len)
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

    def _draw_arrowhead(self, tip: Tuple[int, int], angle: float, color: Tuple[int, int, int], size: int = 9):
        """Helper to draw a filled equilateral arrowhead polygon."""
        wing_angle = math.pi * 0.82
        p1 = (int(tip[0] + size * math.cos(angle + wing_angle)),
              int(tip[1] + size * math.sin(angle + wing_angle)))
        p2 = (int(tip[0] + size * math.cos(angle - wing_angle)),
              int(tip[1] + size * math.sin(angle - wing_angle)))
        pygame.draw.polygon(self.screen, color, [tip, p1, p2])

    def _draw_hud(self, sim: Simulation):
        """Draw HUD: controls helper, vector legend, scoreboard, telemetry, and boost meter."""
        # --- Top-Center Match Scoreboard ---
        cx = SCREEN_WIDTH // 2
        score_box_w = 200
        score_box_h = 36
        score_box_x = cx - score_box_w // 2
        score_box_y = 14
        pygame.draw.rect(self.screen, (22, 27, 38), (score_box_x, score_box_y, score_box_w, score_box_h), border_radius=6)
        pygame.draw.rect(self.screen, (55, 65, 81), (score_box_x, score_box_y, score_box_w, score_box_h), width=2, border_radius=6)

        score_surf = self.large_font.render(f"{sim.score_blue}  -  {sim.score_orange}", True, COLOR_TEXT)
        blue_lbl = self.font.render("BLUE", True, COLOR_GOAL_BLUE)
        orange_lbl = self.font.render("ORANGE", True, COLOR_GOAL_ORANGE)

        s_rect = score_surf.get_rect(center=(cx, score_box_y + score_box_h // 2))
        b_rect = blue_lbl.get_rect(right=s_rect.left - 14, centery=s_rect.centery)
        o_rect = orange_lbl.get_rect(left=s_rect.right + 14, centery=s_rect.centery)

        self.screen.blit(blue_lbl, b_rect)
        self.screen.blit(score_surf, s_rect)
        self.screen.blit(orange_lbl, o_rect)

        # Orange Bot Status Pill (Top Center, below scoreboard)
        if sim.enable_orange and sim.car_orange is not None:
            state_label = sim.orange_bot.current_state if sim.orange_bot else "ACTIVE"
            bot_text = f"ORANGE BOT: {state_label}"
            bot_col = COLOR_CAR_ORANGE
        else:
            bot_text = "ORANGE BOT: OFF"
            bot_col = (156, 163, 175)
        bot_surf = self.font.render(bot_text, True, bot_col)
        b_box = bot_surf.get_rect(center=(cx, score_box_y + score_box_h + 16))
        pill_rect = pygame.Rect(b_box.x - 10, b_box.y - 3, b_box.width + 20, b_box.height + 6)
        pygame.draw.rect(self.screen, (22, 27, 38), pill_rect, border_radius=5)
        pygame.draw.rect(self.screen, (55, 65, 81), pill_rect, width=1, border_radius=5)
        self.screen.blit(bot_surf, b_box)

        # --- Top-Left Controls & Legend ---
        help_lines = [
            "Sideswipe 2D Physics Sandbox",
            "[WASD / Arrows] 2D Direction Vector",
            "[Space] Jump / Flip (or Turtle Recovery)",
            "[Shift / O] Rocket Boost",
            "[R] Reset Ball & Car",
            "[B] Toggle Orange Bot (ON/OFF)"
        ]
        y_offset = 15
        for line in help_lines:
            txt = self.font.render(line, True, COLOR_TEXT)
            self.screen.blit(txt, (20, y_offset))
            y_offset += 20

        # Vector indicator legend
        y_offset += 6
        p_txt = self.font.render("--- Purple Vector: Commanded Input (Target Hand)", True, COLOR_INPUT_VECTOR)
        b_txt = self.font.render("--> Blue Vector: Car Heading & Speed (Minute Hand)", True, COLOR_VELOCITY_VECTOR)
        self.screen.blit(p_txt, (20, y_offset))
        self.screen.blit(b_txt, (20, y_offset + 20))

        # --- Top-Right Car Telemetry ---
        car = sim.car
        speed_kmh = car.body.velocity.length * 3.6
        pitch_deg = abs(math.degrees(math.atan2(car.forward_vector[1], abs(car.forward_vector[0]))))
        if car.both_wheels_grounded:
            grounded_str = "GROUNDED"
            grounded_color = (72, 187, 120)
        elif car.can_ground_jump:
            grounded_str = "WHEELIE"
            grounded_color = (72, 187, 120)
        elif car.is_grounded and car.is_upright:
            grounded_str = "UPRIGHT"
            grounded_color = (246, 173, 85)
        else:
            grounded_str = "AIRBORNE"
            grounded_color = (236, 201, 75)

        speed_surf = self.font.render(f"Speed: {speed_kmh:5.1f} km/h", True, COLOR_TEXT)
        pitch_surf = self.font.render(f"Pitch: {pitch_deg:5.1f}°", True, COLOR_TEXT)
        facing_str = "RIGHT" if car.facing_x == 1 else "LEFT"
        facing_surf = self.font.render(f"Facing: {facing_str}", True, COLOR_TEXT)
        state_surf = self.font.render(f"State: {grounded_str}", True, grounded_color)

        self.screen.blit(speed_surf, (SCREEN_WIDTH - 220, 15))
        self.screen.blit(pitch_surf, (SCREEN_WIDTH - 220, 36))
        self.screen.blit(facing_surf, (SCREEN_WIDTH - 220, 57))
        self.screen.blit(state_surf, (SCREEN_WIDTH - 220, 78))

        if car._flip_active:
            jump2_str = "FLIP 360°"
            jump2_color = (246, 173, 85)
        elif car.has_jump2:
            jump2_str = "READY"
            jump2_color = (72, 187, 120)
        else:
            jump2_str = "DEPLETED"
            jump2_color = (156, 163, 175)

        jump2_surf = self.font.render(f"Jump 2: {jump2_str}", True, jump2_color)
        self.screen.blit(jump2_surf, (SCREEN_WIDTH - 220, 99))

        # --- Bottom-Right Boost Meter Gauge ---
        bar_w = 200
        bar_h = 22
        bar_x = SCREEN_WIDTH - bar_w - 25
        bar_y = SCREEN_HEIGHT - bar_h - 20

        # Background
        pygame.draw.rect(self.screen, COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        # Fill
        fill_ratio = max(0.0, min(1.0, car.boost_amount / 100.0))
        fill_w = int(bar_w * fill_ratio)
        if fill_w > 0:
            pygame.draw.rect(self.screen, COLOR_BOOST_BAR, (bar_x, bar_y, fill_w, bar_h), border_radius=5)
        pygame.draw.rect(self.screen, COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), width=2, border_radius=5)

        # Label
        boost_txt = self.font.render(f"BOOST: {int(car.boost_amount):3d}%", True, (0, 0, 0) if fill_ratio > 0.4 else COLOR_TEXT)
        self.screen.blit(boost_txt, (bar_x + 10, bar_y + 3))
