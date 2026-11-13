"""Heads-up display: scoreboard, controls help, vector legend, Blue car telemetry, raycast button, boost meter."""

import math

import pygame

from src.config.render_config import (
    COLOR_BOOST_BAR,
    COLOR_GOAL_BLUE,
    COLOR_GOAL_ORANGE,
    COLOR_INPUT_VECTOR,
    COLOR_RAYCAST_BLUE,
    COLOR_TEXT,
    COLOR_UI_BAR_BG,
    COLOR_VELOCITY_VECTOR,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from src.sim.core.simulation import Simulation


def draw_hud(
    screen: pygame.Surface, font: pygame.font.Font, large_font: pygame.font.Font, sim: Simulation, show_raycasts: bool
) -> pygame.Rect:
    """Draw HUD: controls helper, vector legend, scoreboard, telemetry, and boost meter.

    Returns the raycast toggle button's rectangle (for click handling).
    """
    # --- Top-Center Match Scoreboard ---
    cx = SCREEN_WIDTH // 2
    score_box_w = 200
    score_box_h = 36
    score_box_x = cx - score_box_w // 2
    score_box_y = 14
    pygame.draw.rect(screen, (22, 27, 38), (score_box_x, score_box_y, score_box_w, score_box_h), border_radius=6)
    pygame.draw.rect(
        screen, (55, 65, 81), (score_box_x, score_box_y, score_box_w, score_box_h), width=2, border_radius=6
    )

    score_surf = large_font.render(f"{sim.score_blue}  -  {sim.score_orange}", True, COLOR_TEXT)
    blue_lbl = font.render("BLUE", True, COLOR_GOAL_BLUE)
    orange_lbl = font.render("ORANGE", True, COLOR_GOAL_ORANGE)

    s_rect = score_surf.get_rect(center=(cx, score_box_y + score_box_h // 2))
    b_rect = blue_lbl.get_rect(right=s_rect.left - 14, centery=s_rect.centery)
    o_rect = orange_lbl.get_rect(left=s_rect.right + 14, centery=s_rect.centery)

    screen.blit(blue_lbl, b_rect)
    screen.blit(score_surf, s_rect)
    screen.blit(orange_lbl, o_rect)

    # --- Top-Left Controls & Legend ---
    help_lines = [
        "Sideswipe 2D Physics Sandbox",
        "[WASD / Arrows] 2D Direction Vector",
        "[Space] Jump / Flip (or Turtle Recovery)",
        "[Shift / O] Rocket Boost",
        "[R] Reset Ball & Car",
        "[B] Toggle Orange Bot (ON/OFF)",
        "[L] Toggle Boundary Raycasts (ON/OFF)",
    ]
    y_offset = 15
    for line in help_lines:
        txt = font.render(line, True, COLOR_TEXT)
        screen.blit(txt, (20, y_offset))
        y_offset += 20

    # Vector indicator legend
    y_offset += 6
    p_txt = font.render("--- Purple Vector: Commanded Input (Target Hand)", True, COLOR_INPUT_VECTOR)
    b_txt = font.render("--> Blue Vector: Car Heading & Speed (Minute Hand)", True, COLOR_VELOCITY_VECTOR)
    r_txt = font.render("--- Cyan Lines: 45° Boundary Raycasts (Field Only)", True, COLOR_RAYCAST_BLUE)
    screen.blit(p_txt, (20, y_offset))
    screen.blit(b_txt, (20, y_offset + 20))
    screen.blit(r_txt, (20, y_offset + 40))

    # --- Top-Right Car Telemetry ---
    car = sim.blue_car
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

    speed_surf = font.render(f"Speed: {speed_kmh:5.1f} km/h", True, COLOR_TEXT)
    pitch_surf = font.render(f"Pitch: {pitch_deg:5.1f}°", True, COLOR_TEXT)
    facing_str = "RIGHT" if car.facing_x == 1 else "LEFT"
    facing_surf = font.render(f"Facing: {facing_str}", True, COLOR_TEXT)
    state_surf = font.render(f"State: {grounded_str}", True, grounded_color)

    screen.blit(speed_surf, (SCREEN_WIDTH - 220, 15))
    screen.blit(pitch_surf, (SCREEN_WIDTH - 220, 36))
    screen.blit(facing_surf, (SCREEN_WIDTH - 220, 57))
    screen.blit(state_surf, (SCREEN_WIDTH - 220, 78))

    if car._flip_active:
        jump2_str = "FLIP 360°"
        jump2_color = (246, 173, 85)
    elif car.has_jump2:
        jump2_str = "READY"
        jump2_color = (72, 187, 120)
    else:
        jump2_str = "DEPLETED"
        jump2_color = (156, 163, 175)

    jump2_surf = font.render(f"Jump 2: {jump2_str}", True, jump2_color)
    screen.blit(jump2_surf, (SCREEN_WIDTH - 220, 99))

    # Ball spin telemetry
    spin = sim.ball.body.angular_velocity
    if abs(spin) > 0.5:
        spin_dir = "CW" if spin < 0 else "CCW"
        spin_col = (237, 137, 54) if spin < 0 else (66, 153, 225)
        spin_surf = font.render(f"Ball: {spin_dir} ({spin:+5.1f} rad/s)", True, spin_col)
    else:
        spin_surf = font.render("Ball: NO SPIN (0.0)", True, (156, 163, 175))
    screen.blit(spin_surf, (SCREEN_WIDTH - 220, 120))

    # --- Top-Right Raycast Toggle Button (under telemetry) ---
    btn_x = SCREEN_WIDTH - 220
    btn_y = 146
    btn_w = 200
    btn_h = 26
    button_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    mouse_pos = pygame.mouse.get_pos()
    is_hovered = button_rect.collidepoint(mouse_pos)

    if show_raycasts:
        btn_bg = (18, 48, 52) if is_hovered else (15, 38, 42)
        border_col = (45, 226, 230) if is_hovered else COLOR_RAYCAST_BLUE
        text_col = (200, 250, 255) if is_hovered else COLOR_RAYCAST_BLUE
        btn_text = "[L] RAYCASTS: ON"
        border_w = 2
    else:
        btn_bg = (35, 42, 54) if is_hovered else (22, 27, 38)
        border_col = (148, 163, 184) if is_hovered else (75, 85, 99)
        text_col = (203, 213, 225) if is_hovered else (156, 163, 175)
        btn_text = "[L] RAYCASTS: OFF"
        border_w = 1

    pygame.draw.rect(screen, btn_bg, button_rect, border_radius=5)
    pygame.draw.rect(screen, border_col, button_rect, width=border_w, border_radius=5)
    btn_surf = font.render(btn_text, True, text_col)
    btn_rect = btn_surf.get_rect(center=button_rect.center)
    screen.blit(btn_surf, btn_rect)

    # --- Bottom-Right Boost Meter Gauge ---
    bar_w = 200
    bar_h = 22
    bar_x = SCREEN_WIDTH - bar_w - 25
    bar_y = SCREEN_HEIGHT - bar_h - 20

    # Background
    pygame.draw.rect(screen, COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
    # Fill
    fill_ratio = max(0.0, min(1.0, car.boost_amount / 100.0))
    fill_w = int(bar_w * fill_ratio)
    if fill_w > 0:
        pygame.draw.rect(screen, COLOR_BOOST_BAR, (bar_x, bar_y, fill_w, bar_h), border_radius=5)
    pygame.draw.rect(screen, COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), width=2, border_radius=5)

    # Label
    boost_txt = font.render(f"BOOST: {int(car.boost_amount):3d}%", True, (0, 0, 0) if fill_ratio > 0.4 else COLOR_TEXT)
    screen.blit(boost_txt, (bar_x + 10, bar_y + 3))

    return button_rect
