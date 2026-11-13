"""The stylised Octane-like car sprite: wheels, spoiler, chassis, cabin, lights and boost flame."""

from collections.abc import Callable

import pygame

from src.config.render_config import (
    COLOR_BOOST_FLAME,
    COLOR_CABIN_EDGE,
    COLOR_CABIN_GLASS,
    COLOR_CAR_ACCENT,
    COLOR_CAR_BLUE,
    COLOR_CAR_DARK,
    COLOR_CAR_ORANGE,
    COLOR_CAR_ORANGE_ACCENT,
    COLOR_CAR_ORANGE_DARK,
    COLOR_FLAME_CORE,
    COLOR_GRILLE,
    COLOR_HEADLIGHT,
    COLOR_RIM,
    COLOR_RIM_HUB,
    COLOR_SPOILER,
    COLOR_SPOILER_STRUT,
    COLOR_TAILLIGHT,
    COLOR_TIRE_EDGE,
    COLOR_WHEELS,
    PPM,
)
from src.config.sim_config import CAR_SCALE, WHEEL_RADIUS
from src.sim.core.car import Car


def draw_car(screen: pygame.Surface, car: Car, to_screen: Callable[[float, float], tuple[int, int]]):
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
        sw = to_screen(w_pos.x, w_pos.y)
        tire_r = int(WHEEL_RADIUS * PPM)
        rim_r = int(WHEEL_RADIUS * 0.52 * PPM)

        # Outer tire
        pygame.draw.circle(screen, COLOR_WHEELS, sw, tire_r)
        pygame.draw.circle(screen, COLOR_TIRE_EDGE, sw, tire_r, width=2)
        # Inner rim
        pygame.draw.circle(screen, COLOR_RIM, sw, rim_r)
        pygame.draw.circle(screen, COLOR_RIM_HUB, sw, max(2, int(rim_r * 0.4)))

    # 2. Rear Spoiler / Wing (elevated on dual struts at rear of car)
    cs = CAR_SCALE
    for sx in (-0.90, -0.65):
        strut_base = body.local_to_world((sx * cs, 0.20 * cs * facing))
        strut_top = body.local_to_world((sx * cs, 0.48 * cs * facing))
        pygame.draw.line(
            screen,
            COLOR_SPOILER_STRUT,
            to_screen(strut_base.x, strut_base.y),
            to_screen(strut_top.x, strut_top.y),
            2,
        )

    # Wing blade
    wing_p1 = body.local_to_world((-1.05 * cs, 0.50 * cs * facing))
    wing_p2 = body.local_to_world((-0.55 * cs, 0.48 * cs * facing))
    s_w1 = to_screen(wing_p1.x, wing_p1.y)
    s_w2 = to_screen(wing_p2.x, wing_p2.y)
    pygame.draw.line(screen, COLOR_SPOILER, s_w1, s_w2, 4)

    # 3. Main Tapered Chassis Polygon
    screen_verts = []
    for v in car.active_chassis_vertices:
        w = body.local_to_world(v)
        screen_verts.append(to_screen(w.x, w.y))

    pygame.draw.polygon(screen, body_col, screen_verts)
    pygame.draw.polygon(screen, accent_col, screen_verts, width=2)

    # Lower body dark rocker panel
    dark_p1 = body.local_to_world((-0.95 * cs, -0.30 * cs * facing))
    dark_p2 = body.local_to_world((0.90 * cs, -0.30 * cs * facing))
    dark_p3 = body.local_to_world((0.85 * cs, -0.15 * cs * facing))
    dark_p4 = body.local_to_world((-0.95 * cs, -0.15 * cs * facing))
    s_dp = [to_screen(p.x, p.y) for p in [dark_p1, dark_p2, dark_p3, dark_p4]]
    pygame.draw.polygon(screen, dark_col, s_dp)

    # 4. Cockpit Cabin Window (Glass windshield with cyan tint)
    c1 = body.local_to_world((-0.30 * cs, 0.36 * cs * facing))
    c2 = body.local_to_world((0.30 * cs, 0.34 * cs * facing))
    c3 = body.local_to_world((0.40 * cs, 0.12 * cs * facing))
    c4 = body.local_to_world((-0.25 * cs, 0.12 * cs * facing))
    s_cabin = [to_screen(p.x, p.y) for p in [c1, c2, c3, c4]]
    pygame.draw.polygon(screen, COLOR_CABIN_GLASS, s_cabin)
    pygame.draw.polygon(screen, COLOR_CABIN_EDGE, s_cabin, width=2)

    # 5. Headlight & Front Grille
    headlight_pos = body.local_to_world((1.02 * cs, 0.05 * cs * facing))
    s_hl = to_screen(headlight_pos.x, headlight_pos.y)
    pygame.draw.circle(screen, COLOR_HEADLIGHT, s_hl, 3)

    g_top = body.local_to_world((1.04 * cs, 0.16 * cs * facing))
    g_bot = body.local_to_world((1.04 * cs, -0.22 * cs * facing))
    sg_top = to_screen(g_top.x, g_top.y)
    sg_bot = to_screen(g_bot.x, g_bot.y)
    pygame.draw.line(screen, COLOR_GRILLE, sg_top, sg_bot, 2)

    # Taillight
    tail_pos = body.local_to_world((-1.00 * cs, 0.05 * cs * facing))
    s_tl = to_screen(tail_pos.x, tail_pos.y)
    pygame.draw.circle(screen, COLOR_TAILLIGHT, s_tl, 3)

    # 6. Rocket Boost Exhaust Flame
    if car.is_boosting:
        flame_tip_w = body.local_to_world(((-1.02 - 1.25) * cs, 0.0))
        flame_top_w = body.local_to_world((-1.02 * cs, 0.22 * cs * facing))
        flame_bot_w = body.local_to_world((-1.02 * cs, -0.22 * cs * facing))

        s_tip = to_screen(flame_tip_w.x, flame_tip_w.y)
        s_top = to_screen(flame_top_w.x, flame_top_w.y)
        s_bot = to_screen(flame_bot_w.x, flame_bot_w.y)

        pygame.draw.polygon(screen, COLOR_BOOST_FLAME, [s_top, s_tip, s_bot])
        # Inner white flame core
        core_tip_w = body.local_to_world(((-1.02 - 0.65) * cs, 0.0))
        s_core_tip = to_screen(core_tip_w.x, core_tip_w.y)
        pygame.draw.polygon(screen, COLOR_FLAME_CORE, [s_top, s_core_tip, s_bot])
