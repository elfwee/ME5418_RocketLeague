"""Rendering configuration: screen size, scale and the colour palette."""

from src.config.sim_config import TOTAL_HEIGHT, TOTAL_WIDTH

# --- Canvas & Scaling ---
PPM = 40.0  # 40 pixels per meter
SCREEN_WIDTH = int(TOTAL_WIDTH * PPM)  # 1400 px
SCREEN_HEIGHT = int(TOTAL_HEIGHT * PPM)  # 720 px
GIF_FRAME_SIZE = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)  # 700 x 360 px evaluation GIF frames

# --- Color Palette (RGB) ---
COLOR_BG = (15, 18, 25)
COLOR_WALL = (90, 105, 128)
COLOR_GOAL_ORANGE = (237, 137, 54)
COLOR_GOAL_BLUE = (66, 153, 225)
COLOR_CAR_BLUE = (49, 130, 206)
COLOR_CAR_DARK = (26, 54, 93)
COLOR_CAR_ACCENT = (237, 242, 247)
COLOR_CAR_ORANGE = (237, 137, 54)
COLOR_CAR_ORANGE_DARK = (154, 52, 18)
COLOR_CAR_ORANGE_ACCENT = (251, 191, 36)
COLOR_WHEELS = (30, 41, 59)
COLOR_RIM = (148, 163, 184)
COLOR_SPOILER = (30, 41, 59)
COLOR_HEADLIGHT = (254, 240, 138)
COLOR_TAILLIGHT = (239, 68, 68)
COLOR_BOOST_FLAME = (246, 173, 85)
COLOR_BALL = (247, 250, 252)
COLOR_BALL_ACCENT = (229, 62, 62)
COLOR_TEXT = (226, 232, 240)
COLOR_UI_BAR_BG = (45, 55, 72)
COLOR_BOOST_BAR = (236, 201, 75)

# --- Vector Visualizer Colors ---
COLOR_INPUT_VECTOR = (168, 85, 247)  # Purple arrow for direction input vector
COLOR_VELOCITY_VECTOR = (59, 130, 246)  # Blue arrow for car velocity vector

# --- Boundary Raycast Colors ---
COLOR_RAYCAST_BLUE = (34, 211, 238)  # Cyan/teal ray lines for Blue car
COLOR_RAYCAST_ORANGE = (251, 146, 60)  # Neon orange/amber ray lines for Orange car
COLOR_RAYCAST_HIT = (255, 255, 255)  # White hit point marker

# --- Car Sprite Details ---
COLOR_TIRE_EDGE = (15, 23, 42)
COLOR_RIM_HUB = (241, 245, 249)
COLOR_SPOILER_STRUT = (71, 85, 105)
COLOR_CABIN_GLASS = (125, 211, 252)
COLOR_CABIN_EDGE = (224, 242, 254)
COLOR_GRILLE = (30, 41, 59)
COLOR_FLAME_CORE = (255, 255, 255)
