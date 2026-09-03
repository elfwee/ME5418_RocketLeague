"""Simulation and visualization configuration parameters for 2D Rocket League."""
import math

# --- Field Dimensions (SI meters) ---
FIELD_WIDTH = 26.0         # Playable floor width between goal lines (meters)
FIELD_HEIGHT = 15.0        # Arena ceiling height (meters)
CORNER_RADIUS = 2.5        # Fillet radius for arena corners (meters)
GOAL_DEPTH = 2.5           # Depth of recessed goal pockets (meters)

# --- Elevated Goal Geometry ---
# Goal mouth is elevated high so ground-rolling balls hit the lower wall
# and bounce vertically (90 deg) across the front of the goal
GOAL_BOTTOM_Y = 4.8        # Lower crossbar elevation above floor (meters)
GOAL_TOP_Y = 9.2           # Upper crossbar elevation above floor (meters)

# --- Canvas & Scaling ---
MARGIN_X = 3.0             # Margin to encompass goal pockets on screen
MARGIN_Y = 1.5             # Margin to pad top and bottom
TOTAL_WIDTH = FIELD_WIDTH + 2.0 * MARGIN_X    # 32.0 m
TOTAL_HEIGHT = FIELD_HEIGHT + 2.0 * MARGIN_Y  # 18.0 m

PPM = 40.0                 # 40 pixels per meter
SCREEN_WIDTH = int(TOTAL_WIDTH * PPM)         # 1280 px
SCREEN_HEIGHT = int(TOTAL_HEIGHT * PPM)       # 720 px

# --- Physics Timing ---
SIM_HZ = 60                # Frame rate
PHYSICS_SUBSTEPS = 2       # Physics solver sub-steps per frame (120 Hz effective)
FIXED_DT = 1.0 / (SIM_HZ * PHYSICS_SUBSTEPS)
GRAVITY = (0.0, -28.0)     # Heavier downward gravity in m/s^2 for responsive falling
 
# --- Car Parameters ---
CAR_WIDTH = 2.1            # Chassis length (bumper-to-bumper) in meters
CAR_HEIGHT = 0.85          # Chassis height in meters
CAR_MASS = 1200.0          # Mass in kg
CAR_DRIVE_ACCEL = 24.0     # Ground drive acceleration (m/s^2)
CAR_MAX_GROUND_SPEED = 14.5# Top ground speed (m/s)
CAR_BOOST_ACCEL = 42.0     # Boost acceleration (m/s^2) - net upward thrust is (42 - 28 = 14 m/s^2)
CAR_MAX_AIR_SPEED = 25.0   # Top aerial speed (m/s)
CAR_JUMP_SPEED = 9.5       # Instant jump velocity impulse (m/s)
CAR_STEER_KP = 240.0       # Proportional steering torque gain for snappy Sideswipe rotation
CAR_STEER_KD = 26.0        # Derivative steering damping gain
CAR_MAX_BOOST = 100.0      # Boost capacity (%)
CAR_BOOST_DRAIN = 30.0     # Boost consumption per second (%/s)
CAR_BOOST_REFILL = 80.0    # Boost recharge per second on surface (%/s)
RWD_PITCH_CUTOFF_DEG = 75.0# Degrees pitch above which ground drive traction ceases

# --- Ball Parameters ---
BALL_RADIUS = 1.25         # Ball radius in meters
BALL_MASS = 350.0          # Ball mass in kg
BALL_RESTITUTION = 0.82    # Elasticity of ball bounce
BALL_FRICTION = 0.25       # Surface friction
BALL_AIR_DRAG = 0.005      # Air resistance damping
BALL_MAX_SPEED = 38.0      # Hard cap on ball velocity (m/s)

# --- Collision Types (Pymunk categories) ---
COLLISION_ARENA = 1
COLLISION_CAR_BODY = 2
COLLISION_CAR_WHEEL = 3
COLLISION_BALL = 4
COLLISION_GOAL_SENSOR = 5

# --- Color Palette (RGB) ---
COLOR_BG = (15, 18, 25)
COLOR_ARENA_BG = (22, 27, 38)
COLOR_FLOOR = (45, 55, 72)
COLOR_WALL = (90, 105, 128)
COLOR_CEILING = (45, 55, 72)
COLOR_GOAL_ORANGE = (237, 137, 54)
COLOR_GOAL_BLUE = (66, 153, 225)
COLOR_CAR_BLUE = (49, 130, 206)
COLOR_CAR_DARK = (26, 54, 93)
COLOR_CAR_ACCENT = (237, 242, 247)
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
COLOR_INPUT_VECTOR = (168, 85, 247)      # Purple arrow for direction input vector
COLOR_VELOCITY_VECTOR = (59, 130, 246)   # Blue arrow for car velocity vector
