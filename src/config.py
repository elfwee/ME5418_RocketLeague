"""Simulation and visualization configuration parameters for 2D Rocket League."""
import math

# --- Field Dimensions (SI meters) ---
FIELD_WIDTH = 26.0         # Playable floor width between goal lines (meters)
FIELD_HEIGHT = 15.0        # Arena ceiling height (meters)
CORNER_RADIUS = 2.5        # Fillet radius for arena corners (meters)
GOAL_DEPTH = 3.75           # Depth of recessed goal pockets (meters) - 1.5x of 2.5m
GOAL_CORNER_RADIUS = 1.2    # Fillet radius for inside corners of the goal pocket (meters)
ARENA_SEGMENT_RADIUS = 0.06  # Collision thickness of arena boundary segments (meters)

# --- Elevated Goal Geometry ---
# Goal mouth is elevated high so ground-rolling balls hit the lower wall
# and bounce vertically (90 deg) across the front of the goal
# Goal opening height: 5.28m (1.2x of original 4.4m)
GOAL_BOTTOM_Y = 4.36        # Lower crossbar elevation above floor (meters)
GOAL_TOP_Y = 9.64           # Upper crossbar elevation above floor (meters)

# --- Canvas & Scaling ---
MARGIN_X = 4.5             # Margin to encompass expanded goal pockets on screen
MARGIN_Y = 1.5             # Margin to pad top and bottom
TOTAL_WIDTH = FIELD_WIDTH + 2.0 * MARGIN_X    # 35.0 m
TOTAL_HEIGHT = FIELD_HEIGHT + 2.0 * MARGIN_Y  # 18.0 m

PPM = 40.0                 # 40 pixels per meter
SCREEN_WIDTH = int(TOTAL_WIDTH * PPM)         # 1400 px
SCREEN_HEIGHT = int(TOTAL_HEIGHT * PPM)       # 720 px

# --- Physics Timing ---
SIM_HZ = 60                # Frame rate
PHYSICS_SUBSTEPS = 4       # Physics solver sub-steps per frame (240 Hz effective)
FIXED_DT = 1.0 / (SIM_HZ * PHYSICS_SUBSTEPS)
GRAVITY = (0.0, -28.0)     # Heavier downward gravity in m/s^2 for responsive falling
GRAVITY_MAG = abs(GRAVITY[1])
SOLVER_ITERATIONS = 14     # Pymunk constraint solver iterations (contact accuracy)

# --- Car Chassis (25% smaller, scale factor 0.75) ---
CAR_SCALE = 0.75           # Global scale factor for car & visual accessories
CAR_WIDTH = 1.575          # Chassis length (bumper-to-bumper) in meters (2.1 * 0.75)
CAR_HEIGHT = 0.6375        # Chassis height in meters (0.85 * 0.75)
CAR_MASS = 135.0           # Mass in kg (180 * 0.75, keeps 6:1 car-to-ball mass ratio)
CHASSIS_FRICTION = 0.50    # Chassis-vs-world friction (only bites on scrapes/wheelies)
CHASSIS_ELASTICITY = 0.25  # Chassis bounciness against arena surfaces

# --- Wheels & Raycast Suspension ---
WHEEL_RADIUS = 0.1875      # Visual/effective tyre radius in meters (0.25 * 0.75)
WHEEL_AXLE_Y = -0.285      # Local Y of both axles (-0.38 * 0.75)
WHEEL_REAR_X = -0.4875     # Local X of the driven (rear) axle (-0.65 * 0.75)
WHEEL_FRONT_X = 0.4125     # Local X of the front axle (0.55 * 0.75)

SUSPENSION_FREQ = 4.2      # Suspension natural frequency (Hz)
SUSPENSION_DAMPING_RATIO = 0.85  # <1 underdamped, 1 critical; 0.85 keeps it lively but settled
SUSPENSION_TRAVEL = 0.075   # Droop distance below static ride height in meters (0.10 * 0.75)
SUSPENSION_MAX_G = 8.0     # Per-wheel normal force ceiling, in units of m*g

_SUSPENSION_OMEGA = 2.0 * math.pi * SUSPENSION_FREQ
SUSPENSION_STIFFNESS = _SUSPENSION_OMEGA ** 2                              # 1/s^2 per unit mass
SUSPENSION_DAMPER = 2.0 * SUSPENSION_DAMPING_RATIO * _SUSPENSION_OMEGA     # 1/s per unit mass
SUSPENSION_STATIC_SAG = GRAVITY_MAG / SUSPENSION_STIFFNESS                 # Sag under own weight
SUSPENSION_REST_LEN = WHEEL_RADIUS + SUSPENSION_STATIC_SAG                 # Axle-to-ground at full droop
SUSPENSION_RAY_LEN = SUSPENSION_REST_LEN + SUSPENSION_TRAVEL               # Ground probe length
SUSPENSION_RAY_OFFSET = 0.20                                               # Inward probe offset above axle to prevent bottom-out misses
SUSPENSION_MAX_ACCEL = SUSPENSION_MAX_G * GRAVITY_MAG
# --- Traction ---
TIRE_GRIP = 2.2            # Coulomb friction coefficient of the tyre contact patch
CAR_STICKY_ACCEL = 16.0    # Downforce along the surface normal while grounded (m/s^2)
CAR_DRIVE_ACCEL = 24.0     # Ground drive acceleration request (m/s^2), grip-limited
CAR_COAST_DECEL = 9.0      # Engine-braking deceleration when throttle is released (m/s^2)
CAR_MAX_GROUND_SPEED = 14.5# Top ground speed (m/s)
CAR_WHEELIE_SPEED_FACTOR = 0.10 # Speed multiplier when in wheelie/stoppie (10% of normal ground speed)
CAR_WHEELIE_PITCH_THRESHOLD = 0.25 # Sine of pitch angle (~14.5 deg) required to trigger wheelie speed cap
WHEEL_MIN_NORMAL_DOT = 0.15# Reject ray hits whose normal is not roughly under the car
SURFACE_ALIGN_MIN_DOT = 0.20  # Only auto-align to surfaces no steeper than ~78 degrees

# Equilibrium suspension compression once weight and downforce are both carried; used to
# spawn the car exactly at its settled ride height instead of dropping it in.
SUSPENSION_LOADED_SAG = (GRAVITY_MAG + CAR_STICKY_ACCEL) / SUSPENSION_STIFFNESS
CAR_RIDE_HEIGHT = SUSPENSION_REST_LEN - SUSPENSION_LOADED_SAG - WHEEL_AXLE_Y

# --- Kickoff Spawn Positions (meters along X on Blue side) ---
CAR_SPAWN_X_DEFENSIVE = 8.5   # Rear defensive kickoff position on Blue side
CAR_SPAWN_X_ATTACK = 13.5     # Forward attacking kickoff position on Blue side

# --- Attitude Control (rate + acceleration limited, so it cannot overshoot or ring) ---
# omega_target = clamp(angle_error * RATE_GAIN, +-omega_max); alpha = clamp(d omega, +-alpha_max).
# Overshoot-free as long as RATE_GAIN <= 2 * alpha_max / omega_max.
CAR_STEER_RATE_GAIN = 14.0      # 1/s, how aggressively heading error becomes turn rate
CAR_GROUND_ANGULAR_SPEED = 9.0  # rad/s cap while wheels are loaded
CAR_GROUND_ANGULAR_ACCEL = 110.0# rad/s^2 cap while wheels are loaded
CAR_AIR_ANGULAR_SPEED = 8.0     # rad/s cap for air control
CAR_AIR_ANGULAR_ACCEL = 70.0    # rad/s^2 cap for air control
CAR_AIR_SPIN_DECAY_TAU = 0.45   # Exponential spin decay time constant in air (s)
CAR_PITCH_INPUT_THRESHOLD = 0.35# |dir_y| above this overrides ground-contour following
INPUT_DEADZONE = 0.15           # Below this input magnitude no heading is commanded
THROTTLE_DEADZONE = 0.08        # Below this |dir_x| the car coasts instead of driving
FACING_FLIP_THRESHOLD = 0.15    # |dir_x| needed to mirror the car left/right

# --- Jump & Rocket Boost ---
CAR_JUMP_SPEED = 9.5           # Instant jump velocity impulse for Jump 1 (m/s)
CAR_DOUBLE_JUMP_SPEED = 9.0    # Instant jump velocity impulse for neutral Jump 2 (m/s)
CAR_DODGE_SPEED = 10.5         # Velocity impulse for directional dodge Jump 2 (m/s)
CAR_DODGE_DURATION = 0.42      # Seconds to complete the 360-degree rotation during a dodge
CAR_BOOST_ACCEL = 42.0         # Boost acceleration (m/s^2) - net upward thrust is (42 - 28 = 14 m/s^2)
CAR_MAX_AIR_SPEED = 25.0   # Top aerial speed along the heading (m/s)
CAR_BOOST_SPEED_FADE = 3.0 # Thrust fades out over this speed band instead of clamping velocity
CAR_MAX_BOOST = 100.0      # Boost capacity (%)
CAR_BOOST_DRAIN = 30.0     # Boost consumption per second (%/s)
CAR_BOOST_REFILL = 80.0    # Boost recharge per second on surface (%/s)

# --- Turtle (upside-down) Recovery ---
# Implemented as a real flip: a vertical hop impulse plus a high-authority attitude sweep,
# instead of teleporting the rigid body.
TURTLE_UP_THRESHOLD = -0.55# Roof normal Y below this counts as inverted
TURTLE_PROBE_LEN = 0.80    # Downward probe from the centre of mass to detect a surface below (1.05 * 0.75)
TURTLE_TRIGGER_DELAY = 0.08# Seconds inverted before recovery fires
TURTLE_HOP_SPEED = 4.5     # Vertical hop velocity granted by the recovery flip (m/s)
TURTLE_FLIP_DURATION = 0.45# Seconds of high-authority attitude control after the hop
CAR_FLIP_ANGULAR_SPEED = 22.0   # rad/s cap during a recovery flip
CAR_FLIP_ANGULAR_ACCEL = 260.0  # rad/s^2 cap during a recovery flip

# --- Ball Parameters (25% smaller, scale factor 0.75) ---
BALL_RADIUS = 0.9375       # Ball radius in meters (1.25 * 0.75)
BALL_MASS = 22.5           # Ball mass in kg (30.0 * 0.75)
BALL_RESTITUTION = 0.80    # Elasticity of ball bounce
BALL_FRICTION = 0.35       # Surface friction
BALL_AIR_DRAG = 0.20       # Linear air resistance coefficient (1/s)
BALL_SPIN_DRAG = 0.35      # Angular air resistance coefficient (1/s)
BALL_MAX_SPEED = 38.0      # Hard cap on ball velocity (m/s)
BALL_MAX_SPIN = 30.0       # Hard cap on ball angular velocity (rad/s)
CAR_BALL_RESTITUTION = 0.60# Effective bounciness of a car-on-ball strike

# --- Collision Types (Pymunk categories) ---
COLLISION_ARENA = 1
COLLISION_CAR_BODY = 2
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
COLOR_INPUT_VECTOR = (168, 85, 247)      # Purple arrow for direction input vector
COLOR_VELOCITY_VECTOR = (59, 130, 246)   # Blue arrow for car velocity vector

# --- Opponent Car & Bot Settings ---
ENABLE_ORANGE_CAR = False                # Default flag: whether Orange car is present (enabled by default in main.py)
ORANGE_IS_BOT = True                     # Default flag: whether Orange car is AI-controlled
BOT_DODGE_STRIKE_DIST = 2.2              # Distance to ball to trigger dodge flip strike (m)
BOT_AERIAL_MIN_BOOST = 15.0              # Minimum boost required to commit to aerial launch
BOT_DEFENSE_ZONE_X = 22.0                # X coordinate dividing midfield from defense (m)
