# Technical Architecture & Simulation Specification: Rocket League 2D (Sideswipe Physics Sandbox)

## 1. Executive Summary

This repository houses a high-fidelity, deterministic 2D side-scrolling physics simulation of Rocket League (styled after *Rocket League Sideswipe*). Designed as an advanced sandbox for robotics research, reinforcement learning (RL), and autonomous vehicle control, the platform models vehicle-ball dynamics, non-linear tire traction, two-stage jump-dodge ballistics, rocket boost aerodynamics, and an automated rule-based AI opponent.

The project represents a generational evolution from **V1** (a top-down planar search-and-push environment) to **V2** (a full vertical-plane rigid-body dynamical system with raycast spring-damper suspension, elevated goals, and aerial maneuvering).

```
+---------------------------------------------------------------------------------------+
|                                    ME5418 Sandbox                                     |
|                                                                                       |
|     +-------------------+       +---------------------+       +-----------------+     |
|     |  User Keyboard    | ----> |      CarAction      | <---- |   HeuristicBot  |     |
|     |  (WASD / Arrows)  |       | (dir_x,dir_y,j,bst) |       |     (HZISM)     |     |
|     +-------------------+       +---------------------+       +-----------------+     |
|                                            |                                          |
|                                            v                                          |
|                     +---------------------------------------------+                   |
|                     |             Simulation Engine               |                   |
|                     |    - Pymunk Space (Rigid Bodies, Shapes)    |                   |
|                     |    - 240 Hz Sub-stepping (60 Hz x 4 steps)  |                   |
|                     |    - Collision Arbiter & Goal Sensors       |                   |
|                     +---------------------------------------------+                   |
|                                     /             \                                   |
|                                    v               v                                  |
|                      +--------------------+  +--------------------+                   |
|                      |  Pygame Renderer   |  | State Dictionary   |                   |
|                      |  - Vectors (P/B)   |  |  - RL Observations |                   |
|                      |  - Octane Visuals  |  |  - Headless Eval   |                   |
|                      |  - HUD & Telemetry |  +--------------------+                   |
|                      +--------------------+                                           |
+---------------------------------------------------------------------------------------+
```

---

## 2. System Architecture & Modular Documentation Breakdown

The codebase is organized modularly under `src/`. For ease of navigation and specialized context, each subsystem includes its own comprehensive technical documentation:

* 📚 **[System & RL Environment (`src/README.md`)](src/README.md)**: Gymnasium environment API, 52-dimensional flat observation space, action spaces, reward shaping formulation, and control scheme.
* ⚙️ **[Physics & Vehicle Kinematics (`src/sim/core/README.md`)](src/sim/core/README.md)**: 240 Hz sub-stepping solver, raycast spring-damper suspension, Coulomb traction, attitude control, two-stage jump-dodge ballistics, rocket boost, ball aerodynamics, and elevated goal arena geometry.
* 🤖 **[Opponent AI Architecture (`src/sim/ai/README.md`)](src/sim/ai/README.md)**: Hierarchical Zone & Intercept State Machine (HZISM), kickoff tactics, shadow defense, zone goalie routine, and aerial save intercepts.
* 🎨 **[Visualization & HUD Telemetry (`src/sim/visualization/README.md`)](src/sim/visualization/README.md)**: 40 PPM Pygame rendering pipeline, procedural Octane chassis, vector debug visualizers, and live telemetry HUD.

```
ME5418_Group41_RocketLeague/
├── README.md                     # High-level project overview & quickstart
├── environment.yml               # Conda environment configuration
└── src/
    ├── __main__.py / cli.py      # `python -m src` command-line interface
    ├── README.md                 # Full module tree and Gymnasium RL documentation
    ├── config/                   # sim_config (physics, geometry), render_config, gym_config, nn_config
    ├── gym_env/                  # RocketLeagueEnv, action codec, rewards, ActionRepeat wrapper
    ├── neural_nets/              # DQN / ActorCritic networks, checkpoint loading, demos
    ├── learning_agents/          # Dueling DDQN and PPO agents, shared training loop, plotting
    ├── eval/                     # --eval --mix and --head_to_head matches
    └── sim/
        ├── game.py               # Interactive Pygame sandbox and headless benchmark runner
        ├── core/                 # actions, arena, ball, car, simulation, observation, spawns
        ├── control/              # Keyboard, heuristic bot, idle, random and policy controllers
        ├── ai/                   # Hierarchical Zone & Intercept State Machine (HZISM) AI
        └── visualization/        # Pygame renderer, car sprite, HUD, event handling
```

---

## 3. Core Physics & Mathematical Formulation

The simulation is powered by [Pymunk](http://www.pymunk.org/) (Python bindings for Chipmunk2D) configured with 60 solver iterations, deterministic step sizing, and zero global damping (drag is modeled explicitly at the entity level to prevent spurious energy dissipation).

### 3.1. Temporal Discretization & Solver Timing
* **Base Frame Rate ($f_{sim}$)**: $60\text{ Hz}$ ($\Delta t_{frame} = 16.67\text{ ms}$)
* **Sub-step Multiplier ($N_{sub}$)**: 4 sub-steps per frame
* **Effective Physics Rate**: $240\text{ Hz}$ ($\Delta t_{sub} = \frac{1}{240}\text{ s} \approx 4.167\text{ ms}$)
* **Gravity ($\vec{g}$)**: $(0.0, -28.0)\text{ m/s}^2$ (elevated downward gravity for snappy, arcade-authentic ballistics)

---

### 3.2. Vehicle Dynamics (`src/sim/core/car.py`)

The car is modeled not as a sliding bounding box, but as an active physical chassis supported by independent raycast suspension and ground traction.

#### A. Raycast Suspension System
Two independent downward rays are cast along the car's negative local up-axis ($-\vec{u}_{car}$) at the rear axle ($x = -0.4875\text{ m}$) and front axle ($x = 0.4125\text{ m}$):
$$\vec{x}_{start} = \vec{x}_{axle} + \vec{u}_{car} \cdot d_{offset}, \quad \vec{x}_{end} = \vec{x}_{axle} - \vec{u}_{car} \cdot L_{ray}$$

Each wheel acts as a damped spring along the surface contact normal $\hat{n}$:
$$F_{susp} = \frac{m}{2} \cdot \operatorname{clamp}\left( k \cdot \Delta x - c \cdot (\vec{v}_{axle} \cdot \hat{n}), \, 0, \, a_{max} \right)$$
* Natural Frequency ($\omega_n$): $2\pi \times 4.2\text{ rad/s}$ ($k = \omega_n^2 \approx 696.4\text{ s}^{-2}$)
* Damping Ratio ($\zeta$): $0.85$ (slightly underdamped; $c = 2\zeta \omega_n \approx 44.86\text{ s}^{-1}$)
* Max Acceleration Ceiling ($a_{max}$): $8.0 \times g \approx 224.0\text{ m/s}^2$
* **Action-Reaction Principle**: Ground reaction forces are applied symmetrically to dynamic bodies beneath the wheels (e.g., enabling realistic ball dribbling atop the car roof).

#### B. All-Wheel Drive (AWD) & Coulomb Traction
Traction force is applied tangentially along the contact surface $\hat{t} = (-\hat{n}_y, \hat{n}_x)$:
$$F_{tangent} = \operatorname{clamp}(m \cdot a_{drive}, \, -\mu F_N, \, \mu F_N)$$
* **Friction Coefficient ($\mu$)**: $2.2$ (high-grip sticky tires)
* **Normal Load ($F_N$)**: Cumulative load from suspension compression and downward downforce ($a_{sticky} = 16.0\text{ m/s}^2$)
* **Top Ground Speed**: $14.5\text{ m/s}$ ($52.2\text{ km/h}$)
* **Wheelie Speed Limiting**: When only one axle touches the surface and pitch sine $|\hat{f} \cdot \hat{n}| > 0.25$ ($\sim 14.5^\circ$), top speed is capped at $10\%$ ($1.45\text{ m/s}$) to prevent unrealistically flying off rear-wheel wheelies.

#### C. Closed-Loop Attitude Control
In-flight and ground-contour orientation is governed by an acceleration- and rate-limited proportional controller designed to eliminate overshoot and ringing:
$$\omega_{target} = \operatorname{clamp}\left( \operatorname{wrap}(\theta_{target} - \theta) \cdot K_{rate}, \, -\omega_{max}, \, \omega_{max} \right)$$
$$\alpha = \operatorname{clamp}\left( \frac{\omega_{target} - \omega}{\Delta t}, \, -\alpha_{max}, \, \alpha_{max} \right)$$
$$\tau = I_{moment} \cdot \alpha$$
* Steering Rate Gain ($K_{rate}$): $14.0\text{ s}^{-1}$
* Overshoot Criterion: Maintained by constraint $K_{rate} \le \frac{2\alpha_{max}}{\omega_{max}}$
* Air limits: $\omega_{max} = 8.0\text{ rad/s}$, $\alpha_{max} = 70.0\text{ rad/s}^2$
* Spin Decay: When input is neutral, angular velocity decays exponentially: $\omega(t) = \omega_0 e^{-t / \tau_{decay}}$ with $\tau_{decay} = 0.45\text{ s}$.

#### D. Two-Stage Jump & Dodge Mechanics
1. **Jump 1 (Ground / Wheelie Launch)**:
   * Requires contact from either both wheels or drivable wheelie stance ($\text{pitch} < 75^\circ$).
   * Imparts an instant vertical velocity impulse: $\vec{J} = (0.0, \, m \cdot v_{jump})$ with $v_{jump} = 11.76\text{ m/s}$.
   * Impulse direction is **strictly vertical in world coordinates $(0, 1)$**, preventing orientation-induced drift.
   * Suspension unloads completely, resetting ground contact. Jump 2 is armed.
2. **Jump 2 (Airborne / Bumper Upright)**:
   * **Neutral Double Jump** ($|\vec{u}_{input}| \le 0.15$): Imparts fresh launch velocity along the vehicle up-vector $\vec{u}_{car}$ with speed $11.76\text{ m/s}$, achieving an exact $2.0\times$ single-jump ballistic apex.
   * **Directional Dodge Flip** ($|\vec{u}_{input}| > 0.15$): Imparts a $13.0\text{ m/s}$ linear impulse in the commanded input vector direction and activates a forced $360^\circ$ sinusoidal angular velocity sweep over duration $T_{dodge} = 0.42\text{ s}$.
   * Flip direction automatically accounts for horizontal vehicle facing (front flip vs. back flip).
   * Once Jump 2 is spent, further jumps are blocked until drivable surface contact recharges the counter.

#### E. Rocket Boost Dynamics
* Thrust Acceleration ($a_{boost}$): $42.0\text{ m/s}^2$ along vehicle nose $\hat{f}$
* Net Upward Vertical Acceleration: $a_{boost} - g = 42.0 - 28.0 = 14.0\text{ m/s}^2$
* **Aerial Lift Gravity Compensation**: For upward diagonal flight ($45^\circ, 135^\circ$), an anti-gravity component counters $m \cdot g \cdot \hat{f}_y$, providing smooth aerial climb without vertical sag.
* Velocity Fade: Soft fade band over $3.0\text{ m/s}$ up to $v_{max, air} = 25.0\text{ m/s}$.
* Fuel Economy: 100% tank capacity; drains at $30\%/\text{s}$, recharges at $80\%/\text{s}$ while resting on ground without boost pressed.

#### F. Turtle Recovery
When inverted on its roof ($\vec{u}_y < -0.55$) for $t > 0.08\text{ s}$, the car initiates an automatic physical recovery:
1. Applies a vertical hop impulse ($v_{hop} = 4.5\text{ m/s}$).
2. Engages high-authority flip torque ($\omega_{max} = 22\text{ rad/s}, \alpha_{max} = 260\text{ rad/s}^2$) for $0.45\text{ s}$ to snap upright without teleportation.

#### G. Convex Chassis Hull & Vertical Front Bumper Profile
To guarantee authentic vehicle-ball collision mechanics and prevent the car from wedging underneath the ball on kickoff or ground approaches, the chassis is formed by a strictly convex 7-vertex polygon featuring a blunt, vertical front bumper grille:
* **Front Bumper Face**: Spans $(1.05 \cdot s, -0.24 \cdot s)$ to $(1.05 \cdot s, 0.18 \cdot s)$ (where $s = \text{CAR\_SCALE} = 0.75$).
* **Horizontal Normal Alignment**: In world space, this vertical bumper face spans $y \in [1.83, 2.15]\text{ m}$, aligning directly against the equatorial lower hemisphere of the grounded ball ($y_{center} = 2.50\text{ m}, R = 0.9375\text{ m}$).
* **Impact Mechanics**: Ground collisions produce a pure horizontal contact normal $(1.0, 0.0)$, pushing/punching the ball directly forward downfield rather than generating an upward vertical normal that scoops the ball onto the roof.

---

### 3.3. Ball Dynamics (`src/sim/core/ball.py`)

* Radius ($R_{ball}$): $0.9375\text{ m}$
* Mass ($m_{ball}$): $22.5\text{ kg}$ (yielding a stable 6:1 car-to-ball mass ratio)
* Restitution ($e$): $0.80$ against arena walls; $0.60$ against car body
* Aerodynamics: Exponential decay drag models:
  $$\vec{v}(t + \Delta t) = \vec{v}(t) \cdot e^{-c_{drag} \Delta t}, \quad c_{drag} = 0.20\text{ s}^{-1}$$
  $$\omega(t + \Delta t) = \omega(t) \cdot e^{-c_{spin} \Delta t}, \quad c_{spin} = 0.35\text{ s}^{-1}$$
* Speed caps: Hard clamped at $v_{max} = 38.0\text{ m/s}$, $\omega_{max} = 30.0\text{ rad/s}$ to prevent physics tunneling through boundary segments.

---

## 4. Arena Geometry & Elevated Goal Design (`src/sim/core/arena.py`)

The playing field measures $26.0\text{ m} \times 15.0\text{ m}$ bounded by rounded corner fillets with radius $R = 2.5\text{ m}$.

```
Y (meters)
18.0 +------------------------------------------------------------+
     |                     ARENA CEILING                          |
16.5 |      /`--------------------------------------------'\      | (Fillet R=2.5m)
     |     /                                                \     |
11.14|----+ (Upper Wall / Backboard)                        +-----|
     |    |                                                 |     |
 9.64|----+--+ (Top Crossbar)                     (Crossbar)+-+---|
     |GOAL|  |                                              | |GL | Elevated Goal Mouth
     |NET |  |                PLAYABLE FIELD                | |NET| Height = 5.28m
 4.36|----+--+ (Bottom Crossbar)                  (Crossbar)+-+---|
     |    |                                                 |     |
     |    | (Lower Vertical Wall: Bounces rolling balls UP) |     |
 1.5 |     \                                                /     |
     |      \_--------------------------------------------_/      | (Fillet R=2.5m)
 0.0 +------------------------------------------------------------+
     0.0   4.5                                           30.5    35.0   X (meters)
           Left Goal Line                           Right Goal Line
```

### Key Architectural Characteristics:
1. **Elevated Goal Mouth**:
   * Bottom crossbar elevation: $y = 4.36\text{ m}$
   * Top crossbar elevation: $y = 9.64\text{ m}$
   * Vertical mouth opening: $5.28\text{ m}$ ($1.2\times$ scale)
   * Recessed depth: $3.75\text{ m}$ ($1.5\times$ scale) with smooth fillet corners ($R_{corner} = 1.2\text{ m}$).
2. **Vertical Ground Wall Deflection**:
   * Balls rolling across the floor strike the lower vertical wall beneath $y = 4.36\text{ m}$ and bounce vertically upward at $\sim 90^\circ$, creating high-air crossing opportunities directly in front of the net.
3. **100% In-Goal Validation**:
   * A goal is scored if and only if the ball circle is completely past the goal line:
     $$\text{Orange Scores (Left Goal)}: \quad x_{ball} + R_{ball} \le x_{left}$$
     $$\text{Blue Scores (Right Goal)}: \quad x_{ball} - R_{ball} \ge x_{right}$$
   * Prevents premature scoring triggers on boundary grazing.

---

## 5. Hierarchical Zone & Intercept State Machine (HZISM) AI (`src/sim/ai/heuristic_bot.py`)

The automated Orange opponent operates on an asynchronous finite state machine structured around match geometry, ball trajectory prediction, and zone partitioning:

```
                  +--------------------------------------+
                  |               KICKOFF                |
                  |     (Start / Post-Goal Reset)        |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |               ATTACK                 |
                  |     (Drive & Strike toward Net)      |
                  +--------------------------------------+
                         /           |           \
                        /            |            \
                       v             |             v
          +-------------------+      |      +------------------+
          |    ROTATE_BACK    |      |      |      DEFEND      |
          |  (Shadow Defense) |      |      | (Goal Clearance) |
          +-------------------+      |      +------------------+
                                     v
                            +------------------+
                            |      AERIAL      |
                            | (High Ball Save) |
                            +------------------+
```

### State Responsibilities:
* **`KICKOFF`**: Commits full throttle with continuous boost toward midfield, triggering a power dodge flip when closing within $3.2\text{ m}$.
* **`ROTATE_BACK` (Shadow Defense)**: Executes anti-own-goal recovery pathing when caught downfield of the ball ($c_x \le b_x - 0.2$). Positions safe recovery target behind ball toward own post without colliding into ball toward own net.
* **`DEFEND`**: Zone goalie routine inside $x > 22.0\text{ m}$. Intercepts ball using ballistic trajectory lookahead ($\Delta t = 0.1\text{s} - 0.5\text{s}$); initiates vertical aerial saves when incoming shot trajectory exceeds $y = 3.0\text{ m}$.
* **`AERIAL`**: Commands calculated launch angle $\theta = \operatorname{atan2}(\Delta y, \Delta x)$, fires jump impulse, and applies full rocket thrust with a dodge strike upon closing within $2.0\text{ m}$.
* **`ATTACK`**: Direct offensive drive toward opponent goal mouth; initiates power jumps on bouncing balls and uses speed flips on open strikes.

---

## 6. Visualization & Telemetry Interface (`src/sim/visualization/renderer.py`)

The Pygame rendering pipeline runs at 40 pixels per meter ($\text{PPM} = 40.0$), displaying a $1400 \times 720$ canvas.

```
+---------------------------------------------------------------------------------------+
| Sideswipe 2D Physics Sandbox                   0  -  0                  Speed: 38.4 km/h
| [WASD / Arrows] 2D Direction Vector          BLUE - ORANGE              Pitch:  12.3 deg
| [Space] Jump / Flip (or Turtle)                                         Facing:    RIGHT
| [Shift / O] Rocket Boost                    ORANGE BOT: ATTACK          State:  GROUNDED
| [R] Reset | [B] Toggle Bot                                              Jump 2:    READY
|                                                                                       |
| --- Commanded Input (Purple Vector)                                                  |
| --> Car Velocity / Heading (Blue Vector)                                              |
|                                                                                       |
|                                                                                       |
|                                          O (Ball)                                     |
|                                         /                                             |
|                                  ______/                                              |
|              [BLUE CAR]--------->                                                     |
|                                                                                       |
|                                                                                       |
| ====================================================================== [BOOST: 85%] = |
+---------------------------------------------------------------------------------------+
```

### Visualization Elements:
1. **Vector Visualizer**:
   * **Purple Vector**: Real-time representation of commanded directional vector input $(dir_x, dir_y)$.
   * **Blue Vector**: Real-time representation of actual car forward heading and linear velocity magnitude.
2. **Octane-Style Procedural Graphics**:
   * Multi-polygon wedge chassis with front aerodynamic taper and cabin window glass.
   * Dual suspension wheels with rubber tires and alloy rims.
   * Dynamic twin-strut rear spoiler and dual-core rocket exhaust flames active during boost.
   * Dynamic bidirectional facing animation: Car mirrors across its local Y-axis when driving or aiming in reverse.
3. **HUD & Real-Time Telemetry**:
   * Scoreboard tracking goals for Blue and Orange teams.
   * Orange bot active state pill (`KICKOFF`, `ATTACK`, `DEFEND`, `ROTATE_BACK`, `AERIAL`).
   * Numerical readout of car velocity ($\text{km/h}$), pitch angle ($^\circ$), facing orientation, ground contact mode (`GROUNDED`, `WHEELIE`, `UPRIGHT`, `AIRBORNE`), and Jump 2 state (`READY`, `DEPLETED`, `FLIP 360°`).
   * Color-coded boost fuel gauge.

---

## 7. Control Scheme

Input is keyboard only (a connected gamepad is detected and named at startup, but its buttons are not read).

| Action | Keyboard Mapping | Description |
| :--- | :--- | :--- |
| **2D Direction Vector** | `W`/`S`/`A`/`D` or `Arrow Keys` | Cardinal / diagonal orientation command |
| **Jump / Flip** | `Spacebar` | Jump 1 from ground; Jump 2 (Double jump / 360° flip) in air |
| **Rocket Boost** | `Left/Right Shift`, `O`, `J` | High-thrust directional aerial acceleration |
| **Reset Field** | `R` | Resets car, ball, and kickoff positions |
| **Toggle AI Bot** | `B` | Dynamically adds or removes Orange AI opponent |
| **Toggle Boundary Raycasts** | `L` or Click HUD button | Shows/hides 8 egocentric 45° field raycasts |
| **Exit** | `Escape` | Closes simulation window |

---

## 8. Simulation State API (RL & Headless Observation)

Calling `sim.get_state()` exposes complete numerical state representations suitable for custom Gymnasium/Gym wrappers:

```python
{
    "time": 4.35,                          # Round time since last kickoff (seconds)
    "match_time": 18.75,                   # Cumulative match time across kickoffs (seconds)
    "goal_scored_step": "blue",            # Non-null only on the single frame a goal is scored ("blue"/"orange"/None)
    "ball": {
        "position": (13.0, 2.75),          # (x, y) coordinates in meters
        "velocity": (4.2, -1.1),           # (vx, vy) velocity in m/s
        "angle": 0.34,                     # Rotation in radians
        "angular_velocity": 0.0
    },
    "blue_car": {
        "position": (9.2, 1.84),
        "nose_position": (10.2, 1.81),
        "velocity": (12.4, 0.0),
        "angle": 0.02,
        "angular_velocity": 0.0,
        "is_grounded": True,
        "ground_normal": (0.0, 1.0),
        "wheel_contacts": 2,               # Number of wheels in contact (0, 1, 2)
        "both_wheels_grounded": True,
        "has_jump2": True,                 # Jump 2 availability flag
        "is_flipping": False,              # Dodge flip animation status
        "boost": 94.2,                     # Remaining boost percentage (0.0 - 100.0)
        "is_boosting": False,
        "input_vector": (1.0, 0.0),        # Last commanded input vector
        "facing_x": 1,                     # Facing direction (+1: Right, -1: Left)
        "boundary_distances": [20.9, 20.4, 14.4, 10.8, 2.9, 0.64, 0.45, 0.64],  # 8 rays at 45° intervals
        "boundary_raycasts": [             # Complete ray contact details (field only)
            {
                "angle_relative_deg": 0.0,
                "angle_world_rad": 0.02,
                "direction": (1.0, 0.0),
                "distance": 20.9,
                "hit_point": (29.4, 2.01),
                "hit_normal": (-1.0, 0.0)
            },
            ...
        ]
    },
    "relations": {
        "blue_car_to_ball": {"magnitude": 4.12, "direction": (0.92, 0.39)},
        "orange_car_to_ball": {"magnitude": 5.84, "direction": (-0.85, 0.52)},
        "ball_to_orange_goal": {"magnitude": 19.38, "direction": (1.0, 0.0)},
        "ball_to_blue_goal": {"magnitude": 10.38, "direction": (-1.0, 0.0)}
    },
    "score": {"blue": 1, "orange": 0},
    "last_goal": "blue",
    "last_touch": "blue",                  # Team that touched the ball last this kickoff round
    "orange_enabled": True,
    "car_orange": { ... }                  # Full symmetrical telemetry for opponent
}
```

### Observation (`src/sim/core/observation.py`)

`build_observation(sim, ego)` returns the 52 floats the agents see (`env.get_observation(team)` in the
Gymnasium env; `sim.get_state_norm()` / `sim.get_state_norm_mirror()` are thin delegates). Every feature is
normalised:

| Observation | Normalization Method | Bounded Range |
| :--- | :--- | :--- |
| **$(x, y)$ positions** | $((x - x_{center}) / w_{half}, (y - y_{center}) / h_{half})$ around arena center $(17.5, 9.0)$ | $[-1, 1] \times [-1, 1]$ |
| **$(v_x, v_y)$ linear velocity** | $(v_x / v_{max}, v_y / v_{max})$ with $v_{max,car}=28.0, v_{max,ball}=38.0$ | $[-1, 1] \times [-1, 1]$ |
| **Orientation angle** | Continuous circular embedding $(\sin\theta, \cos\theta)$ | $[-1, 1] \times [-1, 1]$ |
| **Angular velocity** | $\omega / \omega_{max}$ with $\omega_{max,car}=25.0, \omega_{max,ball}=30.0$ | $[-1, 1]$ |
| **Relative distances** | $d / \text{ARENA\_DIAGONAL}$ where $\text{ARENA\_DIAGONAL}\approx 39.357\text{ m}$ | $[0, 1]$ |
| **Relative directions** | Normalized 2D unit vector $(d_x, d_y)$ | $[-1, 1] \times [-1, 1]$ |
| **8 Boundary raycasts** | $d / \text{ARENA\_DIAGONAL}$ (field boundary distance only) | $[0, 1]$ |
| **Boost amount** | $\text{boost} / \text{CAR\_MAX\_BOOST}$ | $[0, 1]$ |
| **Wheel contacts** | Axle contact count $/ 2.0$ (replaces binary grounded) | $\{0.0, 0.5, 1.0\}$ |
| **Second-jump ready** | $1.0$ if Jump 2 available else $0.0$ | $\{0.0, 1.0\}$ |
| **Dodge flip active (`is_flipping`)** | $1.0$ if flip maneuver currently active else $0.0$ | $\{0.0, 1.0\}$ |
| **Boost recovery (`is_boost_recovery`)** | $1.0$ if grounded and not boosting else $0.0$ | $\{0.0, 1.0\}$ |
| **Opponent present (`opponent_present`)** | $1.0$ if opponent exists else $0.0$ | $\{0.0, 1.0\}$ |

The fixed layout (named index constants in `observation.py`):
1. **Ego Car (21 floats)**: `[pos_x, pos_y, vel_x, vel_y, sin, cos, ang_vel, boost, wheel_contacts, has_jump2, is_flipping, is_turtling, is_boost_recovery, 8x raycast_distances]`
2. **Ball (5 floats)**: `[pos_x, pos_y, vel_x, vel_y, ang_vel]`
3. **Ego Relations (9 floats)**: ego car to ball, ball to opponent goal, ball to own goal, each `[distance, dir_x, dir_y]`
4. **Opponent Present Flag (1 float)**: `1.0` if the opponent exists else `0.0`
5. **Opponent (16 floats)**: the opponent's 13 car features (as the ego car, with `is_boosting` in place of `is_boost_recovery`, no raycasts) + opponent to ball `[distance, dir_x, dir_y]`; zero-filled when there is no opponent.

Orange's observation is built with Orange as the ego car and then mirrored through the arena's vertical centre
line (`mirror_observation`: x-like features change sign, rays take the order `[0, 7, 6, 5, 4, 3, 2, 1]`), so a
policy trained on Blue sees the same kind of situation when it drives Orange.

```python
from src.sim.core.observation import build_observation
obs_blue = build_observation(sim, "blue")      # 52 floats
obs_orange = build_observation(sim, "orange")  # 52 floats, mirrored
```

---

## 9. Quickstart Commands

### Interactive Simulation
```bash
# Launch interactive mode with Blue player vs. Orange Heuristic Bot
python -m src --play

# Launch single-player practice sandbox (Orange car disabled)
python -m src --play --no-orange
```

### Headless Benchmark
```bash
# Run 300 headless simulation steps with telemetry logging
python -m src --play --headless
```
