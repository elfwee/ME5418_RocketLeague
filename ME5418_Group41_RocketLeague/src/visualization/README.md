# Visualization & Telemetry Interface (`src/visualization/`)

This directory contains the 2D rendering pipeline, procedural Octane vehicle graphics, vector visualizers, and real-time HUD telemetry powered by Pygame.

---

## 1. Directory File Overview

| File | Description |
| :--- | :--- |
| [`renderer.py`](renderer.py) | Full Pygame rendering engine, coordinate mapping (meters to pixels), procedural vehicle drawing, HUD overlay, vector visualizers, and debug raycast rendering. |

---

## 2. Rendering Specifications & Coordinate Mapping

* **Canvas Resolution**: $1400 \times 720\text{ pixels}$
* **Pixel-to-Meter Ratio**: $\text{PPM} = 40.0\text{ px/m}$ (e.g., $35.0\text{ m}$ field width = $1400\text{ px}$, $18.0\text{ m}$ height = $720\text{ px}$)
* **Coordinate Transformation**:
  * Physics Space (Pymunk): Origin $(0, 0)$ at bottom-left corner; $+X$ is right, $+Y$ is **up**.
  * Screen Space (Pygame): Origin $(0, 0)$ at top-left corner; $+X$ is right, $+Y$ is **down**.
  * Transform:
    $$x_{screen} = x_{world} \cdot \text{PPM}$$
    $$y_{screen} = \text{SCREEN\_HEIGHT} - (y_{world} \cdot \text{PPM})$$

---

## 3. Visual Layout & HUD Architecture

```text
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

---

## 4. Visual Features & Telemetry Elements

### A. Vector Visualizers
* **Purple Vector**: Real-time representation of commanded directional vector input $(dir_x, dir_y)$.
* **Blue Vector**: Real-time representation of the car's actual velocity vector and heading magnitude.

### B. Octane-Style Procedural Graphics
* **Chassis**: Multi-polygon wedge body with aerodynamic front taper and tinted cabin window.
* **Suspension & Wheels**: Animated wheel assemblies with rubber tread and alloy rims that compress and extend with suspension travel.
* **Aero & Exhaust**: Dynamic twin-strut rear spoiler; dual-core particle exhaust flame rendered during rocket boost.
* **Bidirectional Mirroring**: Smoothly flips vehicle graphics across its local Y-axis when driving or aiming in reverse.

### C. HUD & Real-Time Telemetry
* **Scoreboard**: Tracks goals for Blue and Orange teams with goal flash alerts.
* **AI Bot State Pill**: Displays the current active state of the Orange bot (`KICKOFF`, `ATTACK`, `DEFEND`, `ROTATE_BACK`, `AERIAL`).
* **Telemetry Readouts**:
  * Car velocity ($\text{km/h}$)
  * Pitch angle ($^\circ$)
  * Facing orientation (`RIGHT` / `LEFT`)
  * Ground contact mode (`GROUNDED`, `WHEELIE`, `UPRIGHT`, `AIRBORNE`)
  * Jump 2 status (`READY`, `DEPLETED`, `FLIP 360°`)
* **Boost Gauge**: Color-coded fuel bar with percentage readout.

### D. Egocentric Lidar Raycasts
* The renderer can visualize 8 radial boundary raycasts separated at $45^\circ$ intervals around the vehicle, showing ray trajectories, contact points, and surface normals.
* **Toggle**: Can be toggled on/off at runtime via the `L` key or by clicking the HUD button.
