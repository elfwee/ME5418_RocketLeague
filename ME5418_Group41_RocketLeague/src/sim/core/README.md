# Core Simulation & Physics Engine (`src/sim/core/`)

This directory contains the rigid-body dynamical simulation, vehicle kinematics, ball ballistics, arena collision boundaries, and action definitions for Rocket League 2D.

---

## 1. Directory File Overview

| File | Description |
| :--- | :--- |
| [`actions.py`](actions.py) | Defines the unified `CarAction` dataclass (directional vector input `dir_x, dir_y`, binary `jump`, binary `boost`). |
| [`arena.py`](arena.py) | Enclosed boundary geometry, corner fillets, and elevated goal mouths with vertical deflection walls. |
| [`ball.py`](ball.py) | Rigid-body ball with restitution, mass scaling, aerodynamic drag, spin decay, and velocity capping. |
| [`car.py`](car.py) | Vehicle chassis dynamics, independent raycast suspension, AWD Coulomb traction, attitude control, two-stage jump-dodge ballistics, rocket boost, and turtle recovery. |
| [`control.py`](control.py) | `CarControl` dispatcher supporting `"none"`, `"human"`, `"random"`, and `"heuristic_bot"` control modes. |
| [`simulation.py`](simulation.py) | Pymunk space coordinator, $240\text{ Hz}$ sub-stepping loop, collision arbiters, goal detection, and state dictionary generation. |

---

## 2. Temporal Discretization & Solver Settings

The physics engine is powered by [Pymunk](http://www.pymunk.org/) (Chipmunk2D bindings) configured with deterministic numerical settings:

* **Base Frame Rate ($f_{sim}$)**: $60\text{ Hz}$ ($\Delta t_{frame} = 16.67\text{ ms}$)
* **Sub-step Multiplier ($N_{sub}$)**: 4 sub-steps per frame
* **Effective Physics Rate**: $240\text{ Hz}$ ($\Delta t_{sub} = \frac{1}{240}\text{ s} \approx 4.167\text{ ms}$)
* **Solver Iterations**: 14 constraint iterations per sub-step
* **Global Damping**: 1.0 (zero global damping; aerodynamic drag is applied per entity)
* **World Gravity ($\vec{g}$)**: $(0.0, -28.0)\text{ m/s}^2$ (downward arcade-tuned gravity for responsive aerial play)

---

## 3. Vehicle Dynamics (`car.py`)

The car is modeled as a rigid chassis supported by independent raycast suspension and ground traction rather than a simplified sliding box.

### A. Raycast Suspension System
Two independent downward rays are cast along the car's negative local up-axis ($-\vec{u}_{car}$) at the rear axle ($x = -0.4875\text{ m}$) and front axle ($x = 0.4125\text{ m}$):
$$\vec{x}_{start} = \vec{x}_{axle} + \vec{u}_{car} \cdot d_{offset}, \quad \vec{x}_{end} = \vec{x}_{axle} - \vec{u}_{car} \cdot L_{ray}$$

Each wheel acts as a damped spring along the contact surface normal $\hat{n}$:
$$F_{susp} = \frac{m}{2} \cdot \operatorname{clamp}\left( k \cdot \Delta x - c \cdot (\vec{v}_{axle} \cdot \hat{n}), \, 0, \, a_{max} \right)$$

* **Natural Frequency ($\omega_n$)**: $2\pi \times 4.2\text{ rad/s}$ ($k = \omega_n^2 \approx 696.4\text{ s}^{-2}$)
* **Damping Ratio ($\zeta$)**: $0.85$ (slightly underdamped; $c = 2\zeta \omega_n \approx 44.86\text{ s}^{-1}$)
* **Max Acceleration Ceiling ($a_{max}$)**: $8.0 \times g \approx 224.0\text{ m/s}^2$
* **Action-Reaction Principle**: Ground reaction forces are applied symmetrically to dynamic bodies beneath the wheels, allowing authentic ball dribbling atop the vehicle roof.

### B. All-Wheel Drive (AWD) & Coulomb Traction
Traction force is applied tangentially along the contact surface $\hat{t} = (-\hat{n}_y, \hat{n}_x)$:
$$F_{tangent} = \operatorname{clamp}(m \cdot a_{drive}, \, -\mu F_N, \, \mu F_N)$$

* **Friction Coefficient ($\mu$)**: $2.2$ (high-grip sticky tires)
* **Normal Load ($F_N$)**: Cumulative load from suspension compression and downward downforce ($a_{sticky} = 16.0\text{ m/s}^2$)
* **Top Ground Speed**: $14.5\text{ m/s}$ ($52.2\text{ km/h}$)
* **Wheelie Speed Limiting**: When only one axle touches the surface and pitch sine $|\hat{f} \cdot \hat{n}| > 0.25$ ($\sim 14.5^\circ$), top speed is capped at $10\%$ ($1.45\text{ m/s}$) to prevent unrealistically flying off rear-wheel wheelies.

### C. Closed-Loop Attitude Control
In-flight and ground-contour orientation is governed by an acceleration- and rate-limited proportional controller designed to eliminate overshoot and ringing:
$$\omega_{target} = \operatorname{clamp}\left( \operatorname{wrap}(\theta_{target} - \theta) \cdot K_{rate}, \, -\omega_{max}, \, \omega_{max} \right)$$
$$\alpha = \operatorname{clamp}\left( \frac{\omega_{target} - \omega}{\Delta t}, \, -\alpha_{max}, \, \alpha_{max} \right)$$
$$\tau = I_{moment} \cdot \alpha$$

* **Steering Rate Gain ($K_{rate}$)**: $14.0\text{ s}^{-1}$
* **Overshoot Criterion**: Maintained by constraint $K_{rate} \le \frac{2\alpha_{max}}{\omega_{max}}$
* **Air limits**: $\omega_{max} = 8.0\text{ rad/s}$, $\alpha_{max} = 70.0\text{ rad/s}^2$
* **Spin Decay**: When directional input is neutral, angular velocity decays exponentially: $\omega(t) = \omega_0 e^{-t / \tau_{decay}}$ with $\tau_{decay} = 0.45\text{ s}$.

### D. Two-Stage Jump & Dodge Mechanics
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

### E. Rocket Boost Dynamics
* **Thrust Acceleration ($a_{boost}$)**: $42.0\text{ m/s}^2$ along vehicle nose $\hat{f}$
* **Net Upward Vertical Acceleration**: $a_{boost} - g = 42.0 - 28.0 = 14.0\text{ m/s}^2$
* **Aerial Lift Gravity Compensation**: For upward diagonal flight ($45^\circ, 135^\circ$), an anti-gravity component counters $m \cdot g \cdot \hat{f}_y$, providing smooth aerial climb without vertical sag.
* **Velocity Fade**: Soft fade band over $3.0\text{ m/s}$ up to $v_{max, air} = 25.0\text{ m/s}$.
* **Fuel Economy**: 100% tank capacity; drains at $30\%/\text{s}$, recharges at $80\%/\text{s}$ while resting on the ground without boost pressed.

### F. Turtle Recovery
When inverted on its roof ($\vec{u}_y < -0.55$) for $t > 0.08\text{ s}$, the car initiates an automatic physical recovery:
1. Applies a vertical hop impulse ($v_{hop} = 4.5\text{ m/s}$).
2. Engages high-authority flip torque ($\omega_{max} = 22\text{ rad/s}, \alpha_{max} = 260\text{ rad/s}^2$) for $0.45\text{ s}$ to snap upright without teleportation.

### G. Convex Chassis Hull & Vertical Front Bumper Profile
To prevent the car from wedging underneath the ball on kickoff or ground approaches, the chassis is formed by a strictly convex 7-vertex polygon featuring a blunt, vertical front bumper grille:
* **Front Bumper Face**: Spans $(1.05 \cdot s, -0.24 \cdot s)$ to $(1.05 \cdot s, 0.18 \cdot s)$ (where $s = \text{CAR\_SCALE} = 0.75$).
* **World Alignment**: In world space, this vertical bumper face spans $y \in [1.83, 2.15]\text{ m}$, aligning directly against the equatorial lower hemisphere of the grounded ball ($y_{center} = 2.50\text{ m}, R = 0.9375\text{ m}$).
* **Impact Mechanics**: Ground collisions produce a pure horizontal contact normal $(1.0, 0.0)$, punching the ball directly forward downfield rather than scooping it onto the roof.

---

## 4. Ball Dynamics (`ball.py`)

* **Radius ($R_{ball}$)**: $0.9375\text{ m}$
* **Mass ($m_{ball}$)**: $22.5\text{ kg}$ (yielding a stable 6:1 car-to-ball mass ratio; car mass = $135\text{ kg}$)
* **Restitution ($e$)**: $0.80$ against arena walls; $0.60$ against car body
* **Aerodynamics**: Exponential decay drag models:
  $$\vec{v}(t + \Delta t) = \vec{v}(t) \cdot e^{-c_{drag} \Delta t}, \quad c_{drag} = 0.20\text{ s}^{-1}$$
  $$\omega(t + \Delta t) = \omega(t) \cdot e^{-c_{spin} \Delta t}, \quad c_{spin} = 0.35\text{ s}^{-1}$$
* **Speed Clamping**: Clamped at $v_{max} = 38.0\text{ m/s}$, $\omega_{max} = 30.0\text{ rad/s}$ to guarantee zero tunneling through boundary segments.

---

## 5. Arena Geometry & Elevated Goal Design (`arena.py`)

The playing field measures $26.0\text{ m} \times 15.0\text{ m}$ bounded by rounded corner fillets with radius $R = 2.5\text{ m}$.

```text
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
