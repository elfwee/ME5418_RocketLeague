# ME5418_RocketLeague

> Team 41:
> Lu Jia Yih Jensen, Benjamin Teh Jhen Hing, Wee Fook Choon

---

A high-fidelity 2D vertical-plane robotic simulation platform modeling vehicle-ball interaction, aerial maneuvering, and reinforcement learning environments inspired by **Rocket League Sideswipe**.

## ![Demo](./demo.gif)

---

## 1. Project Overview

- **Physics Engine**: Deterministic $240\text{ Hz}$ physics solver ($60\text{ Hz} \times 4\text{ sub-steps}$) powered by Pymunk (Chipmunk2D).
- **Vehicle Dynamics**: Raycast spring-damper suspension, Coulomb tire traction, 2D vector steering, two-stage jump-dodge ballistics, rocket boost aerodynamics, and turtle auto-recovery.
- **Opponent AI**: Rule-based Hierarchical Zone & Intercept State Machine (HZISM) acting as an adversarial baseline.
- **RL Interface**: Standard Gymnasium environment (`rocket-league-v1`) with continuous/box observation space (52 normalized features) and factored action spaces.

## 2. Architecture & Modular Documentation

Detailed technical specifications and mathematical formulations are organized modularly within each subsystem directory:

- 📚 **[System & RL Environment (`src/README.md`)](./ME5418_Group41_RocketLeague/src/README.md)**: Gymnasium observation/action spaces, reward shaping formulation, telemetry API, and control scheme.
- ⚙️ **[Physics & Vehicle Kinematics (`src/core/README.md`)](./ME5418_Group41_RocketLeague/src/core/README.md)**: Suspension equations, Coulomb traction, attitude control, jump ballistics, ball restitution/drag, and arena geometry.
- 🤖 **[Opponent AI Architecture (`src/ai/README.md`)](./ME5418_Group41_RocketLeague/src/ai/README.md)**: Finite state machine structure, kickoff tactics, shadow defense, zone goalie interception, and aerial saves.
- 🎨 **[Visualization & HUD Telemetry (`src/visualization/README.md`)](./ME5418_Group41_RocketLeague/src/visualization/README.md)**: Pygame rendering pipeline, procedural Octane chassis, vector debug visualizers, and HUD layout.

---

## 3. Quickstart & Usage

### Step 1: Cloning and working directory

```bash
git clone https://github.com/elfwee/ME5418_RocketLeague.git
cd ./ME5418_RocketLeague/ME5418_Group41_RocketLeague
```

### Step 2: Environment Setup

Create and activate the Conda environment using the provided configuration:

```bash
conda env create -f environment.yml
conda activate me5418_projenv
```

_(Alternatively, install the core dependencies via pip: `pip install gymnasium pygame pymunk numpy`)_

### Step 3: Run Gym Environment Demo

Launch sample episodes running under random actions with graphic visualization:

```bash
python demo.py
```

### Step 4: Run Interactive Sandbox

Drive the car manually via keyboard or gamepad against the AI bot:

```bash
python src/main.py
```

- **Controls**:
    - `W`/`A`/`S`/`D` or `Arrow Keys` (or Gamepad Left Stick): 2D Direction Vector
    - `Space` (or Gamepad A): Jump 1 (Ground) / Jump 2 (Neutral double jump or 360° dodge flip) / Turtle recovery
    - `Left Shift` / `O` (or Gamepad B): Rocket Boost
    - `B`: Toggle Orange AI Bot on/off
    - `L`: Toggle 8 egocentric boundary raycasts
    - `R`: Reset Ball & Car to kickoff positions
    - `Escape`: Quit simulation

### Step 5: Run Headless Benchmark

Run automated headless stepping for fast physics verification:

```bash
python src/main.py --headless
```

---

## AI Tool Declaration

We used Google Gemini to write and debug code. We are responsible for the content and quality of the submitted work.
