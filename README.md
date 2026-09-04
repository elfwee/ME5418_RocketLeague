# ME5418_RocketLeague

A multi-version robotic control and simulation platform modeling autonomous vehicle-ball interaction, aerial maneuvering, and reinforcement learning environments inspired by **Rocket League**.

---

## Repository Structure

* **[`RocketLeagueV2/`](RocketLeagueV2/) (Current Version)**:
  * High-fidelity 2D vertical-plane physics simulation styled after *Rocket League Sideswipe*.
  * Features raycast spring-damper suspension, Coulomb tire traction, 2D vector steering, two-stage jump-dodge ballistics, rocket boost aerodynamics, and a Hierarchical Zone & Intercept State Machine (HZISM) AI opponent.
  * Deterministic $240\text{ Hz}$ physics solver ($60\text{ Hz} \times 4\text{ sub-steps}$) powered by Pymunk and visualized via Pygame.
  * **Comprehensive Technical Specification**: See [RocketLeagueV2/TECHNICAL_SUMMARY.md](RocketLeagueV2/TECHNICAL_SUMMARY.md).

* **[`RocketLeagueV1/`](RocketLeagueV1/) (Legacy Version)**:
  * Planar 2D top-down simulation of a circular robot agent pushing balls into border goals.
  * Designed for fundamental search-and-push learning algorithms.

---

## Quickstart (RocketLeagueV2)

### 1. Requirements
Ensure Python 3.8+ with Pygame and Pymunk installed:
```bash
pip install pygame pymunk
```

### 2. Run Interactive Simulation
```bash
cd RocketLeagueV2
python3 src/main.py
```
* **Controls**:
  * `W`/`A`/`S`/`D` or `Arrow Keys` (or Gamepad Left Stick): 2D Direction Vector
  * `Space` (or Gamepad A): Jump 1 / Jump 2 (neutral double jump or 360° dodge flip) / Turtle recovery
  * `Left Shift` / `O` (or Gamepad B): Rocket Boost
  * `B`: Toggle Orange AI Bot
  * `R`: Reset Ball & Car

### 3. Run Headless Benchmark
```bash
cd RocketLeagueV2
python3 src/main.py --headless
```

### 4. Run Test Suite
```bash
cd RocketLeagueV2
python3 -m unittest discover tests -v
```
