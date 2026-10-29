# Simulation Source Package (`src/`)

This directory contains the complete simulation engine, Gymnasium RL environment wrapper, AI controller, and rendering pipeline for Rocket League 2D.

---

## 1. Subpackage Structure

```text
src/
├── __init__.py
├── __main__.py               # Enables `python -m src` (delegates to main.py)
├── main.py                   # Entry point: interactive sandbox, headless benchmark, gym demo
│
├── config/                   # Centralised configuration
│   ├── sim_config.py         # Physics rates, field geometry, car/ball tunings, display settings
│   ├── gym_config.py         # GymConfig: env id, episode/training step limits, action directions
│   └── nn_config.py          # DQNConfig: network sizes, learning rate, replay buffer, batch size
│
├── gym_env/                  # Gymnasium RL environment
│   └── rocket_league_env.py  # RocketLeagueEnv (action/observation space, reward, step, reset)
│
├── neural_nets/              # Deep Q-Network agents
│   ├── dqn_tf.py             # DQN implementation in TensorFlow/Keras
│   └── dqn_torch.py          # Dueling DQN implementation in PyTorch
│
└── sim/                      # 2D simulation (physics, AI opponent, rendering)
    ├── game.py               # Interactive & headless game loops (play_interactive, play_headless)
    │
    ├── core/                 # 2D physics engine & entities (Pymunk)
    │   ├── actions.py        # Unified 2D CarAction dataclass
    │   ├── arena.py          # Field walls, elevated goals, and deflection ramps
    │   ├── ball.py           # Rigid-body ball with aerodynamic drag and spin decay
    │   ├── car.py            # Chassis dynamics, raycast suspension, boost, jump-dodge
    │   ├── control.py        # CarControl: none / human (keyboard) / heuristic bot input sources
    │   ├── simulation.py     # Physics manager, contact callbacks, sub-stepping loop
    │   └── README.md         # Deep-dive physics & mathematical formulation
    │
    ├── ai/                   # Scripted adversarial baseline bot
    │   ├── heuristic_bot.py  # Hierarchical Zone & Intercept State Machine (HZISM)
    │   └── README.md         # State machine architecture & decision logic
    │
    └── visualization/        # Graphic visualization & HUD (Pygame)
        ├── renderer.py       # Procedural Octane vehicle, vector debug, telemetry HUD
        └── README.md         # Coordinate mapping, HUD specs, and visual features
```

---

## 2. Gymnasium Environment API (`gym_env/rocket_league_env.py`)

The environment implements the standard `gymnasium.Env` interface:

```python
import gymnasium as gym
from src.gym_env.rocket_league_env import RocketLeagueEnv

# Instantiate directly or via gym.make
env = RocketLeagueEnv(render_mode="human", bot_type="heuristic_bot")
obs, info = env.reset()
```

### A. Action Space
* `MultiDiscrete([9, 2, 2])`:
  * Direction (9): `0`: None, `1`: Right, `2`: Left, `3`: Down, `4`: Up, `5`: Down-Right, `6`: Up-Right, `7`: Down-Left, `8`: Up-Left
  * Jump (2): `0`: Off, `1`: Jump / Dodge flip
  * Boost (2): `0`: Off, `1`: Rocket boost

### B. Observation Space (52-Dimensional Flat Vector)
* **Space**: `gym.spaces.Box(low=-1.0, high=1.0, shape=(52,), dtype=np.float32)`
* Returned via `sim.get_state_norm(as_flat_array=True)`:

| Index Range | Category | Dimension | Normalized Features |
| :--- | :--- | :---: | :--- |
| `[0:2]` | Ego Car Position | 2 | $(x - x_c)/w_h, (y - y_c)/h_h$ |
| `[2:4]` | Ego Car Velocity | 2 | $v_x / 28.0, v_y / 28.0$ |
| `[4:6]` | Ego Car Orientation | 2 | Circular embedding $(\sin\theta, \cos\theta)$ |
| `[6:7]` | Ego Car Angular Velocity | 1 | $\omega / 25.0$ |
| `[7:8]` | Ego Boost Level | 1 | $\text{boost} / 100.0 \in [0, 1]$ |
| `[8:9]` | Wheel Contacts | 1 | $\text{contact\_count} / 2.0 \in \{0.0, 0.5, 1.0\}$ |
| `[9:10]` | Jump 2 Available | 1 | $1.0$ if ready else $0.0$ |
| `[10:11]` | Dodge Flip Active | 1 | $1.0$ if flipping else $0.0$ |
| `[11:12]` | Turtling Active | 1 | $1.0$ if roof-inverted else $0.0$ |
| `[12:13]` | Boost Recovery Active | 1 | $1.0$ if grounded & not boosting else $0.0$ |
| `[13:21]` | 8 Radial Boundary Raycasts | 8 | Normalized distance to arena walls at $45^\circ$ intervals |
| `[21:23]` | Ball Position | 2 | $(x - x_c)/w_h, (y - y_c)/h_h$ |
| `[23:25]` | Ball Velocity | 2 | $v_x / 38.0, v_y / 38.0$ |
| `[25:26]` | Ball Angular Velocity | 1 | $\omega / 30.0$ |
| `[26:29]` | Ego-to-Ball Relation | 3 | Normalized magnitude $[0, 1]$ and 2D unit direction $(\hat{d}_x, \hat{d}_y)$ |
| `[29:32]` | Ball-to-Opponent Goal Relation | 3 | Normalized magnitude $[0, 1]$ and 2D unit direction $(\hat{d}_x, \hat{d}_y)$ |
| `[32:35]` | Ball-to-Own Goal Relation | 3 | Normalized magnitude $[0, 1]$ and 2D unit direction $(\hat{d}_x, \hat{d}_y)$ |
| `[35:36]` | Opponent Present Flag | 1 | $1.0$ if opponent enabled else $0.0$ |
| `[36:49]` | Opponent Car State | 13 | Position (2), velocity (2), orientation (2), angular velocity (1), boost (1), wheel contacts (1), jump 2 (1), flipping (1), turtling (1), boosting (1) |
| `[49:52]` | Opponent-to-Ball Relation | 3 | Normalized magnitude $[0, 1]$ and 2D unit direction (or zeros) |

### C. Reward Function Formulation
* **Goal Scoring**:
  * Blue scores in Orange goal: $+20.0$ (or $+2.0$ if Orange conceded an own-goal)
  * Orange scores in Blue goal: $-20.0$
  * Terminates the episode upon goal event.
* **Dense Intermediate Shaping**:
  * **Ball Progress**: Positively rewards ball displacement toward the opponent's goal scaled by position proximity factors:
    $$r_{progress} = \Delta x_{ball} \cdot e^{\frac{1}{1.2 - x_{ball}^2}} \cdot (1 - |y_{ball}|)^2 \quad (\text{for } y_{ball} > -0.55, \text{ else } 0)$$
  * **Ball Distance Penalty**: $-0.01 \times d_{\text{norm}}(\text{car}, \text{ball})$ where $d_{\text{norm}} \in [0, 1]$ is the normalized relative distance
  * **Time Step Penalty**: $-0.001$ per step
  * **Control Churn Penalty**: $-0.001$ on abrupt changes in commanded direction vector

---

## 3. Control Scheme

| Action | Keyboard Mapping | Gamepad | Description |
| :--- | :--- | :--- | :--- |
| **2D Direction Vector** | `W`/`S`/`A`/`D` or `Arrow Keys` | Left Stick | Cardinal / analog steering input |
| **Jump / Flip** | `Spacebar` | Button A | Jump 1 from ground; Jump 2 (neutral or flip) in air |
| **Rocket Boost** | `Shift`, `O`, `J` | Button B / R2 | Aerial propulsion thrust |
| **Reset Field** | `R` | — | Resets car and ball to kickoff positions |
| **Toggle AI Bot** | `B` | — | Adds or removes the Orange bot |
| **Toggle Raycasts** | `L` | — | Shows/hides 8 radial boundary raycasts |
| **Quit** | `Escape` | — | Exits simulation |

---

## 4. Module Deep Dives

For exhaustive mathematical equations and subsystem specifications, see:
* [Core Physics Engine (`src/sim/core/README.md`)](sim/core/README.md)
* [Opponent AI Controller (`src/sim/ai/README.md`)](sim/ai/README.md)
* [Pygame Rendering Engine (`src/sim/visualization/README.md`)](sim/visualization/README.md)
