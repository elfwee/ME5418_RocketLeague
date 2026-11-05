# Simulation & RL Source Package (`src/`)

This directory contains the complete simulation engine, Gymnasium RL environment wrapper, neural network definitions, learning agent training/evaluation pipelines, AI controller, and rendering pipeline for Rocket League 2D.

---

## 1. Subpackage Structure

```text
src/
├── __init__.py
├── __main__.py                  # Enables `python -m src` (delegates to main.py)
├── main.py                      # Unified CLI entry point (--play, --demo, --train, --eval)
│
├── config/                      # Centralised configuration
│   ├── sim_config.py            # Physics rates, field geometry, car/ball tunings, display settings
│   ├── gym_config.py            # GymConfig: env id, episode/training step limits, action directions
│   └── nn_config.py             # DQNConfig & PPOConfig: matched architectures & RL hyperparameters
│
├── gym_env/                     # Gymnasium RL environment
│   └── rocket_league_env.py     # RocketLeagueEnv (action/observation space, reward, step, reset)
│
├── neural_nets/                 # Neural network architectures & forward/backward pass demos
│   ├── dueling_ddqn_torch.py    # DQN / Double DQN / Dueling Double DQN in PyTorch
│   └── ppo_torch.py             # PPO Actor-Critic network (Policy + Value heads) in PyTorch
│
├── learning_agents/             # Complete RL learning agents (training, evaluation, checkpointing)
│   ├── __init__.py
│   ├── experience_replay.py     # ReplayMemory FIFO circular transition buffer for off-policy DQN
│   ├── dueling_ddqn_agent.py    # DuelingDDQNAgent (--train / --eval for Dueling DDQN, DDQN, DQN)
│   └── ppo_agent.py             # PPOAgent (--train / --eval with RolloutBuffer, GAE, clipped PPO)
│
└── sim/                         # 2D simulation (physics, AI opponent, rendering)
    ├── game.py                  # Interactive & headless game loops (play_interactive, play_headless)
    │
    ├── core/                    # 2D physics engine & entities (Pymunk)
    │   ├── actions.py           # Unified 2D CarAction dataclass
    │   ├── arena.py             # Field walls, elevated goals, and deflection ramps
    │   ├── ball.py              # Rigid-body ball with aerodynamic drag and spin decay
    │   ├── car.py               # Chassis dynamics, raycast suspension, boost, jump-dodge
    │   ├── control.py           # CarControl: none / human (keyboard) / heuristic bot input sources
    │   ├── simulation.py        # Physics manager, contact callbacks, sub-stepping loop
    │   └── README.md            # Deep-dive physics & mathematical formulation
    │
    ├── ai/                      # Scripted adversarial baseline bot
    │   ├── heuristic_bot.py     # Hierarchical Zone & Intercept State Machine (HZISM)
    │   └── README.md            # State machine architecture & decision logic
    │
    └── visualization/           # Graphic visualization & HUD (Pygame)
        ├── renderer.py          # Procedural Octane vehicle, vector debug, telemetry HUD
        └── README.md            # Coordinate mapping, HUD specs, and visual features
```

---

## 2. Learning Agents (`learning_agents/`)

Both learning agents share an identical feature trunk (`LayerNorm`-stabilized 2-layer MLP of width $256$, totaling $150,949$ trainable parameters), `Adam` learning rate ($3 \times 10^{-4}$), discount factor ($\gamma = 0.99$), minibatch size ($64$), and $4\times$ action repeat ($15\text{ Hz}$ decision frequency with edge-triggered jump release):

- **Dueling Double DQN (`learning_agents/dueling_ddqn_agent.py`)**:
  - **Off-Policy Replay**: Stores transitions $(s_t, a_t, r_t, s_{t+1}, \text{terminated}_t)$ in `ReplayMemory` (`experience_replay.py`, capacity $100,000$, warm-up $2,000$ steps) and updates every `train_freq = 4` decisions.
  - **Target Decoupling**: Uses the online network to select the greedy next action $a^* = \arg\max_{a'} Q_\theta(s_{t+1}, a')$ and the periodic target network $Q_{\theta^-}$ (synced every $1,000$ steps) to evaluate it, trained via Huber loss (`SmoothL1Loss`) and global gradient norm clipping ($10.0$).
- **Proximal Policy Optimization (`learning_agents/ppo_agent.py`)**:
  - **On-Policy Rollouts & GAE**: Collects fixed-horizon trajectories ($T = 2048$ macro-steps) in a `RolloutBuffer`, bootstrapping value tails on episode timeouts (`truncated=True`) while zeroing terminal goals (`terminated=True`), and computes Generalized Advantage Estimation ($\text{GAE}(\gamma=0.99, \lambda=0.95)$).
  - **Clipped Surrogate Optimization**: Normalizes advantages per rollout and performs $K = 4$ epochs of shuffled minibatch ($64$) updates using the clipped surrogate objective ($\epsilon = 0.2$), Huber critic loss ($c_1 = 0.5$), and entropy bonus ($c_2 = 0.01$).

---

## 3. Gymnasium Environment API (`gym_env/rocket_league_env.py`)

The environment implements the standard `gymnasium.Env` interface:

```python
import gymnasium as gym
from src.gym_env.rocket_league_env import RocketLeagueEnv

# Instantiate directly or via gym.make
env = RocketLeagueEnv(render_mode="human", bot_type="heuristic_bot")
obs, info = env.reset()
```

### A. Action Space

- `Discrete(36)` (default when `discrete_action=True`) or `MultiDiscrete([9, 2, 2])` (when `discrete_action=False`):
    - Direction (9): `0`: None, `1`: Right, `2`: Left, `3`: Down, `4`: Up, `5`: Down-Right, `6`: Up-Right, `7`: Down-Left, `8`: Up-Left
    - Jump (2): `0`: Off, `1`: Jump / Dodge flip
    - Boost (2): `0`: Off, `1`: Rocket boost
    - In flattened `Discrete(36)` mode, action index $a \in \{0, \dots, 35\}$ decodes as `direction = a // 4`, `jump = (a % 4) // 2`, `boost = a % 2`.

### B. Observation Space (52-Dimensional Flat Vector)

- **Space**: `gym.spaces.Box(low=-1.0, high=1.0, shape=(52,), dtype=np.float32)`
- Returned via `sim.get_state_norm(as_flat_array=True)`:

| Index Range | Category                       | Dimension | Normalized Features                                                                                                                                    |
| :---------- | :----------------------------- | :-------: | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[0:2]`     | Ego Car Position               |     2     | $(x - x_c)/w_h, (y - y_c)/h_h$                                                                                                                         |
| `[2:4]`     | Ego Car Velocity               |     2     | $v_x / 28.0, v_y / 28.0$                                                                                                                               |
| `[4:6]`     | Ego Car Orientation            |     2     | Circular embedding $(\sin\theta, \cos\theta)$                                                                                                          |
| `[6:7]`     | Ego Car Angular Velocity       |     1     | $\omega / 25.0$                                                                                                                                        |
| `[7:8]`     | Ego Boost Level                |     1     | $\text{boost} / 100.0 \in [0, 1]$                                                                                                                      |
| `[8:9]`     | Wheel Contacts                 |     1     | $\text{contact\_count} / 2.0 \in \{0.0, 0.5, 1.0\}$                                                                                                    |
| `[9:10]`    | Jump 2 Available               |     1     | $1.0$ if ready else $0.0$                                                                                                                              |
| `[10:11]`   | Dodge Flip Active              |     1     | $1.0$ if flipping else $0.0$                                                                                                                           |
| `[11:12]`   | Turtling Active                |     1     | $1.0$ if roof-inverted else $0.0$                                                                                                                      |
| `[12:13]`   | Boost Recovery Active          |     1     | $1.0$ if grounded & not boosting else $0.0$                                                                                                            |
| `[13:21]`   | 8 Radial Boundary Raycasts     |     8     | Normalized distance to arena walls at $45^\circ$ intervals                                                                                             |
| `[21:23]`   | Ball Position                  |     2     | $(x - x_c)/w_h, (y - y_c)/h_h$                                                                                                                         |
| `[23:25]`   | Ball Velocity                  |     2     | $v_x / 38.0, v_y / 38.0$                                                                                                                               |
| `[25:26]`   | Ball Angular Velocity          |     1     | $\omega / 30.0$                                                                                                                                        |
| `[26:29]`   | Ego-to-Ball Relation           |     3     | Normalized magnitude $[0, 1]$ and 2D unit direction $(\hat{d}_x, \hat{d}_y)$                                                                           |
| `[29:32]`   | Ball-to-Opponent Goal Relation |     3     | Normalized magnitude $[0, 1]$ and 2D unit direction $(\hat{d}_x, \hat{d}_y)$                                                                           |
| `[32:35]`   | Ball-to-Own Goal Relation      |     3     | Normalized magnitude $[0, 1]$ and 2D unit direction $(\hat{d}_x, \hat{d}_y)$                                                                           |
| `[35:36]`   | Opponent Present Flag          |     1     | $1.0$ if opponent enabled else $0.0$                                                                                                                   |
| `[36:49]`   | Opponent Car State             |    13     | Position (2), velocity (2), orientation (2), angular velocity (1), boost (1), wheel contacts (1), jump 2 (1), flipping (1), turtling (1), boosting (1) |
| `[49:52]`   | Opponent-to-Ball Relation      |     3     | Normalized magnitude $[0, 1]$ and 2D unit direction (or zeros)                                                                                         |

### C. Reward Function Formulation

- **Goal Scoring**:
    - Blue scores in Orange goal: $+20.0$ (or $+2.0$ if Orange conceded an own-goal)
    - Orange scores in Blue goal: $-20.0$
    - Terminates the episode upon goal event.
- **Dense Intermediate Shaping**:
    - **Ball Progress**: Positively rewards ball displacement toward the opponent's goal scaled by position proximity factors (retaining a $0.25$ baseline when ground-rolling at $y_{\text{ball}} \le -0.55$):
      $$r_{\text{progress}} = \Delta x_{\text{ball}} \cdot e^{\frac{1}{1.2 - x_{\text{ball}}^2}} \cdot y_{\text{factor}}, \quad y_{\text{factor}} = \begin{cases} (1 - |y_{\text{ball}}|)^2 & \text{if } y_{\text{ball}} > -0.55 \\ 0.25 & \text{otherwise} \end{cases}$$
    - **Potential-Based Ball Approach & Distance Shaping**: $+1.5 \times (d_{\text{prev}} - d_{\text{curr}}) - 0.005 \times d_{\text{curr}}$, where $d \in [0, 1]$ is the normalized car-to-ball distance
    - **Time Step Penalty**: $-0.001$ per step

---

## 4. Control Scheme

| Action                  | Keyboard Mapping                | Gamepad       | Description                                         |
| :---------------------- | :------------------------------ | :------------ | :-------------------------------------------------- |
| **2D Direction Vector** | `W`/`S`/`A`/`D` or `Arrow Keys` | Left Stick    | Cardinal / analog steering input                    |
| **Jump / Flip**         | `Spacebar`                      | Button A      | Jump 1 from ground; Jump 2 (neutral or flip) in air |
| **Rocket Boost**        | `Shift`, `O`, `J`               | Button B / R2 | Aerial propulsion thrust                            |
| **Reset Field**         | `R`                             | —             | Resets car and ball to kickoff positions            |
| **Toggle AI Bot**       | `B`                             | —             | Adds or removes the Orange bot                      |
| **Toggle Raycasts**     | `L`                             | —             | Shows/hides 8 radial boundary raycasts              |
| **Quit**                | `Escape`                        | —             | Exits simulation                                    |

---

## 5. Module Deep Dives

For exhaustive mathematical equations and subsystem specifications, see:

- [Core Physics Engine (`src/sim/core/README.md`)](sim/core/README.md)
- [Opponent AI Controller (`src/sim/ai/README.md`)](sim/ai/README.md)
- [Pygame Rendering Engine (`src/sim/visualization/README.md`)](sim/visualization/README.md)
