# Simulation & RL Source Package (`src/`)

This directory contains the complete simulation engine, Gymnasium RL environment wrapper, neural network definitions, learning agent training/evaluation pipelines, AI controller, and rendering pipeline for Rocket League 2D.

---

## 1. Subpackage Structure

```text
src/
├── __main__.py                  # Enables `python -m src`
├── cli.py                       # Command-line interface (--play, --head_to_head, --demo, --train, --curriculum, --eval, --plot)
├── main.py                      # Compatibility shim re-exporting cli.main
├── paths.py                     # ROOT_DIR and RUNS_DIR (the only definitions)
│
├── config/                      # Centralised configuration
│   ├── sim_config.py            # Physics rates, field geometry, car/ball tunings, spawn ranges
│   ├── render_config.py         # Screen size, pixels per metre, colour palette
│   ├── gym_config.py            # GymConfig: env id, episode length, action directions, reward constants
│   ├── nn_config.py             # DQNConfig & PPOConfig: matched architectures & RL hyperparameters
│   └── curriculum_learning.py   # CurriculumConfig: stage order, self-play rounds, plateau patience
│
├── gym_env/                     # Gymnasium RL environment
│   ├── rocket_league_env.py     # RocketLeagueEnv (spaces, step, reset, rendering); one step = one sim frame
│   ├── action_codec.py          # The 9 x 2 x 2 action encoding: encode / decode / clear_jump / mirror
│   ├── rewards.py               # Goal, ball-progress and approach-shaping reward terms
│   ├── wrappers.py              # ActionRepeat: hold each decision for N frames, release jump, sum reward
│   └── demo.py                  # --demo --gym: env checker + one random episode
│
├── neural_nets/                 # PyTorch networks (import only torch)
│   ├── dueling_ddqn_torch.py    # DQN / Double DQN / Dueling Double DQN
│   ├── ppo_torch.py             # PPO Actor-Critic (flat or factored policy heads + value head)
│   ├── checkpoint.py            # load_policy(path): rebuilds any saved network from its state_dict
│   ├── policy.py                # greedy_action(net, obs) for every network type
│   └── demos.py                 # --demo --dqn/--double_dqn/--dueling_ddqn/--ppo forward/backward passes
│
├── learning_agents/             # RL learning agents (training, evaluation, checkpointing)
│   ├── dueling_ddqn_agent.py    # DuelingDDQNAgent: DQN learning rule, train() / evaluate()
│   ├── ppo_agent.py             # PPOAgent: GAE + clipped surrogate, train() / evaluate()
│   ├── trainer.py               # run_training: the episode loop shared by both agents
│   ├── experience_replay.py     # ReplayMemory circular transition buffer (off-policy DQN)
│   ├── rollout_buffer.py        # RolloutBuffer on-policy storage (PPO)
│   ├── env_factory.py           # make_env (with action repeat), device, observation tensors
│   ├── eval_loop.py             # Greedy evaluation episodes, run summary, GIF export
│   ├── training_logger.py       # Metrics CSV, best-model tracking, run log
│   ├── dashboard.py             # The 2x2 training dashboard (live runs and saved CSVs)
│   ├── run_dirs.py              # runs/YYYYMMDD_hhmm_<model>/ folders, RunPaths, checkpoint lookup
│   └── plot_learning_curves.py  # --plot: multi-run comparison and dashboards from runs/
│
├── eval/                        # Policy-vs-policy matches
│   ├── mixed_eval.py            # --eval --mix: any DQN variant / PPO as Blue vs. mirrored Orange (env episodes)
│   └── head_to_head.py          # --head_to_head: continuous scoreboard match (sim level)
│
└── sim/                         # 2D simulation (physics, controllers, AI opponent, rendering)
    ├── game.py                  # Interactive & headless game loops (play_interactive, play_headless)
    │
    ├── core/                    # 2D physics engine & entities (Pymunk)
    │   ├── actions.py           # Unified 2D CarAction dataclass
    │   ├── arena.py             # Field walls, elevated goals, and deflection ramps
    │   ├── ball.py              # Rigid-body ball with aerodynamic drag and spin decay
    │   ├── car.py               # Chassis dynamics, raycast suspension, boost, jump-dodge
    │   ├── simulation.py        # Physics manager, contact callbacks, sub-stepping loop, goals
    │   ├── observation.py       # build_observation(sim, ego): the 52 floats, Orange view mirrored
    │   ├── spawns.py            # Kickoff rotation and random spawn sampling (seeded, sim-owned RNG)
    │   └── README.md            # Deep-dive physics & mathematical formulation
    │
    ├── control/                 # Car input sources, all with get_action() / reset()
    │   ├── human.py             # Keyboard
    │   ├── scripted.py          # Heuristic bot, idle, random
    │   └── policy.py            # A trained checkpoint (mirrored on Orange)
    │
    ├── ai/                      # Scripted adversarial baseline bot
    │   ├── heuristic_bot.py     # Hierarchical Zone & Intercept State Machine (HZISM)
    │   └── README.md            # State machine architecture & decision logic
    │
    └── visualization/           # Graphic visualization & HUD (Pygame)
        ├── renderer.py          # Arena, ball, raycasts, vector debug overlays
        ├── car_sprite.py        # Procedural Octane car sprite
        ├── hud.py               # Scoreboard, telemetry, raycast button, boost meter
        ├── event_handling.py    # Window / keyboard / mouse events
        └── README.md            # Coordinate mapping, HUD specs, and visual features

```

---

## 2. Learning Agents (`learning_agents/`)

Both learning agents run in the same episode loop (`trainer.py`) and share an identical feature trunk (`LayerNorm`-stabilized 2-layer MLP of width $256$, totaling $150,949$ trainable parameters), `Adam` learning rate ($3 \times 10^{-4}$), discount factor ($\gamma = 0.99$), and $4\times$ action repeat ($15\text{ Hz}$ decision frequency with edge-triggered jump release):

- **Dueling Double DQN (`learning_agents/dueling_ddqn_agent.py`)**:
  - **Off-Policy Replay**: Stores transitions $(s_t, a_t, s_{t+1}, R_t, \text{terminated}_t)$ in `ReplayMemory` (`experience_replay.py`, capacity $100,000$, warm-up $2,000$ steps) and updates on $64$-sample minibatches every `train_freq = 4` decisions.
  - **Target Decoupling**: Uses the online network to select the greedy next action $a^* = \arg\max_{a'} Q_\theta(s_{t+1}, a')$ and the periodic target network $Q_{\theta^-}$ (synced every $1,000$ steps) to evaluate it, trained via Huber loss (`SmoothL1Loss`) and global gradient norm clipping ($10.0$).
- **Proximal Policy Optimization (`learning_agents/ppo_agent.py`)**:
  - **On-Policy Rollouts & GAE**: Collects fixed-horizon trajectories ($T = 4,096$ macro-steps) in a rollout buffer, bootstrapping value tails on episode timeouts (`truncated=True`) while zeroing terminal goals (`terminated=True`), and computes Generalized Advantage Estimation ($\text{GAE}(\gamma=0.99, \lambda=0.95)$).
  - **Clipped Surrogate Optimization**: Normalizes advantages across the entire $4,096$-sample rollout batch and performs $K = 6$ epochs of shuffled minibatch ($256$) updates using the clipped surrogate objective ($\epsilon = 0.2$), MSE critic loss ($c_1 = 0.5$), and entropy bonus ($c_2 = 0.01$).

---

## 3. Gymnasium Environment API (`gym_env/rocket_league_env.py`)

The environment implements the standard `gymnasium.Env` interface:

```python
import gymnasium as gym
from src.gym_env.rocket_league_env import RocketLeagueEnv

# Instantiate directly or via gym.make; one step() is one 60 Hz sim frame
env = RocketLeagueEnv(render_mode="human", bot_type="heuristic_bot")
obs, info = env.reset(seed=0)  # the seed fixes the random spawn sequence

# The agents decide every 4 frames (15 Hz), releasing jump after the first frame
from src.gym_env.wrappers import ActionRepeat

env = ActionRepeat(env, 4)
```

### A. Action Space

- `Discrete(36)` (default when `discrete_action=True`) or `MultiDiscrete([9, 2, 2])` (when `discrete_action=False`):
    - Direction (9): `0`: None, `1`: Right, `2`: Left, `3`: Down, `4`: Up, `5`: Down-Right, `6`: Up-Right, `7`: Down-Left, `8`: Up-Left
    - Jump (2): `0`: Off, `1`: Jump / Dodge flip
    - Boost (2): `0`: Off, `1`: Rocket boost
    - In flattened `Discrete(36)` mode, action index $a \in \{0, \dots, 35\}$ decodes as `direction = a // 4`, `jump = (a % 4) // 2`, `boost = a % 2` (`gym_env/action_codec.py` is the only code that knows this layout).

### B. Observation Space (52-Dimensional Flat Vector)

- **Space**: `gym.spaces.Box(low=-1.0, high=1.0, shape=(52,), dtype=np.float32)`
- Built by `build_observation(sim, ego)` in `sim/core/observation.py` (named index constants for every block below). Orange's view is the Orange-as-ego vector passed through `mirror_observation`: x-like features change sign and the rays take the cyclic order `[0, 7, 6, 5, 4, 3, 2, 1]`, so a policy trained on Blue can drive Orange. `env.get_observation(team)` returns either view:

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

### C. Reward Function Formulation (`gym_env/rewards.py`, constants in `GymConfig`)

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

Input is keyboard only (a connected gamepad is detected and named at startup, but its buttons are not read).

| Action                  | Keyboard Mapping                     | Description                                         |
| :---------------------- | :----------------------------------- | :-------------------------------------------------- |
| **2D Direction Vector** | `W`/`S`/`A`/`D` or `Arrow Keys`      | Cardinal / diagonal steering input                  |
| **Jump / Flip**         | `Spacebar`                           | Jump 1 from ground; Jump 2 (neutral or flip) in air |
| **Rocket Boost**        | `Left/Right Shift`, `O`, `J`         | Aerial propulsion thrust                            |
| **Reset Field**         | `R`                                  | Resets car and ball to kickoff positions            |
| **Toggle AI Bot**       | `B`                                  | Adds or removes the Orange car                      |
| **Toggle Raycasts**     | `L` or click the HUD button          | Shows/hides 8 radial boundary raycasts              |
| **Quit**                | `Escape`                             | Exits simulation                                    |

---

## 5. Module Deep Dives

For exhaustive mathematical equations and subsystem specifications, see:

- [Core Physics Engine (`src/sim/core/README.md`)](sim/core/README.md)
- [Opponent AI Controller (`src/sim/ai/README.md`)](sim/ai/README.md)
- [Pygame Rendering Engine (`src/sim/visualization/README.md`)](sim/visualization/README.md)
