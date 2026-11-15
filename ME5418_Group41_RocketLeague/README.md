# ME5418_RocketLeague

> Team 41:
> Lu Jia Yih Jensen, Benjamin Teh Jhen Hing, Wee Fook Choon

---

A high-fidelity 2D vertical-plane robotic simulation platform and deep reinforcement learning framework modeling vehicle-ball interaction, aerial maneuvering, and autonomous control via **Dueling Double Deep Q-Network (Dueling DDQN)** and **Proximal Policy Optimization (PPO)** inspired by **Rocket League Sideswipe**.

## ![Demo](./assets/demo.gif)

---

## 1. Project Overview

- **Physics Engine**: Deterministic $240\text{ Hz}$ physics solver ($60\text{ Hz} \times 4\text{ sub-steps}$) powered by Pymunk (Chipmunk2D).
- **Vehicle Dynamics**: Raycast spring-damper suspension, Coulomb tire traction, 2D vector steering, two-stage jump-dodge ballistics, rocket boost aerodynamics, and turtle auto-recovery.
- **Opponent AI**: Rule-based Hierarchical Zone & Intercept State Machine (HZISM) acting as an adversarial baseline.
- **RL Interface**: Standard Gymnasium environment (`rocketleague_v1`) with normalized continuous observation space and discrete/factored action spaces (`Discrete(36)` / `MultiDiscrete([9, 2, 2])`).
- **Neural Network Architectures (`src/neural_nets/`)**: Matched-capacity PyTorch architectures (`150,949` trainable parameters) with `LayerNorm`-stabilized shared feature trunks for:
  - **Dueling Double DQN (`src/neural_nets/dueling_ddqn_torch.py`)**: Decoupled State-Value $V(s)$ and Action-Advantage $A(s, a)$ streams.
  - **PPO Actor-Critic (`src/neural_nets/ppo_torch.py`)**: Decoupled categorical Policy head $\pi_\theta(a|s)$ and Critic head $V_\phi(s)$.
- **Learning Agents (`src/learning_agents/`)**: Full training and evaluation pipelines with action repeat ($4\times$), truncation-aware value bootstrapping, checkpointing, and live learning curves:
  - **Off-Policy Dueling DDQN Agent (`src/learning_agents/dueling_ddqn_agent.py`)**: Experience replay (`src/learning_agents/experience_replay.py`), Double-DQN target decoupling, Huber loss, and periodic target network synchronization.
  - **On-Policy PPO Agent (`src/learning_agents/ppo_agent.py`)**: Fixed-horizon rollout collection, Generalized Advantage Estimation ($\text{GAE}(\gamma, \lambda)$), and multi-epoch clipped surrogate minibatch updates.

## 2. Architecture & Modular Documentation

Detailed technical specifications and mathematical formulations are organized modularly within each subsystem directory:

- 🤖 **[Learning Agents (`src/learning_agents/README.md`)](./src/learning_agents/README.md)**: Training (`--train`) and evaluation (`--eval`) of `DuelingDDQNAgent` (`dueling_ddqn_agent.py`) and `PPOAgent` (`ppo_agent.py`) through one shared training loop (`trainer.py`), plus policy-vs-policy matches, round-robin tournaments and best-checkpoint selection in `src/eval/`.
- 🧠 **[Neural Networks (`src/neural_nets/README.md`)](./src/neural_nets/README.md)**: PyTorch `nn.Module` definitions (`dueling_ddqn_torch.py` and `ppo_torch.py`), checkpoint loading, and standalone forward/backward pass verification (`--demo`).
- 📚 **[System & RL Environment (`src/README.md`)](./src/README.md)**: Package structure, Gymnasium observation/action spaces, reward shaping formulation, and control scheme.
- ⚙️ **[Physics & Vehicle Kinematics (`src/sim/core/README.md`)](./src/sim/core/README.md)**: Suspension equations, Coulomb traction, attitude control, jump ballistics, ball restitution/drag, and arena geometry.
- 🛡️ **[Opponent AI Architecture (`src/sim/ai/README.md`)](./src/sim/ai/README.md)**: Finite state machine structure, kickoff tactics, shadow defense, zone goalie interception, and aerial saves.
- 🎨 **[Visualization & HUD Telemetry (`src/sim/visualization/README.md`)](./src/sim/visualization/README.md)**: Pygame rendering pipeline, procedural Octane chassis, vector debug visualizers, and HUD layout.

---

## 3. Quickstart & Usage

### Step 1: Extract the zipped folder, then open the directory

```bash
cd ./ME5418_Group41_RocketLeague
```

### Step 2: Environment Setup

Create and activate the Conda environment using the provided configuration:

```bash
conda env create -f environment.yml
conda activate me5418_projenv
```

_(Alternatively, install the core dependencies via pip: `pip install torch gymnasium pygame pymunk numpy matplotlib`)_

### Step 3: Train & Evaluate Learning Agents

Every mode lists its options with `--help` (e.g. `python -m src --eval --help`). All learning agent training and evaluation commands are executed from the project root (`ME5418_Group41_RocketLeague/`). Each training run creates its own folder `runs/YYYYMMDD_hhmm_<model_name>/` (e.g. `runs/20261113_1655_ppo/`) and saves its checkpoint (`model_<model_name>.pt`), training log, metrics CSV, and reward/loss plots there. The current model is also snapshotted every `checkpoint_interval` episodes (set in `src/config/nn_config.py`, default 500, `0` disables) to `checkpoints/model_<model_name>_ep<NNNNNN>.pt` inside the run folder; these can be passed to `--model_path` like any other checkpoint. Evaluation without `--model_path` loads the checkpoint of the most recent run of that model.

#### Train a Learning Agent

```bash
# Train Dueling Double DQN (default DQN architecture)
python -m src --train --dueling_ddqn

# Train Proximal Policy Optimization (PPO)
python -m src --train --ppo

# Optional DQN ablation variants:
python -m src --train --double_dqn
python -m src --train --dqn

# To start from a checkpoint for training (any of --dueling_ddqn / --double_dqn / --dqn / --ppo)
python -m src --train --ppo --model_path path_to_model.pt

# Limit the number of episodes
python -m src --train --ppo --episodes 2000
```

#### Curriculum Learning

Trains one agent through three stages, each starting from the previous stage's best model: no opponent, then the heuristic bot, then self-play rounds against a frozen copy of the last best model (refreshed every round). A stage ends when the 100-episode mean reward has not improved for `--patience` episodes. Each stage / round gets its own run folder (`runs/<ts>_<name>_s1_none/`, ...), and the final model is copied to `runs/<ts>_<name>_curriculum/model_<name>.pt`.

```bash
python -m src --curriculum --ppo [--patience 1000] [--self_play_rounds 5] [--max_stage_episodes N]
python -m src --curriculum --dueling_ddqn   # also --double_dqn / --dqn
```

#### Evaluate a Trained Learning Agent

Optional flags for `--eval`: `[--episodes N]`, `[--no_random_spawn]` (use fixed kickoff spawns for reproducible comparisons), `[--gif]` (record the first episode as a GIF next to the checkpoint), `[--gif_path out.gif]`.

```bash
# 1. Blue = Policy, Orange = Heuristic Bot (default --blue mode)
python -m src --eval --dueling_ddqn [--render] [--model_path path_to_model.pt]
python -m src --eval --ppo          [--render] [--model_path path_to_model.pt]

# 2. Orange = Mirrored Policy, Blue = Human User (WASD/Arrows + Space + Shift)
python -m src --eval --dueling_ddqn --orange --render [--model_path path_to_model.pt]
python -m src --eval --ppo          --orange --render [--model_path path_to_model.pt]

# 3. Same-Policy Bot-vs-Bot (both Blue and Orange controlled by the same trained policy)
python -m src --eval --dueling_ddqn --bot-vs-bot [--render] [--model_path path_to_model.pt]
python -m src --eval --ppo          --bot-vs-bot [--render] [--model_path path_to_model.pt]

# 4. Mixed-Policy Bot-vs-Bot (e.g., Blue = Dueling DDQN vs Orange = PPO, or vice versa)
python -m src --eval --mix --dueling_ddqn --ppo [--render] [--blue_model path_b.pt] [--orange_model path_o.pt]
python -m src --eval --mix --ppo --dueling_ddqn [--render] [--blue_model path_b.pt] [--orange_model path_o.pt]
```

#### Continuous Head-to-Head Match (`--head_to_head`)

Run a continuous scoreboard match (goal kickoffs reset automatically; supports PPO, Dueling DDQN/DQN checkpoints via `--blue`/`--orange`, or `none`/`bot` for `HeuristicBot`).

```bash
# Default: checkpoint of the most recent training run in runs/ (or specify checkpoints / 'none' for HeuristicBot)
python -m src --head_to_head [--blue PATH|none] [--orange PATH|none] [--random_spawn] [--seed N]

# Headless match with score/time limits
python -m src --head_to_head --headless --max_goals 5 --max_seconds 120 --blue runs/YYYYMMDD_hhmm_dueling_ddqn/model_dueling_ddqn.pt --orange none
```

#### Pick the Best Checkpoint in a Run (`--best_checkpoint`)

Find the strongest model in a training run folder. Every model (`model_*.pt` plus `checkpoints/*.pt`) first plays the heuristic bot once from each side; only models with a positive goal difference qualify. The qualifiers then play king of the hill: each challenger plays the current champion once from each side, and takes over if its goal difference exceeds `--threshold`.

```bash
# Default: 300 s games, fixed kickoffs, challenger must win by more than 3 goals
python -m src --best_checkpoint runs/YYYYMMDD_hhmm_ppo

# Shorter games with repeatable random kickoffs and a lower threshold
python -m src --best_checkpoint runs/YYYYMMDD_hhmm_ppo --max_seconds 120 --random_spawn --seed 0 --threshold 1
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `RUN_DIR` | _(required)_ | Training run folder containing `model_*.pt` and/or `checkpoints/*.pt`. |
| `--threshold N` | `3` | Goal difference (over both sides) a challenger must **exceed** to replace the champion. |
| `--max_seconds S` | `300.0` | Simulation time per game in seconds. |
| `--max_goals N` | _none_ | End a game early after this many total goals. |
| `--random_spawn` | `False` | Randomise every kickoff. |
| `--seed N` | _none_ | Seed for repeatable random spawns. |

The best model's path is printed at the end, along with a summary of every challenge. If no model beats the heuristic bot, the command stops with an error naming the closest model.

#### Plot Learning Curves & Dashboards

```bash
# Auto-discover metrics in runs/ (including per-run folders) and generate multi-algorithm comparative curves
python -m src --plot

# Generate a 4-panel dashboard for a specific agent run
python -m src --plot --dashboard runs/YYYYMMDD_hhmm_ppo/metrics_ppo.csv
```

### Step 4: Interactive Play & Component Demos

#### Play the game interactively

```bash
# Play interactively as Blue car vs Orange Heuristic Bot (default)
python -m src --play
python -m src --play --blue

# Play interactively as Orange car vs Blue Heuristic Bot
python -m src --play --orange

# Watch bot-vs-bot match (Blue Heuristic Bot vs Orange Heuristic Bot)
python -m src --play --bot-vs-bot

# Run headless simulation benchmark
python -m src --play --headless
```

#### Demo the Gymnasium environment

```bash
python -m src --demo --gym
```

#### Demo the neural network forward & backward passes

Run the neural network demo to instantiate the network and verify a complete forward pass and backward pass (`loss.backward()` layer gradients $\to$ `optimizer.step()`):

```bash
# Dueling Double DQN (Default DQN architecture)
python -m src --demo --dueling_ddqn

# Double DQN (Single-stream head with Double-DQN target)
python -m src --demo --double_dqn

# Vanilla DQN (Single-stream head with standard max target)
python -m src --demo --dqn

# PPO Actor-Critic (Policy + Value heads with clipped surrogate + GAE update)
python -m src --demo --ppo
```

## 4. Evaluate across models

Run a headless side-swapped round-robin tournament across all `.pt` checkpoints placed in a folder (default: `eval_model/`) together with `HeuristicBot`, using the continuous simulation engine (`Simulation` + `PolicyController` / `HeuristicController`):

```bash
# Default: evaluate all checkpoints in eval_model/ + HeuristicBot
# (100 matches per directed Blue-vs-Orange pairing, 1-goal / 60s cap, seed 50, random spawns)
python -m src --round_robin

# Custom folder, output directory, match budget, seed and worker count
python -m src --round_robin \
  --models_dir eval_model \
  --out_dir eval_model/eval_report \
  --matches 100 \
  --max_goals 1 \
  --max_seconds 60 \
  --seed 50 \
  --workers 11
```

### Options (`python -m src --round_robin --help`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--models_dir DIR` | `eval_model` | Folder containing curated `.pt` checkpoints (architecture auto-inferred). |
| `--out_dir DIR` | `eval_model/eval_report` | Output folder for CSVs, raw logs, and Markdown summary report. |
| `--matches N` | `100` | Matches played per directed `(Blue, Orange)` pairing. |
| `--max_goals N` | `1` | Goals required to end a match (`0` disables goal cap). |
| `--max_seconds S` | `60.0` | Simulation time limit per match in seconds (`0-0` timeout = Draw). |
| `--seed N` | `50` | Base seed for repeatable random spawns (match `i` uses `seed + i`). |
| `--no_random_spawn` | `False` | Use fixed kickoff spawns instead of random spawns. |
| `--no_heuristic` | `False` | Exclude `HeuristicBot` from the tournament. |
| `--no_mirrors` | `False` | Skip `Bot vs Bot` mirror pairings. |
| `--workers N` | `cpu_count - 1` | Number of parallel CPU worker processes (`1` runs sequentially). |

### Generated Artifacts (`--out_dir`)

- `overall_standings.csv`: Ranked overall standings (non-mirror and all-match win rate, goal differential, and Blue/Orange splits).
- `pairings_summary.csv`: Per-pairing statistics for all directed `(Blue, Orange)` matchups.
- `matrix_score_b_o_d.csv`: $N \times N$ directed matrix (`BlueWins-OrangeWins-Draws`).
- `matrix_combined_200_matches.csv`: $N \times N$ side-swapped combined matrix (`RowWins-ColWins-Draws`).
- `matrix_blue_win_rate_pct.csv` & `matrix_blue_goal_diff.csv`: Blue win-rate and goal-differential matrices.
- `raw_matches.csv` & `raw_matches.log`: Per-match logs (`pairing_id`, `seed`, `winner`, goals, `sim_seconds`, `steps`, `last_touch`).
- `eval_report.md`: Markdown summary containing the standings and head-to-head matrices.

---

## AI Tool Declaration

We used Google Gemini and Claude Code to write and debug code. We are responsible for the content and quality of the submitted work.
