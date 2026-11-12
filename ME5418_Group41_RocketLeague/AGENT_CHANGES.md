# A2C Agent: Change Log

This log covers finishing `RocketLeagueAgent` (`src/agent/agent.py`) and the changes that work needed elsewhere in the project.

## Usage

```bash
python -m src --train                          # train from scratch (100k episodes) -> runs/a2c_<timestamp>/
python -m src --train --episodes 5000          # set the number of episodes
python -m src --train --pretrain runs/<run>/model.pt   # start from a saved checkpoint
python -m src --play --agent runs/<run>/model.pt       # watch a trained agent in a window
tensorboard --logdir runs/                     # view live training curves
```

Each training run writes these files to the folder that holds `model.pt`:

| File | Contents |
|---|---|
| `model.pt` | Network and optimizer state |
| `metrics.csv` | One row per episode: `episode, total_steps, reward, length, outcome, mean_loss` |
| `tensorboard/` | TensorBoard event files |
| `gifs/episode_<N>.gif` | Every `visualise_n_episodes`-th training episode (default 100), recorded while it runs in a window |
| `plots/kpi.png` | Episode reward, episode length, blue goal rate and A2C loss, plotted by episode |

## Code changes

### `src/agent/agent.py` (completed)
The file was a pseudocode skeleton. It now has a working implementation that keeps the skeleton's method names.

- **`RocketLeagueAgent(model_filepath, visualise_n_episodes=100, bot_type="none", rollout_length=16, seed=None, **a2c_kwargs)`**
  - Creates the env with `gym.make(GymConfig.gym_id, ...)`, which ends episodes at 3000 steps. The A2C model's input and output sizes are read from the env.
  - If a `metrics.csv` already exists in the run folder, episode and step numbers continue from it. Resumed runs append to the log without overlapping the earlier episodes.
- **`reset(seed=None)`** resets the env and the per-episode counters.
- **`learn(number_of_episodes, pretrain_model_filepath=None)`** trains with n-step A2C:
  - Collects `rollout_length` steps, then computes discounted returns backwards. The return is bootstrapped from V(s_last), using the same scheme as `test_a2c_torch`.
  - Calls `A2C._backprop` once per rollout.
  - When an episode ends because of the time limit rather than a goal, γ·V(s′) is added to the last reward. A timeout is not a true terminal state.
  - Every `visualise_n_episodes` episodes (default 100), that training episode runs in its own env with `render_mode="human"` and `gif_path=...`. It's a normal training episode: actions are sampled and updates continue. The episode plays in a window in real time (60 Hz, so up to 50 s for 3000 steps). When it ends, the GIF is saved even on a timeout, the window closes and a checkpoint is saved. All other episodes run in the headless env.
  - At the end it saves the model, writes the plots and closes the TensorBoard writer.
- **`tensorboard_logging(stats)`** logs `episode/reward`, `episode/length`, `episode/outcome` (−1 orange goal, 0 timeout, 1 blue goal) and `train/loss`.
  - Tensorboard is imported the first time this method runs, not at the top of the file. If TensorFlow is installed, tensorboard loads it, and loading it before torch has finished loading crashes with an LLVM error.
  - If tensorboard isn't installed, the method prints one notice and skips logging.
- **`csv_logging(stats)`** appends one row per episode to `metrics.csv`.
- **`save_plots(window=100)`** draws a 2×2 chart from `metrics.csv` to `plots/kpi.png`. It uses matplotlib's Agg backend, so no display is needed.
- **`play(sim)`** returns a greedy `CarAction` for the blue car from `sim.get_state_norm(as_flat_array=True)`.
- **`train_agent(...)`** runs training. The default output path is `runs/a2c_<timestamp>/model.pt`.
- **`play_agent(model_filepath, bot_type="none")`** loads a checkpoint and shows the agent playing in a window.

The episode's `mean_loss` is the average loss of the rollouts that finished during that episode. The update that crosses an episode boundary counts toward the next episode.

### `src/agent/__init__.py` (new)
Empty file that makes `src.agent` a package.

### `src/neural_nets/a2c_torch.py`
- Added `A2C.save(path)`, which saves `{"model": state_dict, "optimizer": state_dict}` and creates the parent directory if needed.
- Added `A2C.load(path)`, which restores both state dicts (`map_location="cpu"`).
- Added `import os`.
- `_forwardpass`, `_backprop`, `act` and `predict` are unchanged.

### `src/gym_env/rocket_league_env.py`
- Moved the code that turns an action index into a `CarAction` out of `step()` into a module-level function, `decode_action(action, discrete_action=True)`. `step()` now calls it, and `RocketLeagueAgent.play()` uses it without needing an env.

### `src/main.py`
- Added `--train [--episodes N] [--pretrain PATH]`, which calls `train_agent`.
- Added `--play --agent PATH`, which calls `play_agent`.
- Added a small helper, `_arg_value(flag, default)`, that reads the value after a flag.
- Added both commands to the usage text.

### `environment.yml`
- Added `tensorboard`.
- Added `pillow`, which `src/gym_env/gif_saver.py` already needed but which wasn't listed.

## Conda environment changes (`me5418_projenv`)
- **Installed** `tensorboard==2.21.0`, which also pulled in `tensorboard-data-server`, `markdown` and `werkzeug`.
- **Uninstalled** `tensorflow==2.21.0`. Nothing in the project used it, and it caused the duplicate-LLVM crash described above. Its leftover dependencies (`keras`, `h5py`, etc.) are still installed and are not used.

## Verification
Tests were run in `me5418_projenv` with episodes capped at 100–150 steps:
- Training produced `model.pt`, `metrics.csv`, `plots/kpi.png`, a GIF and TensorBoard event files. Every N-th episode trained in a `human` render env and saved its GIF. Each of those episodes opened exactly one window, and no window was left open after `learn()` returned. All four TensorBoard metrics were logged for every episode.
- Resuming with `pretrain_model_filepath` loaded the checkpoint and continued episode numbering in the CSV.
- `play(Simulation())` returned a valid `CarAction`.
- The gym env checker (`my_check_env`), `test_a2c_torch` and the `python -m src` usage text still work.
- **Not tested:** `--play --agent` (it needs a display), and a full-length training run.

## Merge into `bentjh01_env`: two environments
`bentjh01_env` was merged onto this layout. bentjh's env is ported as `src/gym_env/bentjh_gym.py` (`RocketLeagueSideSwipeGym`, gym id `bentjh01_2d_rocketleague_v1`). It sits alongside `RocketLeagueEnv` (`rocketleague_v1`).

| | `rocketleague_v1` | `bentjh01_2d_rocketleague_v1` |
|---|---|---|
| Observation | `sim.get_state_norm(as_flat_array=True)` | 19-dim from `sim.get_state_norm()` (car orientation as sin, cos), built in `gym_env/sim2gym_layer.py` |
| Action | `Discrete(36)` via `decode_action` | same |
| Reward | `RocketLeagueEnv._calculate_reward` | `gym_env/reward.py` (`GymReward`): ball velocity towards the opponent goal, ±`BentjhGymConfig.score_gain` on a goal |

- Both envs take `render_mode`, `bot_type` and `gif_path`. Both expose `observe(sim)`, which `RocketLeagueAgent.play()` uses so the observation matches the env the model was trained on.
- `RocketLeagueAgent`, `train_agent` and `play_agent` take `env_id`.
- `sim2gym_layer.sim_action_to_gym_action` is the inverse of `decode_action`.

```bash
python -m src --train --env bentjh [--episodes N]          # train A2C on bentjh's env
python -m src --play --agent runs/<run>/model.pt --env bentjh
python -m src --demo --bentjh_gym                          # drive with keyboard, prints reward
```
