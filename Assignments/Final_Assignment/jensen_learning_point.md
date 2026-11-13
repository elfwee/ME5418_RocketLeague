ME5418 - Final Project: Engineering, Physics Symmetry & Training Optimization Learning Points
Team 41 - Lu Jia Yih Jensen

1 Context & Scope (New Post-Baseline Challenges)
Building upon the earlier MDP, reward-shaping, and curriculum lessons captured in `final_learning_point.txt`, the final phase of the project shifted from single-agent feasibility to training convergence speed, cross-team symmetry (Blue vs. mirrored Orange), multi-agent evaluation (`--mix`, `--head_to_head`), and branch integration (`dueling_dqn`, `bt_train`, and `main`). Testing real trained models exposed subtle interactions between rigid-body solver accuracy, environment I/O overhead, neural network checkpoint schemas, and collaborative Git workflows.

2 Deep Dive: Problems Encountered, Root Causes & Lessons Learned

• Case 1: Physics Constraint Under-Convergence Breaking Policy Mirroring (SOLVER_ITERATIONS: 14 -> 60)
  - Problem Observed: Policies trained solely from the Blue perspective (left-to-right) failed to perform consistently when controlling the Orange car (right-to-left) via horizontally mirrored observations (`get_state_norm_mirror`) and mirrored actions (`dir_x = -dir_x`), despite exact mathematical mirroring of all 52 input features (`max_diff = 0.000e+00` verified against `get_orange_perspective_obs`). By contrast, a model trained at higher solver iterations exhibited clean mirrored play on both sides.
  - Root Cause Diagnosis:
    1. Pymunk (Chipmunk2D) resolves simultaneous multi-contact constraints (chassis wheel raycasts/ground support, ball-car impulse, and car-car collisions) iteratively. At `SOLVER_ITERATIONS = 14`, constraint impulses had not fully converged, causing residual order-dependent asymmetry between left-facing and right-facing impacts—enough to push mirrored Orange states off the Blue policy's trained state manifold.
    2. Static entity registration order also gave one car consistent priority in Pymunk's contact solver and spawn clamping (`sample_random_spawns()` previously shifted only `ox_car` outward when cars spawned within 2.0 m).
  - Implemented Fix:
    1. Increased `SOLVER_ITERATIONS` in `src/config/sim_config.py` from `14` to `60` so multi-contact rebounds converge cleanly across mirrored orientations.
    2. Added `Simulation._set_orange_solver_first(((self._spawn_index // 2) % 2) == 1)` in `src/sim/core/simulation.py` to alternate primary Pymunk solver/update order over a 4-kickoff cycle, and centered minimum spawn-separation clamping symmetrically around `mid_x` in `sample_random_spawns()`.
  - Key Lesson Learned: Observation-space and action-space symmetry cannot guarantee symmetric agent behavior if the underlying physics integrator is under-converged or order-biased. Furthermore, even with unbiased `60`-iteration dynamics (`Blue 2 - 0 Orange` in `7.03s` vs. `Blue 0 - 2 Orange` in `7.03s` under `--head_to_head --orange none` / `--blue none`), chaotic head-on 3-body collisions near `x = 17.5` naturally diverge from `~1e-14` floating-point rounding after ~9-16 s rallies—meaning short-horizon symmetry tests are diagnostic while long-horizon rally parity is statistical.

• Case 2: RL Training Throughput Bottlenecks Hidden Outside PyTorch Backpropagation
  - Problem Observed: Comparing refactored training on `main` against the earlier `dueling_dqn` branch (`483f9c7`) revealed substantially slower wall-clock training progress for Dueling Double DQN under identical hyperparameters (`DQNConfig`: `hidden_size=256`, `batch_size=64`, `train_freq=4`, `action_repeat=4`, `learning_rate=3e-4`, `episode_max_steps=3_600`).
  - Root Cause Diagnosis: Profiling the step loop showed the GPU forward/backward pass was not the bottleneck; instead, redundant per-frame work inside `step()` and logging dominated wall time:
    1. During `action_repeat = 4` frames (`_step_with_action_repeat` and `evaluate_mixed_policies`), every intermediate substep executed full `_get_obs()` construction—including 8 Pymunk segment raycasts per active car—only for those 3 intermediate observations to be discarded.
    2. `RocketLeagueEnv._calculate_reward()` called `self.sim.get_state_norm()` (or `get_state_norm_mirror()`) on every 60 Hz frame purely to read scalar positions, `last_touch`, and `goal_scored_step`, triggering full dictionary assembly and boundary raycasts inside the reward function.
    3. `Simulation.get_state_norm()` computed opponent boundary raycasts even when `as_flat_array=True`, despite opponent raycasts having been pruned from the 52-float observation vector.
    4. `save_graph()` computed rolling 100-episode reward/goal averages and 50-episode smoothed loss using an `O(N^2)` Python loop (`np.mean(values[max(0, i-99):i+1])`), stalling training runs as episode counts climbed past 100,000, compounded by `10s` plotting intervals at `dpi=150`.
    5. Resuming training from a saved checkpoint initialized `epsilon` back to `1.0`, overwriting learned Q-values with random exploration before settling.
  - Implemented Fix:
    1. Added `_defer_obs` in `RocketLeagueEnv.step()` (returning preallocated `_dummy_obs` on intermediate `action_repeat` substeps unless `terminated` or closed) and queried `_get_obs()` only once at the end of the repeated macro-step.
    2. Rewrote `RocketLeagueEnv._calculate_reward()` to read `self.sim.last_touch`, `self.sim.goal_scored_this_step`, `_normalize_pos(self.sim.ball.position)`, and `ARENA_DIAGONAL`-normalized car-ball Euclidean distance directly without calling `get_state_norm()`.
    3. Added `_include_blue_rays` / `_include_orange_rays` flags to `Simulation.get_state_norm()` and `get_state_norm_mirror()` so flat observations skip unused opponent raycast segment queries while dictionary telemetry retains them.
    4. Replaced `O(N^2)` smoothing in `DuelingDDQNAgent.save_graph()` and `PPOAgent.save_graph()` with vectorized `np.cumsum`-based `_rolling_mean()` (`O(N)`), relaxed periodic dashboard writes to `30s` intervals at `dpi=100`, and introduced `DQNConfig.epsilon_finetune = 0.20` when resuming from a checkpoint.
  - Key Lesson Learned: In physics-based RL, environment telemetry (raycasts, dict serialization) and `O(N^2)` metric plotting silently consume more CPU time than neural network updates over multi-million-step runs (`>10.46M` steps / `121,080` episodes in `runs/metrics_dueling_ddqn.csv`). Deferred observation extraction (`_defer_obs`) and zero-overhead reward access dramatically accelerate off-policy and on-policy loops alike without altering MDP dynamics.

• Case 3: Guarding Training-Evaluation Distribution Consistency (`random_spawn`)
  - Problem Observed: While debugging solver symmetry, `random_spawn=False` was temporarily hardcoded inside `RocketLeagueEnv.__init__`. Leaving fixed spawns inside the environment constructor silently degrades domain randomization during training whenever anyone runs `--train`. Conversely, evaluating two agents head-to-head without an option for fixed kickoff spawns introduces variance.
  - Implemented Fix: Restored `random_spawn: bool = True` as the default constructor parameter on `RocketLeagueEnv`, plumbed `random_spawn` cleanly through `DuelingDDQNAgent.run`, `PPOAgent.run`, `evaluate_dueling_ddqn`, `evaluate_ppo`, and `evaluate_mixed_policies`, and exposed `--no_random_spawn` / `--fixed_spawn` strictly as evaluation CLI flags in `src/main.py`.
  - Key Lesson Learned: Never bake temporary debugging toggles into shared environment defaults. Parameterize determinism at the evaluation entrypoint so training preserves full kickoff generalization while A/B benchmarking remains reproducible on demand.

• Case 4: Checkpoint Self-Description, Evaluation Safety & CLI Resilience
  - Problems Observed: Integrating `--eval --mix`, `--eval --ppo`, and PR #10's `--head_to_head` exposed three brittleness bugs:
    1. `evaluate_mixed_policies`, `PPOAgent.run`, and `src/main.py` detected network architectures using outdated `state_dict` key names (`"actor_dir.weight"`, `"actor_stream.2.weight"`, `"fc_value.weight"`, `startswith("actor_")`) and constructed `ActorCritic` from `PPOConfig` defaults before loading weights—failing on flat `Discrete(36)` PPO (`"actor.weight"`) vs. factored `MultiDiscrete([9, 2, 2])` PPO (`"actor_heads.0.weight"`) or mismatched `hidden_dim`.
    2. Omitting `--episodes` during headless evaluation (`render=False`) passed `num_episodes=None` into `itertools.count()`, creating an infinite evaluation loop.
    3. `--head_to_head` hardcoded a missing developer-local checkpoint path (`runs/20261106_0911/model_ppo.pt`), crashing on clean checkouts.
  - Implemented Fix:
    1. Replaced stale checks with direct checkpoint introspection (`"actor_heads.0.weight"` / `"actor.weight"` for PPO factored vs. flat, `"advantages.weight"` / `"fc3.weight"` for Dueling vs. standard DQN, `fc1.weight.shape[0]` for `hidden_dim`) and updated `--mix` checkpoint classification in `src/main.py` to match `("actor", "critic", "fc_actor", "fc_critic")`.
    2. Guarded infinite evaluation (`episodes = None`) strictly on `render and num_episodes is None and not save_gif`, otherwise defaulting to `DQNConfig.eval_episodes` / `PPOConfig.eval_episodes` (`5`).
    3. Added `_resolve_default_h2h_model()` and `_h2h_model_arg()` in `src/main.py` to auto-select available checkpoints in `runs/` and allow `--blue none` / `--orange none` to spawn `HeuristicBot`, plus wired `--max_goals` and `--max_seconds` for bounded headless matches.
  - Key Lesson Learned: Saved PyTorch `state_dict` files already encode their exact layer widths and head structure; inferring topology from the checkpoint decouples evaluation/deployment scripts from active training configs. Similarly, unbounded loops (`itertools.count()`) must always require an interactive window (`render=True`).

• Case 5: Cross-Branch Rebasing, Single-Source Mirroring & Duck-Typed Adapter Contracts
  - Problems Observed: After rebasing onto `origin/main` (PR #10), two subtle integration bugs surfaced:
    1. Both `src/sim/core/simulation.py` (`get_state_norm_mirror`) and the incoming `src/learning_agents/orange_perspective.py` (`get_orange_perspective_obs` with local `_mx`, `_mirror_rays`, `_mirror_relation`, `_mirror_car_core`) implemented the same 52-float horizontal mirroring independently. Duplicate index math creates a latent bug hazard whenever observation layouts evolve.
    2. `orange_perspective.py::_mirrored_sim_view()` built a `types.SimpleNamespace` proxy view for `OrangePerspectiveBot` containing `blue_car`, `ball`, `arena`, `center_x`, and `ball_spawn_y`, omitting `time_elapsed`. Running `OrangePerspectiveBot.get_action()` immediately crashed with `AttributeError: 'types.SimpleNamespace' object has no attribute 'time_elapsed'` inside `HeuristicBot.compute_action()`.
    3. `head_to_head.py` called `handle_event(sim, renderer)` without passing `(blue, orange)`, leaving agent `action_repeat` counters stale on manual reset (`R`).
  - Implemented Fix: Numerically verified exact agreement (`max_diff = 0.000e+00`) between `get_orange_perspective_obs(sim)` and `sim.get_state_norm_mirror(as_flat_array=True)` across fixed/random spawns and multiple steps, then replaced the duplicate helpers with delegation to `sim.get_state_norm_mirror(as_flat_array=True)`. Added `time_elapsed=sim.time_elapsed` to `_mirrored_sim_view()` and passed `blue, orange` to `handle_event(sim, renderer, blue, orange)`.
  - Key Lesson Learned: When reconciling parallel team branches, always establish a single canonical source of truth backed by a numerical equivalence check before deleting duplicates, and exercise adapter/proxy objects (`SimpleNamespace`) with runtime smoke tests rather than relying on import checks alone.

3 Empirical Results & Verification Outcomes

• Training Convergence (`SOLVER_ITERATIONS = 60` Dueling Double DQN):
  - Completed `121,080` logged episodes (`10,461,028` steps) in `runs/metrics_dueling_ddqn.csv`.
  - Reached peak checkpoint performance of `20.19` 100-episode mean reward and `99.0%` 100-episode rolling goal rate at episode `106,104` (`step 9,570,559` in `runs/run_dueling_ddqn.log`).

• Post-Reconciliation Evaluation (`runs/model_dueling_ddqn.pt`):
  - `--eval --dueling_ddqn` (Blue vs. HeuristicBot, 5 default episodes): `100.0%` goal rate (`score_blue: 5, score_orange: 0`, `mean_reward: 20.89`).
  - `--eval --dueling_ddqn --orange --episodes 2` (Mirrored Orange vs. HeuristicBot): `100.0%` goal rate (`score_blue: 0, score_orange: 2`, `mean_reward: 21.92`).
  - `--head_to_head --headless --max_goals 2 --max_seconds 30`:
    - `--orange none` (Trained DQN Blue vs. HeuristicBot Orange): `Blue 2 - 0 Orange` in `7.03s` sim time (`0.3s` wall).
    - `--blue none` (HeuristicBot Blue vs. Mirrored DQN Orange): `Blue 0 - 2 Orange` in `7.03s` sim time (`0.4s` wall).
    - Default DQN self-play: `Blue 1 - 1 Orange` in `12.37s` sim time (`0.5s` wall).
  - Cross-policy compatibility (`AgentControl`, `evaluate_ppo`, and `evaluate_mixed_policies` across both factored `(9, 2, 2)` and flat `36`-action `ActorCritic` checkpoints vs. `runs/model_dueling_ddqn.pt`), `OrangePerspectiveBot` smoke test, and `my_check_env()` all passed cleanly.

4 Synthesis Matrix: Final-Phase Diagnostics & Engineering Takeaways

| Problem / Failure Mode | Root Cause | Implemented Solution | Lesson Learned |
| :--- | :--- | :--- | :--- |
| Mirrored Orange policy fails despite identical mirrored features | `SOLVER_ITERATIONS = 14` under-converged Pymunk multi-contact impulses; fixed solver order | Raised `SOLVER_ITERATIONS` to `60`; alternated solver priority via `_set_orange_solver_first()`; symmetric spawn clamp | Symmetrical RL deployment requires converged rigid-body constraint resolution, not just mirrored inputs |
| Slow training throughput vs. baseline branch | Per-substep obs/raycasts inside `action_repeat`, `get_state_norm()` inside rewards, `O(N^2)` graph smoothing | `_defer_obs` substep skipping, lightweight scalar reward calc, conditional opponent raycasts, `O(N)` `cumsum` rolling means | Telemetry, raycast queries, and logging loops easily outweigh PyTorch backprop in physics RL |
| Fixed spawns leaking into training | `random_spawn=False` hardcoded in `RocketLeagueEnv.__init__` | Restored `random_spawn=True` default; added `--no_random_spawn` eval CLI flag | Keep domain randomization intact in `env.__init__`; isolate determinism flags to evaluation CLI |
| `--mix` / `--ppo` / `--head_to_head` checkpoint crashes & infinite headless eval | Stale layer key checks, unguarded `itertools.count()`, missing local default path | Introspected `state_dict` keys/shapes, fell back to `eval_episodes = 5` headlessly, added `_resolve_default_h2h_model()` & `none` bot support | Treat checkpoints as self-describing; gate unbounded eval loops strictly on active rendering |
| Duplicate mirroring logic & `OrangePerspectiveBot` `AttributeError` | Parallel PR duplicated mirroring and built incomplete `SimpleNamespace` proxy (`missing time_elapsed`) | Delegated `get_orange_perspective_obs` to `Simulation.get_state_norm_mirror` (`max_diff = 0.0`); added `time_elapsed` | Deduplicate into one canonical geometry source verified numerically; test duck-typed proxies end-to-end |
