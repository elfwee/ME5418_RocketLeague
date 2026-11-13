# Refactor plan: `ME5418_Group41_RocketLeague`

Branch: `major_refactor`. Scope: all Python under `ME5418_Group41_RocketLeague/src/` (~8,000 lines, 30 files) plus the docs that describe it.

## 1. Goals and ground rules

**Goal:** one place for each piece of knowledge (action encoding, checkpoint loading, paths, hyperparameters), smaller modules with one job each, and a test net so later changes are safe.

**This is a behaviour-preserving refactor.** These must not change:

| Invariant | Why |
| :-- | :-- |
| Every `python -m src ...` command in `README.md` keeps working with the same flags | It is the submission interface |
| The 52-float observation is bit-identical, including the mirrored Orange view | Existing checkpoints depend on it |
| `state_dict` keys (`fc1`, `ln1`, `fc_value`, `advantages`, `fc3`, `fc_actor`, `actor_heads.N`, `actor`, `fc_critic`, `critic`) | Existing `.pt` files must still load |
| Physics, reward values, hyperparameters, 150,949 parameter count | Results in the report must stay reproducible |
| `runs/YYYYMMDD_hhmm_<model>/` layout and metrics CSV columns | `--plot` and `latest_model_path` read old runs |

Anything that would change behaviour is listed in section 5 as a separate decision, not folded into a refactor commit.

**Working rules:** one phase per PR, each commit leaves `python -m src --play --headless` and the test suite green, pure moves are committed separately from edits so diffs stay reviewable.

## 2. What is wrong today

Findings from reading the whole tree, grouped by cause.

### Duplicated knowledge
- **Action encoding** (`direction = a // 4`, `jump = (a % 4) // 2`, `boost = a % 2`) is written out in five places: `RocketLeagueEnv._decode_action`, `agent_common.clear_jump`, `AgentControl._decode_discrete_action` and `_greedy_action_idx`, and both network demos. `NUM_FLAT_ACTIONS = 36` is redefined in `mixed_eval.py`, and `36` is a literal in `ppo_torch.py`.
- **Checkpoint architecture inference** exists four times: `agent_common.infer_ppo_arch` / `infer_dqn_arch`, `AgentControl._load_policy`, `mixed_eval._load_ppo_net` / `_load_dqn_net`, and key-prefix sniffing inside `main.py`.
- **Greedy action selection** exists in `DuelingDDQNAgent._greedy_action`, `PPOAgent._greedy_action`, `mixed_eval.select_ppo` / `select_dqn`, and `AgentControl._greedy_action_idx`.
- **Action repeat with jump release** is implemented twice: `agent_common.step_with_action_repeat` (env level) and the frame counter in `AgentControl.get_action` (sim level).
- **Two pipelines for "policy plays Orange"**: env `control_car="both"` (used by `--eval --mix`) and sim-level `AgentControl` (used by `--head_to_head`).
- **Dashboard plotting** is duplicated between `agent_common.save_training_dashboard` and `plot_learning_curves.plot_single_dashboard`.
- **Paths**: `ROOT_DIR` is computed in 8 files and `RUNS_DIR` in 3 (one as an f-string in `main.py`, one hard-coded as `"runs"` in the plotter).
- **GAE**: `ppo_torch.compute_gae` (demo) and `PPOAgent._compute_gae` (training) differ in how they bootstrap.
- `DQN` and `PPO` training loops share the same ~60-line episode skeleton (reset, loop, log, best-model save, periodic checkpoint, dashboard).

### Wrong layering
- `Simulation` (831 lines) is half physics and half observation building: `get_state_norm`, `_flatten_state_norm`, and `get_state_norm_mirror` are ~320 lines, build a large nested dict every step, then flatten it. Private `_include_*_rays` parameters leak an optimisation into the signature.
- `neural_nets/*.py` import the Gym env and config to host a 100+ line demo next to each `nn.Module`. Networks should not depend on the environment.
- `agent_common.step_with_action_repeat` reaches into the env's privates (`_defer_obs`, `_get_obs`); `RocketLeagueEnv.reset` writes `sim._spawn_index`, `sim.goal_scored_this_step`, `sim.last_touch`.
- `sim/core/control.py` imports `Simulation` and `HeuristicBot`, so `core` depends on `ai`. `CarControl` is an if-chain over a string; `AgentControl` and `OrangePerspectiveBot` re-implement the same `get_action()` / `reset()` shape elsewhere.
- Reward constants live in `rocket_league_env.py`, not in config. `agent_common.py` (602 lines) mixes env creation, run directories, checkpoints, the eval loop, logging, plotting, and argparse.

### CLI
- `main.py` (340 lines) parses `sys.argv` by hand with `in sys.argv` checks, repeated `idx + 1` blocks, and undocumented aliases (`-ppo`, `--bots`, `--both`, `--save_gif`, `--fixed_spawn`, three spellings of `--blue_model`). A second, argparse-based parser in `agent_common.build_arg_parser` serves only the agents' `__main__` blocks.
- `--blue` / `--orange` mean a boolean side in `--play` / `--eval` and a model path in `--head_to_head`.
- `_resolve_default_h2h_model` has a hard-coded fallback `runs/20261106_0911/model_ppo.pt`.

### Config misuse
- Agents copy config class attributes into renamed instance fields (`learning_rate_a`, `discount_factor_g`, `fcl_nodes`, `network_sync_rate`), so the documented `dataclasses.replace(DQNConfig(), ...)` variant can never reach an agent.
- `getattr(PPOConfig, "factored_action", False)` and similar appear 6 times with inconsistent defaults (`False` in three, `True` in one) for a field that always exists.
- `GymConfig` is a `@dataclass` with no annotations, so it has no fields; `directions` is a mutable class-level list.
- `ENABLE_ORANGE_CAR`, `ORANGE_IS_BOT`, `BOT_AERIAL_MIN_BOOST` sit in `sim_config.py` under physics constants.

### Dead and stale code
- `sim/core/sim_msgs.py` is empty. `Simulation.orange_is_bot` and the `is_bot` argument are stored and never read. `HeuristicBot.difficulty` is never read. `Ball.update` has no callers.
- `OrangePerspectiveBot` and `_mirrored_sim_view` are only referenced from a commented-out line in `head_to_head.py`.
- `get_state_norm_mirror` reads a `car_to_car` relation that is never produced.
- Compatibility aliases with no callers: `Agent`, `run_network` (x2), `test_ppo_torch`; pass-through methods `_save_gif`, `_rolling_mean`, `_step_with_action_repeat` on both agents.
- `sys.path.insert` hacks in 7 files, needed only because modules are run as scripts.
- Docs: `src/README.md` lists `agent_control.py`, `head_to_head.py`, `mixed_eval.py`, `orange_perspective.py` under `learning_agents/` (they are in `eval/`); `README.md` links to `src/learning_agents/` and `src/neural_nets/` READMEs that do not exist and documents `--gif [out.gif]` where the code takes `--gif_path`; `TECHNICAL_SUMMARY.md` uses `src/core/...` paths and describes a 61-test `tests/` suite that is not in the repository.

### Smaller hazards
- `Simulation.step(..., action_orange: CarAction = CarAction())` is a shared mutable default, and `Car.update` calls `action.clamp()` on it in place.
- `HeuristicBot` hard-codes `1.0 / 60.0` (4 places) instead of `SIM_HZ`; `Car.update` mutates the caller's action.
- Seeding goes through the global `random` / `np.random` in the env, `ReplayMemory`, and `head_to_head`; the sim draws spawns from global `random`.
- Magic numbers: spawn ranges `7.8, 16.8, 18.2, 27.2`, clearance `2.2`, car-car restitution `0.70`, GIF size `(700, 360)`.
- `Renderer._draw_hud` is ~160 lines and `_draw_car` ~100; `HeuristicBot` states are bare strings.
- No tests, no `pyproject.toml`, no linter or formatter config.

## 3. Target layout

```text
src/
├── __main__.py
├── cli.py                    # argparse; main.py stays as a thin shim that calls cli.main()
├── paths.py                  # ROOT_DIR, RUNS_DIR (only definition)
├── config/
│   ├── sim_config.py         # physics + geometry only
│   ├── render_config.py      # colours, PPM, screen size
│   ├── gym_config.py         # env id, episode length, directions (tuple), reward constants
│   └── nn_config.py          # unchanged dataclasses
├── sim/
│   ├── core/
│   │   ├── simulation.py     # space, entities, step, reset, goal detection
│   │   ├── spawns.py         # kickoff rotation + random spawn sampling
│   │   ├── observation.py    # build_observation(sim, ego) -> 52 floats; mirror; layout constants
│   │   ├── actions.py, arena.py, ball.py, car.py
│   ├── control/              # one Controller protocol: get_action(), reset()
│   │   ├── base.py, human.py, scripted.py (heuristic, idle, random), policy.py (was AgentControl)
│   ├── ai/heuristic_bot.py
│   ├── visualization/        # renderer.py, hud.py, car_sprite.py, event_handling.py
│   └── game.py
├── gym_env/
│   ├── action_codec.py       # encode/decode/clear_jump/mirror: only place that knows 9x2x2
│   ├── rewards.py            # reward terms as pure functions of sim state
│   ├── wrappers.py           # ActionRepeat wrapper (replaces _defer_obs poking)
│   └── rocket_league_env.py
├── neural_nets/
│   ├── dueling_ddqn_torch.py # nn.Module only
│   ├── ppo_torch.py          # nn.Module only
│   ├── checkpoint.py         # load_policy(path) -> (net, kind); only architecture inference
│   ├── policy.py             # greedy_action(net, obs) for both net types
│   └── demos.py              # run_dqn_network, run_ppo_network
├── learning_agents/
│   ├── env_factory.py, run_dirs.py, training_logger.py, dashboard.py, eval_loop.py
│   ├── trainer.py            # shared episode loop + hooks
│   ├── experience_replay.py, rollout_buffer.py
│   ├── dueling_ddqn_agent.py, ppo_agent.py   # learning rule only
│   └── plot_learning_curves.py
└── eval/                     # head_to_head.py, mixed_eval.py
tests/
pyproject.toml                # ruff + pytest config
```

The package stays named `src` so `python -m src` keeps working.

## 4. Phases

Each phase is independently mergeable. Phases 0 to 3 give most of the value; 6 and 7 carry the most risk.

### Phase 0: Safety net (do first, no source changes)
1. Build the conda env from `environment.yml`. Nothing could be executed while writing this plan: the machine used had no conda and no `torch`, so every claim here comes from reading the code.
2. Add `pyproject.toml` with `ruff` (lint + format, line length 120 to match current style) and `pytest`. Run the formatter once in its own commit.
3. Record golden data from the current code into `tests/golden/`:
   - Observation trajectory: fixed spawns, seeded, scripted action sequence for both cars, ~600 frames; save Blue obs, mirrored Orange obs, rewards, goal events.
   - A second trajectory with `random_spawn=True` and a fixed seed.
   - Parameter counts and `state_dict` key lists for DQN, Double DQN, Dueling DDQN, PPO flat, PPO factored.
   - One tiny saved checkpoint per architecture.
4. Write tests against the goldens: observation equality (exact), reward equality, checkpoint round-trip, `check_env` passes, action codec round-trip over all 36 indices, mirror applied twice is identity.
5. CLI smoke tests (subprocess, headless): `--play --headless`, `--demo --dqn|--double_dqn|--dueling_ddqn|--ppo`, `--train --ppo --episodes 2`, `--train --dueling_ddqn --episodes 2`, `--eval ... --episodes 1`, `--eval --mix`, `--head_to_head --headless --max_seconds 5`, `--plot`.

Exit: suite green on untouched code. If the sim turns out not to be bit-deterministic, relax the observation test to a tolerance and note it here.

### Phase 1: Dead code and hygiene
- Delete `sim_msgs.py`, `Ball.update`, `orange_is_bot` / `is_bot`, `HeuristicBot.difficulty`, the `car_to_car` branch, the unused aliases, and the agent pass-through methods.
- Delete `OrangePerspectiveBot` / `_mirrored_sim_view` (see decision D1).
- Remove all `sys.path.insert` blocks and per-module `__main__` blocks; everything runs through `python -m src`.
- Fix the mutable default in `Simulation.step`; make `Car.update` clamp a copy.
- Replace `1.0 / 60.0` in the bot with `1.0 / SIM_HZ`; name the magic numbers in section 2 as constants in config.
- Add `src/paths.py` and import `ROOT_DIR` / `RUNS_DIR` from it everywhere.

### Phase 2: Single sources of truth
- `gym_env/action_codec.py`: `decode(action, mirror_x)`, `encode(direction, jump, boost)`, `clear_jump`, `NUM_DIRECTIONS`, `NUM_DISCRETE_ACTIONS`. Replace all five copies.
- `neural_nets/checkpoint.py`: `load_policy(path, device)` infers the architecture from the `state_dict` and returns the built network. Replace the four copies, including the sniffing in `main.py`.
- `neural_nets/policy.py`: one `greedy_action(net, obs)` handling DQN, PPO flat, and PPO factored.
- Move both demos to `neural_nets/demos.py`; network modules then import only `torch`. Delete the demo-only `compute_gae` and call the training one.
- Config: agents take a config instance (`DuelingDDQNAgent(config=DQNConfig())`) and read `self.cfg.<field>` directly; drop the renamed copies and every `getattr(Config, ...)`. Make `GymConfig` a proper frozen dataclass with a tuple of directions; move reward constants into it.

### Phase 3: CLI
- Rewrite as `src/cli.py` with `argparse`: a mutually exclusive mode group (`--play`, `--head_to_head`, `--demo`, `--train`, `--eval`, `--plot`) and per-mode validation, so the documented flag syntax is unchanged and `--help` works.
- Keep the short aliases that appear in help text (`--ddqn`, `--bot-vs-bot`); see decision D2 for the rest.
- Algorithm selection becomes one table: `{"dueling_ddqn": (dueling=True, double=True), "double_dqn": ..., "dqn": ..., "ppo": ...}`, replacing the four near-identical `evaluate_*` call blocks.
- Delete `agent_common.build_arg_parser` and the hard-coded `20261106_0911` fallback.
- Fix the README `--gif` description to match the flags.

### Phase 4: Simulation
- Extract `sim/core/observation.py`: build the flat vector directly from entities for a given ego team, with the mirror as a sign/permutation step and named index constants for the layout. `get_state_norm(as_flat_array=True)` and `get_state_norm_mirror` remain as thin delegates so callers do not change yet. Golden tests must stay exact.
- Keep the structured dict form only if the renderer or docs need it; otherwise drop it with `get_state` consumers reduced to `game.play_headless`.
- Extract `sim/core/spawns.py` (kickoff rotation, `sample_random_spawns`), taking an explicit `random.Random` owned by the sim.
- Give `Simulation.reset` the arguments the env needs (`spawn_index`, `clear_events`) so the env stops writing private attributes.
- Leave `car.py` physics alone apart from splitting the facing logic out of `update` into `_update_facing`. `_set_orange_solver_first` swaps bodies between car objects; do not touch it without a dedicated test.

### Phase 5: Gym environment
- `gym_env/wrappers.py`: `ActionRepeat(env, repeat)` that owns jump release and summed reward and asks the env for one observation at the end. The env exposes a public `step(..., build_obs=False)` or `frame_skip` path; delete `_defer_obs`, `_dummy_obs`, and the `env.unwrapped._get_obs` calls.
- `gym_env/rewards.py`: goal reward, ball progress, approach shaping, penalties as small functions; the env keeps only the previous-step state.
- Seed through `self.np_random` and pass it to the sim instead of seeding global RNGs.
- Move `demo_gym` / `my_check_env` to a demo module.

### Phase 6: Learning agents
- Split `agent_common.py` into the five modules in section 3 (pure moves first).
- `trainer.py`: one `run_training(env, agent, logger, num_episodes)` loop that calls `agent.act(state)`, `agent.observe(transition)`, `agent.end_episode()`, and `agent.metrics()`; DQN and PPO keep only their learning rule, buffers, and dashboard panel definitions.
- Replace `run(is_training=...)` with separate `train()` and `evaluate()`; lower-case the `LOG_FILE`-style attributes into a small `RunPaths` dataclass.
- `ReplayMemory`: own `random.Random(seed)`; stop storing the unused `truncated`.
- Merge the plotter's dashboard with `dashboard.py` so there is one 2x2 figure function.

### Phase 7: Controllers and evaluation
- `sim/control/`: a `Controller` protocol with `HumanController`, `HeuristicController`, `IdleController`, `RandomController`, `PolicyController`. `CarControl(control_type=...)` becomes a factory function for compatibility. This also removes the `core -> ai` import.
- `PolicyController` reuses `checkpoint.load_policy`, `policy.greedy_action`, and `action_codec`.
- Run `--eval --mix` and `--head_to_head` on the same pairing code so "policy on Orange" has one implementation. Verify first that both paths give the same Orange actions on a golden trajectory; if they do not, stop and raise it (decision D3).

### Phase 8: Bot and renderer
- `HeuristicBot`: an `Enum` for states and a dispatch table in `compute_action`; group the per-frame context (`car_d`, `ball_d`, `v_fwd_ball`, ...) into one small dataclass instead of 6 to 8 positional arguments. Add a bot regression test first (fixed seed, same action sequence).
- `Renderer`: split `_draw_hud` and `_draw_car` into `hud.py` and `car_sprite.py`; move colours and screen constants to `render_config.py`.

### Phase 9: Documentation
- Regenerate the tree and module descriptions in `src/README.md` and `README.md`; remove links to READMEs that do not exist or add them.
- Fix `TECHNICAL_SUMMARY.md` paths and replace section 9 with the real test suite and command.
- Re-run every README command as the final check.

## 5. Decisions needed

Defaults are what the plan assumes unless told otherwise.

| ID | Question | Default |
| :-- | :-- | :-- |
| D1 | Delete `OrangePerspectiveBot` (unused, commented out)? | Delete |
| D2 | Keep undocumented CLI aliases (`-ppo`, `--bots`, `--both`, `--save_gif`, `--fixed_spawn`, `--blue_model_path`, `--model_path_blue`, ...)? | Drop those not in README or help text |
| D3 | If the two "policy on Orange" pipelines disagree, which is correct? | Env `control_car="both"` path, since training-time semantics live there |
| D4 | Is a frozen submission copy needed, so the zipped folder is unaffected until the refactor is merged? | Tag the current commit before starting |
| D5 | Should this file and `tests/` live inside `ME5418_Group41_RocketLeague/` (and therefore in the submission zip)? | `tests/` inside, `plan.md` at repo root |

## 6. Risks

- **Silent observation drift** (phase 4) would invalidate trained checkpoints without any error. Mitigation: exact golden comparison for both perspectives, written before the change.
- **RNG stream changes** (phases 4 to 6): moving from global `random` to owned generators changes spawn sequences for a given seed. Physics and learning are unaffected, but seeded runs will not replay identically against old logs. Regenerate the random-spawn golden in that commit and say so in the PR.
- **Trainer abstraction** (phase 6) can hide algorithm differences such as PPO's next-value backfill and truncation bootstrapping. Mitigation: a short seeded training run must produce the same metrics CSV before and after.
- **Solver-order body swap** in `Simulation` is fragile; it is out of scope except for tests.
- **Parallel work**: teammates' branches will conflict with file moves. Land phases 1 to 3 quickly and announce the moves in phases 4, 6, and 7 before starting them.

## 7. Execution status (2026-11-14)

All phases 0 to 9 are done on `major_refactor` (one or more commits per phase, pure moves committed
separately). The starting point is tagged `pre-refactor` (decision D4). Nothing has been pushed or opened as
a PR yet. Decisions D1 to D3 and D5 were taken at their defaults.

**Verification.** `python -m pytest` (118 tests, ~1.5 min; `-m "not slow"` ~10 s) and `ruff check` /
`ruff format --check` pass. The sim is bit-deterministic, so observation goldens are compared exactly
(phase 0 exit criterion needed no relaxation). Every README command was re-run; every checkpoint in `runs/`
loads; evaluations and a seeded random-spawn head-to-head with the trained PPO and Dueling DDQN models print
output identical to `pre-refactor`.

**Goldens that changed, and why** (all other goldens are unchanged since phase 0):
- Phase 4, both seeded training runs: spawn sampling moved off the global `random` module onto the sim's own
  generator. Replaying the old draw order reproduces the old PPO golden exactly.
- Phase 6, the seeded DQN training run: `ReplayMemory` samples from its own generator. Sampling from the global
  stream again reproduces the previous DQN golden exactly. The PPO golden did not change.

**Deviations from the plan as written**
- Phase 1 also removed `agent_common.build_arg_parser` (planned for phase 3): its only callers were the
  per-module `__main__` blocks.
- Phase 5: `reset(seed)` seeds the sim's spawn generator with the reset seed directly instead of deriving it
  from `self.np_random`, which keeps every random-spawn sequence identical. The "don't build the observation"
  path is a `Frame(action, action_orange, build_obs)` step input rather than a `step(..., build_obs=False)`
  keyword, because gym's wrappers (TimeLimit, OrderEnforcing, env checker) only forward the action.
- Phase 6: `AgentConfig.seed` (default None) was added to seed the replay sampler. The dashboard panel
  definitions live in one table (`dashboard.METRIC_PANELS`) used by the trainer and `--plot`, not in the agents.
- Phase 7 (D3): the two Orange pipelines gave identical inputs to both cars for 2,400 frames, for three
  checkpoint pairs; `tests/test_controllers.py` keeps that. They still differ in episode structure, by design:
  eval resets after every goal and at the time limit, head-to-head is one continuous match.
- Phase 8: renderer output was checked pixel-identical on 21 frames, but that check is not in the suite (font
  rendering is machine-dependent).
- Also done: `ENABLE_ORANGE_CAR` / `BOT_AERIAL_MIN_BOOST` left `sim_config.py` (section 2, config misuse);
  `TECHNICAL_SUMMARY.md` needed more than path fixes (wrong observation layout, `get_state()` keys, solver
  iterations, quickstart).

**Intentional behaviour changes** (none affect physics, observations, rewards, checkpoints or run layout)
- CLI: unknown, abbreviated or conflicting flags are errors; the D2 aliases are gone; `--head_to_head` has no
  hard-coded `runs/20261106_0911` fallback.
- `get_state_norm(as_flat_array=False)` and the nested mirror dict are removed (no callers). An Orange view
  with no Orange car raises instead of returning 44 floats.
- `Simulation.reset(spawn_index=...)` no longer forces a fixed kickoff (it had no callers); `env.reset(seed)`
  no longer reseeds global `random` / `np.random`.
- `--plot --dashboard` figures use the same panel titles as the training dashboards.

**Left as is:** the README control tables list gamepad buttons, but only the keyboard is read (pre-existing).

**Update:** the `tests/` suite (with `tests/golden/`), `pyproject.toml` and the `pytest` / `ruff` dependencies
were removed afterwards at the team's request; the verification above was done before their removal. The
control tables in the docs now state that input is keyboard only.
