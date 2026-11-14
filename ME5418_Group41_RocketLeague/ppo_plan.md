# Plan: PPO-only changes so PPO holds up against an opponent

Scope: `PPOConfig` values and small edits inside `ppo_agent.py`. No multiprocessing, no change to `trainer.py`, the curriculum, the reward or DDQN.

## Context

PPO matches Dueling DDQN with no opponent, then falls behind as soon as there is one (`curriculum.log`, 04:00 runs):

| Stage | PPO goal rate | DDQN goal rate |
| :-- | :-- | :-- |
| s1 none | 77.6% (97% at the end) | 79.5% (99% at the end) |
| s2 heuristic bot | 45.8%, flat for 457k steps | 77.3% |
| s3 self-play r5 | 19.0% | 53.6% |

I measured why on the saved checkpoints. Everything ran in memory; nothing was written to the repo.

**1. The bot is beaten by scoring fast, and stage 2 starts PPO from a slow scorer with a handicap.** Greedy play against the heuristic bot, 60 episodes each (so about +/-6 points). "Full goal" is a +20 goal; "zeroed" means observation indices 35 to 51 (the opponent block) are set to 0, as they are in stage 1.

| Model | Opponent inputs | Full goals | Conceded | Mean return | Steps per episode |
| :-- | :-- | :-- | :-- | :-- | :-- |
| PPO stage-1 best (what stage 2 started from) | seen | 43% | 23% | +3.4 | 277 |
| PPO no-opponent run, final checkpoint | seen | 57% | 22% | +6.9 | 190 |
| PPO no-opponent run, final checkpoint | zeroed | **78%** | **13%** | **+13.6** | 62 |
| DDQN stage-1 best | seen | 63% | 23% | +8.4 | 74 |
| DDQN stage-1 best | zeroed | 83% | 10% | +15.6 | 52 |
| PPO stage-2 best (40 episodes) | seen | 35% | 30% | +0.6 | 240 |
| DDQN stage-2 best (40 episodes) | seen | 72% | 20% | +11.0 | 61 |

Two separate effects are visible:

- **Untrained opponent weights.** With no opponent those 17 inputs are always 0, so their `fc1` columns get no gradient and stay at their random initial values. When the bot appears they inject noise into the trunk. Zeroing them is worth +6.7 mean return on a converged PPO model and +7.2 on DDQN's. The sampled policy, which is what trains, suffers more: -2.9 seen against +4.6 zeroed on the stage-1 best model. A converged PPO stage-1 model with those inputs silent already beats DDQN's stage-2 model.
- **A slow stage-1 model is handed over.** The no-opponent return does not reward speed: late in training, episodes of 60 steps or fewer average 20.77 and episodes over 150 steps average 21.06. So "best 100-episode mean return" picked the model from episode 4,451 (202 steps per goal) over the final one (45 steps per goal). This rule lives in `trainer.py` and the curriculum, outside the chosen scope; see the last section.

**2. Stage 2 then does not recover.** Stage-1 best: 43% full goals. Stage-2 best: 35%. Every update spends the full KL budget (approx. KL 0.015 to 0.02), yet entropy stays flat, so the policy is random-walking. Agreement between policy gradients from independent rollouts of a frozen policy (cosine; 1 is pure signal, 0 pure noise):

| Steps per update | Stage-1 model against the bot | No opponent, mid-training |
| :-- | :-- | :-- |
| 4,096 (now) | 0.065 | 0.02 to 0.05 |
| 8,192 | 0.155 | 0.05 to 0.10 |
| 16,384 | 0.266 | 0.12 to 0.17 |
| 32,768 | 0.483 | not measured |

Updates are just as noisy with no opponent, where PPO learns fine. So noise alone does not explain the stall, and a bigger batch is a hypothesis to test, not a proven fix. My reading: from scratch there is nothing to lose and small noisy steps add up, while a skilled policy on a task where the opponent causes most terminal rewards loses skill to the noise as fast as it gains it.

**3. From scratch against the bot there is no signal at all.** Gradient agreement is about 0 at every batch size, and the model from the 08:58 run scored 1 full goal in 40 greedy episodes. Its "25% goal rate" is the bot touching the ball last on the way in. That run will not learn; training has to go no opponent first, then the bot.

**4. Smaller findings.** The value loss gradient on the shared trunk is 7 to 18 times the policy gradient, and the 0.5 grad-norm clip is active on every minibatch. The critic's explained variance is 0.0 to 0.35 on a fresh rollout and 0.55 to 0.90 on the same rollout after the update. Logged `goal` includes +2 goals, and logged return is for the sampled policy, which against the bot is far below the greedy one.

**Intended outcome:** PPO enters stage 2 at its stage-1 skill level instead of a perturbed copy, and stage 2 ends no worse than it starts. On a converged stage-1 model that is about 78% full goals and 13% conceded, already above DDQN's stage-2 model.

**Unchanged:** network and `state_dict` keys, `lr = 3e-4`, `gamma = 0.99`, reward, observation, metrics CSV columns, CLI, and bit-for-bit the no-opponent stage.

## Approach

### Step 0: three runs, no code change

All three fine-tune `runs/20261114_0303_ppo/checkpoints/model_ppo_ep011000.pt` against the heuristic bot for 20,000 episodes (roughly 1.5 to 3M steps, about an hour each on a free core), started with `train_ppo(config=replace(PPOConfig(), bot_type="heuristic_bot", ...), hyperparameter_set=..., model_path=...)` from a `python -c` one-liner.

| Run | Start weights | Config |
| :-- | :-- | :-- |
| S, status quo | checkpoint as saved | current |
| Z, silenced | copy with `fc1.weight[:, 35:52] = 0`, saved outside the repo | current |
| ZB, silenced and big batch | same copy | `rollout_steps=32768, minibatch_size=2048` |

Then greedy evaluation, 100 episodes, of every fifth periodic checkpoint of each run, giving full goals, conceded and mean return over training.

- Z against S shows whether the head start survives training.
- ZB against Z decides Step 3.

### Step 1: silence untrained opponent inputs (`ppo_agent.py`)

New helper `_silence_opponent_inputs(policy_net)`, called in `train()` right after `_build` when the env has no Orange car (`env.unwrapped.sim.orange_car is None`). It zeroes `fc1.weight[:, OPPONENT_PRESENT]` and `fc1.weight[:, OPPONENT_FEATURES]`, using the index constants that already exist in `src/sim/core/observation.py`, and prints one line saying so.

- The no-opponent run itself is unchanged: those inputs are exactly 0, so the columns contribute nothing, and their gradient is exactly 0, so they stay 0.
- When an opponent appears, the stage-1 policy plays exactly as trained and learns the opponent inputs from zero influence.
- It applies to `--train --ppo` and to curriculum stage 1 without touching the curriculum, because both go through `PPOAgent.train()`.
- Guard: `PPOConfig.silence_opponent_inputs: bool = True`. Turn it off only to fine-tune, without an opponent, a model that has already trained against one; otherwise that knowledge would be erased.

### Step 2: per-update diagnostics (`ppo_agent.py`)

- `optimize` also returns clip fraction, epochs run before the KL stop, pre-clip grad norm, and the value-to-policy gradient ratio on the trunk (first minibatch of each update only).
- `_update_from_rollout` adds explained variance on the fresh rollout (before the update).
- `observe` tallies episode endings in the current rollout from the terminal reward: full goal, +2 goal, conceded, timeout.
- `train()` keeps the run folder; after each update the agent appends one row to `updates_<name>.csv` there and prints a one-line summary.

`metrics_<name>.csv`, the dashboard and `trainer.py` are untouched. The existing helper methods stay as separate steps.

### Step 3: larger updates against an opponent, only if ZB beats Z

- `nn_config.py`: `opponent_batch_scale: int = 8`.
- `ppo_agent.py`: when the env has an opponent, `train()` multiplies `rollout_steps` and `minibatch_size` by it (32,768 and 2,048), and `observe` and `optimize` use the effective values. Stage 1 keeps 4,096 and 256, where small noisy steps are measurably fine and sample-efficient.

Adopt if ZB ends at least 3 mean-return points or 10 points of full-goal rate above Z over 100 greedy episodes and never drops below its starting level. Otherwise leave the batch alone and take the stall to Step 5.

Cost with one env: a 32k-step update takes roughly 30 to 50 seconds to collect, so an opponent stage runs for hours.

### Step 4: balance the two losses on the trunk

`value_loss_coef` 0.5 against 0.05, same setup as the Step 0 winner. 0.05 brings the measured trunk ratio from about 11 to about 1; the critic's own layers are unaffected because Adam rescales them. Adopt 0.05 only if fresh-rollout explained variance does not fall and the greedy result is not worse. This is the least certain change.

### Step 5: only if the diagnostics call for it

- Entropy still above 2.5 late in an opponent stage: `entropy_coef` 0.01 to 0.003.
- Fresh-rollout explained variance below 0.2: `gae_lambda` 0.95 to 0.9 (gradient agreement at 4,096 steps: 0.065 at 0.95, 0.083 at 0.9, 0.105 at 0.8).

### Outside the chosen scope, with what each is worth

1. **Which stage-1 model stage 2 receives.** Largest measured effect: 47% against 78% full goals with the opponent inputs zeroed in both cases. Two or three lines in `Curriculum._train_stage`, to hand over the newest periodic checkpoint instead of the best-by-return file for the no-opponent stage. Until then, `--train --ppo --model_path <newest checkpoint>` does it by hand.
2. **The same opponent-input fix for DDQN.** Its stage-1 model goes from 63% to 83% full goals. Without it PPO gets a fix DDQN does not, which matters for the comparison.
3. **Reward for deflected shots.** Against the bot about 40% of PPO's goals pay +2 instead of +20; in 24 of 27 the ego car had hit the ball within 2 seconds. Decided: leave the reward alone.
4. **Parallel environments,** which would make Step 3's batch about 6 times faster to collect.

## Files

| File | Change |
| :-- | :-- |
| `src/learning_agents/ppo_agent.py` | `_silence_opponent_inputs`, update statistics and `updates_<name>.csv`, effective rollout and minibatch size, docstring numbers |
| `src/config/nn_config.py` | `silence_opponent_inputs`, `opponent_batch_scale`, and `value_loss_coef` if Step 4 passes |

Reused as they are: `OPPONENT_PRESENT` and `OPPONENT_FEATURES` (`src/sim/core/observation.py`), `RunPaths`, `compute_gae`, `RolloutBuffer`, `run_training`.

## Verification

The repo has no test suite (removed on purpose), so checks are scratch scripts and CLI runs, not new repo files.

1. **Stage 1 untouched.** For a batch of no-opponent observations the network output is bit-identical before and after silencing, and after `--train --ppo --episodes 200` with no opponent the 17 columns are still exactly 0.
2. **Shock gone.** A silenced stage-1 model gives the same greedy result against the bot with the opponent inputs seen as with them zeroed.
3. **Diagnostics sane.** `updates_<name>.csv` appears, explained variance lies in [-1, 1], epochs never exceed `update_epochs`, outcome counts add up to the episodes in the rollout.
4. **Learning.** The Step 0 curves, then the same greedy evaluation for Steps 3 and 4. Pass if stage 2 from a converged, silenced stage-1 model ends at or above its start (about 78% full goals, 13% conceded).
5. **Nothing else broke.** `--demo --ppo`, `--train --ppo --episodes 30`, `--curriculum --ppo --max_stage_episodes 50 --self_play_rounds 1`, `--eval --ppo`, `--plot`, and fine-tuning an existing checkpoint with `--model_path`.

Three training runs are active on this machine, one of them the 08:58 PPO run that finding 3 says will not learn. Step 0 adds three single-core runs, so wait for a go-ahead before starting them.
