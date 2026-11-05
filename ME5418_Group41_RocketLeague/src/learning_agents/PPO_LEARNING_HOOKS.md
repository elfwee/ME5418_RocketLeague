# PPO Learning Hooks

Four training aids added to `PPOAgent` (`ppo_agent.py`) on top of a `PPOConfig` retune (`src/config/nn_config.py`).
Each hook is driven by a `PPOConfig` field and can be overridden per hyperparameter set through the YAML loader in `PPOAgent.__init__`.

## Why: the environment's numbers

| Quantity | Value | Consequence |
|---|---|---|
| Decision rate | 60 Hz sim / `action_repeat=4` = **15 Hz** | One decision every ~67 ms |
| Episode length | 3000 frames = **750 decisions** (ends early on a goal) | Long episodes; goals are rare early on |
| Goal reward | **±20** (+2 if orange touched last before a blue goal) | Dominates the return |
| Shaping reward | ~0.01–0.2 per decision (ball progress, car→ball approach, distance/time penalty) | Small next to the goal reward |
| Episode return | roughly **−30 to +40** | Large value targets for the critic |

The actor and critic share a 2×256 `LayerNorm + ReLU` trunk (`ActorCritic` in `src/neural_nets/ppo_torch.py`), and the combined gradient is clipped to `max_grad_norm = 0.5`.

## Config retune

| Field | Old | New | Reason |
|---|---|---|---|
| `discount_factor` | 0.99 | **0.995** | Effective horizon 1/(1−γ): 100 → 200 decisions (~6.7 s → ~13 s), long enough to credit the build-up to a goal |
| `rollout_steps` | 2048 | **4096** | ~5 episodes per update instead of ~2.7, so lower-variance advantages from the single env |
| `minibatch_size` | 64 | **256** | 16 minibatches per epoch, less noisy gradients |
| `update_epochs` | 4 | **6** | More reuse of each rollout; safe because the target-KL stop caps drift |

Unchanged: `learning_rate=3e-4`, `gae_lambda=0.95`, `clip_coef=0.2`, `value_loss_coef=0.5`, `entropy_coef=0.01`, `max_grad_norm=0.5`.

> γ and the minibatch size no longer match `DQNConfig`, so the PPO-vs-Dueling-DDQN comparison is no longer like-for-like on those two settings.

---

## 1. Reward scaling: `reward_scale = 0.1`

**What:** rewards are multiplied by `reward_scale` before they go into the rollout buffer.

```python
buf_rewards.append(torch.tensor(reward * self.reward_scale, ...))   # ppo_agent.py ~L376
```

Only the training signal is scaled. `episode_reward`, the console log, best-model saving and the curriculum `stop_fn` all still see the original ±20 rewards.

**Why:** with returns in the tens, the critic's MSE starts in the hundreds. Because the trunk is shared and the *total* gradient norm is clipped to 0.5, the large value gradient takes up most of the clipping budget and shrinks the policy gradient with it. Scaling by 0.1 brings returns to about ±3, so the two losses are comparable and the policy actually learns from each update.

## 2. Whole-batch advantage normalisation

**What:** advantages are normalised once over the full rollout at the top of `optimize()`, instead of within each minibatch.

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # ppo_agent.py ~L157
```

**Why:** goals are rare, so most minibatches contain only small shaping advantages. With per-minibatch normalisation, a minibatch that happens to contain a goal transition rescales everything else in it, so the same advantage counts for different amounts depending on which random minibatch it lands in. Normalising over the whole rollout keeps the weighting consistent, and a goal stays a strong signal.

## 3. Target-KL early stop: `target_kl = 0.02`

**What:** for each minibatch, `optimize()` estimates the KL divergence between the current policy and the rollout policy:

```python
log_ratio = new_log_probs - mb_old_log_probs
ratio = torch.exp(log_ratio)
approx_kl = ((ratio - 1) - log_ratio).mean()          # ppo_agent.py ~L175
if target_kl is not None and approx_kl > 1.5 * target_kl:
    stop both the minibatch and epoch loops           # before the optimizer step
```

The check runs *before* the optimizer step, so the update that would overshoot is skipped. `optimize()` now returns `(mean_loss, mean_entropy, last_approx_kl)`. Set `target_kl = None` to disable the stop.

**Why:** the clip at 0.2 only zeroes the gradient on samples whose ratio has already left the clip range. It doesn't stop the policy as a whole from drifting over 6 epochs × 16 minibatches. The KL stop bounds that drift. This matters most at curriculum stage switches (none → heuristic bot → self-play), where the data distribution changes suddenly and a large update can collapse the policy. It's also what makes 6 epochs safe.

## 4. Learning-rate annealing: `anneal_lr`, `lr_anneal_steps`, `lr_min_frac`

**What:** before each PPO update, the learning rate is set to:

```python
frac = max(lr_min_frac, 1 - total_steps / lr_anneal_steps)   # ppo_agent.py ~L396
lr   = learning_rate * frac
```

`self.total_steps` counts decisions on the agent itself, like `self.optimizer` and `self.policy_net`. That means it **carries across curriculum stages** instead of restarting each `run()`. With the defaults, the LR falls linearly from 3e-4 over 3M decisions and then stays at 3e-5.

**Why:** smaller steps late in training reduce oscillation once the policy is decent. The curriculum stops each stage when the success rate plateaus, not after a fixed budget, so there's no natural end to anneal towards. A global decision budget with a 10% floor ensures the heuristic-bot and self-play stages can still adapt.

---

## Logging

The training episode line now also shows the KL from the last update and the current LR:

```
Episode   14: Reward =  -4.80 | Mean(100) =  -3.20 | GoalRate(100) =  13.3% | Ent = 3.551 | KL = 0.0116 | LR = 2.99e-04 | Steps = 8832
```

**Healthy signs:**
- KL mostly below 0.03 (the stop threshold).
- Entropy falling slowly from ln 36 ≈ 3.58 without collapsing towards 0.
- GoalRate(100) rising.

**Warning signs:**
- KL often at about 0.03 → updates are being cut short; lower `learning_rate` or `update_epochs`.
- KL near 0 with flat entropy → updates are too timid; raise `learning_rate`.

## Tuning quick reference

| Symptom | Knob |
|---|---|
| Value loss dwarfs policy loss | Lower `reward_scale` (e.g. 0.05) |
| Entropy collapses early / agent stops exploring | Raise `entropy_coef` (0.02) |
| Policy collapses at a curriculum stage switch | Lower `target_kl` (0.01) |
| Learning stalls late in the curriculum | Raise `lr_min_frac` or `lr_anneal_steps` |
| Agent ignores long-range goal credit | Raise `discount_factor` (0.997) |
