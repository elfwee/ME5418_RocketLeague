I didn't change any code. My conclusion is that the PPO algorithm isn't what's broken; this setup leaves PPO only a small fraction of D3QN's learning per environment step, and the self-play loop never makes it adapt to D3QN. Everything below comes from headless matches (400–800 per pairing, same rules as your round robin) and one PPO update run from your best checkpoint. Nothing was trained.

## Two explanations the data rules out

- **Greedy (argmax) evaluation of a stochastic policy is not the problem.** Sampled PPO does as badly against D3QN as greedy PPO: 95 wins out of 400 either way.
- **"PPO dodges too much" isn't quite right.** D3QN actually dodges more: about 46 per minute against PPO's 37. The difference is where they dodge. PPO aims 56% of its dodges at the ball and only 7% away from it; D3QN aims 37% at the ball and 28% away, so it also dodges to get back into position. PPO is a ball-chaser. D3QN gets the first touch 72% of the time, and even when PPO touches first, D3QN still wins 70%.

PPO is not generally weak, though. Against the heuristic bot it wins 86% of matches, which matches D3QN's 83%. It loses specifically to the D3QNs that came out of self-play.

## Likely causes, most important first

**1. The critic uses up PPO's update budget.** This is specific to your shared trunk. I ran one real update from `j_model_ppo_s3_self_r2` against the best D3QN:
- With the full loss, the KL early stop (`target_kl = 0.02`) fires after 1–2 epochs.
- An update using **only the value loss**, with no policy gradient at all, moves the policy by KL 0.0235 in one epoch. That's more than the policy gradient itself moves it (0.0186), and enough alone to trigger the stop.
- The cause is that Adam takes similar-sized steps in the shared trunk whatever the gradient size, so the critic's updates keep shifting the actor's features. A large part of each trust region goes on that noise rather than on improving the policy.
- Industry practice: Stable-Baselines3 and the PPO bots built on RLGym (the Rocket League RL toolkit) use separate policy and value networks. CleanRL and Stable-Baselines3 both default to no KL early stop (`target_kl = None`).

**2. PPO gets far less optimisation per env step than D3QN.** In this run, PPO took 32–48 gradient steps per 4,096 decisions; D3QN takes 1,024 (one every 4 decisions, batch 64). That's roughly 20–30× fewer updates, and each sample is reused 1–2 times instead of about 16. The training logs show the effect:

| Run | Goal rate over the run | Decisions per episode |
| :--- | :--- | :--- |
| D3QN self-play r1 | 34% → 68% | 102 → 73 |
| D3QN self-play r2 | 33% → 64% | 118 → 79 |
| PPO self-play r1 | 61% → 74% | about 115-120 throughout |
| PPO self-play r2 | 62% → 64%, falling to 43% at the end | about 115-120 throughout |
| Against heuristic bot, to reach 86% | PPO 2.9M steps, D3QN 1.0M steps | |

Industry PPO makes up for this with massive parallel simulation: RLGym bots collect tens of thousands of steps per update and train for billions of steps. With one environment and a similar step budget, you should expect D3QN to win; that's a valid finding for your report. Your earlier PPO v2 had 8 environments but kept the 0.02 KL stop and 6 epochs, and stages still end on an episode-count plateau. So the parallel environments made training faster in wall-clock time but probably didn't give PPO many more samples. I couldn't check this because those runs are gone.

**3. Your best-model selection silently threw away PPO's self-play round 2.**
- Its "best" model was saved at episode 21 (step 2,298), before the first PPO update at step 4,096, on a lucky 20-episode average that later training never beat.
- So the curriculum's final PPO model is round 1's model; I confirmed the weights are identical.
- The `j_model_ppo_s3_self_r2` in your tournament is the hand-picked `ep003000` checkpoint.
- This happens because the best model is chosen by the noisy training-return average from 20 episodes onward, instead of by evaluation matches against fixed opponents.

**4. PPO never played against D3QN during training.** It only ever met a single frozen copy of itself, and that copy was also a ball-chaser, so over-committing was never punished. OpenAI Five and AlphaStar train against pools of past versions and other agents. Your pool attempt in v2 was still all PPO plus the heuristic bot.

One more thing to keep in mind: tournament matches last about 5 seconds on average, so the ranking mostly measures the kickoff and the first exchange.

## What I'd change, in order

1. **Stop the critic from moving the policy.** Use separate actor and critic networks, or detach the critic from the trunk. Also loosen the KL stop: `target_kl = None`, or about 0.05 with 2–4 epochs. The update diagnostic is a quick smoke test of whether this works.
2. **Give PPO more data.** Use parallel environments and budget training by env steps rather than the episode plateau. Otherwise, compare the two at matched gradient steps.
3. **Fix model selection.** Pick the best model from evaluation matches against a fixed pool (the heuristic bot, the best D3QN, and the previous PPO), and allow no "best" before the first update.
4. **Add D3QN checkpoints to PPO's self-play pool.** For a fair comparison, give D3QN the same pool.

I can add 1 and 3 as opt-in config options and smoke-test them, leaving the D3QN path unchanged.

The diagnostic scripts (`diag_matches.py`, `diag_kickoff.py`, `diag_update.py`) are in `/home/jensen/.claude/jobs/a93bed67/tmp/`, which is deleted along with this job. I can move them into the repo if you want to re-run them. I can also turn these findings into a shareable page for your group or report.