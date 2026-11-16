# Learning Agents (`src/learning_agents/`)

Training and evaluation of the two agents compared in the report. Both use the same episode loop,
the same 150,949-parameter `LayerNorm` trunk width, `Adam(3e-4)`, `gamma = 0.99` and 4x action repeat
(15 Hz decisions, jump released after the first frame). Hyperparameters are in
[`src/config/nn_config.py`](../config/nn_config.py) (`DQNConfig`, `PPOConfig`); build a variant with
`dataclasses.replace(DQNConfig(), batch_size=128)` and pass it as `config=`.

| File | Description |
| :--- | :--- |
| [`dueling_ddqn_agent.py`](dueling_ddqn_agent.py) | `DuelingDDQNAgent`: epsilon-greedy acting, replay, Double-DQN target, Huber loss, target sync; `train()` / `evaluate()`; `train_dueling_ddqn` / `evaluate_dueling_ddqn` (also plain and Double DQN). |
| [`ppo_agent.py`](ppo_agent.py) | `PPOAgent`: rollouts, truncation-aware GAE (`compute_gae`), clipped surrogate with KL early stop; `train()` / `evaluate()`; `train_ppo` / `evaluate_ppo`. |
| [`trainer.py`](trainer.py) | `run_training`: the shared episode loop. Agents plug in through `act`, `observe`, `progress` and `end_episode`. |
| [`experience_replay.py`](experience_replay.py) | `ReplayMemory`: circular buffer with its own seeded sampler (`AgentConfig.seed`). |
| [`rollout_buffer.py`](rollout_buffer.py) | `RolloutBuffer`: on-policy storage with next-value backfilling. |
| [`env_factory.py`](env_factory.py) | `make_env`: the Gymnasium env with rendering / GIF frames and the `ActionRepeat` wrapper. |
| [`eval_loop.py`](eval_loop.py) | Greedy evaluation episodes (one or both cars), run summary, GIF export. |
| [`training_logger.py`](training_logger.py) | Metrics CSV rows, best-model tracking (100-episode mean return), run log. |
| [`dashboard.py`](dashboard.py) | The 2x2 dashboard (return, goal rate, exploration, loss), used live and by `--plot`. |
| [`run_dirs.py`](run_dirs.py) | `runs/YYYYMMDD_hhmm_<model>/` folders (`RunPaths`), latest-checkpoint lookup, periodic checkpoints. |
| [`plot_learning_curves.py`](plot_learning_curves.py) | `--plot`: multi-run comparison and single-run dashboards from metrics CSVs or logs. |

Each training run writes `model_<name>.pt` (best model), `checkpoints/model_<name>_epNNNNNN.pt`,
`metrics_<name>.csv` (`episode,step,reward,mean_reward_100,goal,goal_rate_100` plus
`epsilon,td_loss` or `entropy,value_loss`), `run_<name>.log` and `graph_<name>.png`.

Policy-vs-policy matches live in [`src/eval/`](../eval/): `mixed_eval.py` (`--eval --mix`, env episodes) and
`head_to_head.py` (`--head_to_head`, a continuous match). Both drive Orange the same way: the
mirrored observation and a mirrored action, so any checkpoint trained on Blue plays either side.
