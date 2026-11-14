# Neural Networks (`src/neural_nets/`)

PyTorch networks shared by training, evaluation and the demos. The network modules import only
`torch`; their `state_dict` layer names are what every saved checkpoint is loaded by.

| File | Description |
| :--- | :--- |
| [`trunk.py`](trunk.py) | `Trunk`: the `fc1/ln1 -> fc2/ln2` LayerNorm + ReLU feature extractor. |
| [`dueling_ddqn_torch.py`](dueling_ddqn_torch.py) | `DQN`: `value_trunk` / `advantage_trunk` (`split_trunk=True`) or one shared `fc1/ln1 -> fc2/ln2` trunk, then dueling `fc_value -> value` and `fc_advantages -> advantages` streams, or a single `fc3` Q head (plain / Double DQN). |
| [`ppo_torch.py`](ppo_torch.py) | `ActorCritic`: `actor_trunk` / `critic_trunk` (`split_trunk=True`) or one shared trunk, `fc_actor` with one flat `actor` head (36 logits) or factored `actor_heads.N` (9, 2, 2), and `fc_critic -> critic`. |
| [`checkpoint.py`](checkpoint.py) | `load_policy(path)`: rebuilds any of the above from its `state_dict` alone (`infer_spec`, `build_network`). |
| [`policy.py`](policy.py) | `greedy_action(net, obs)`: the most likely env action for every network type. |
| [`demos.py`](demos.py) | `--demo --dqn / --double_dqn / --dueling_ddqn / --ppo`: one forward pass on a real observation and one backward pass. |

Parameter counts at the default width (256), split / shared trunk: Dueling DQN and flat-head PPO
231,333 / 150,949; factored-head PPO 228,366 / 147,982; plain DQN 89,636 (single head, never split).
`checkpoint.infer_spec` detects the trunk layout from the layer names, so shared-trunk checkpoints
still load.
