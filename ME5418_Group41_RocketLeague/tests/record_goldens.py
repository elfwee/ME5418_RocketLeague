"""Record the golden regression data in `tests/golden/` from the current code.

Run once on the code you trust (the goldens in the repository were recorded at tag
`pre-refactor`), and again only when a behaviour change is intended:

    python tests/record_goldens.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: E402,F401  (CPU-only torch, headless pygame, project root on sys.path)
import numpy as np  # noqa: E402
import scenarios as sc  # noqa: E402
import torch  # noqa: E402


def main():
    out = sc.GOLDEN_DIR
    out.mkdir(parents=True, exist_ok=True)

    for name, kwargs in sc.ENV_SCENARIOS.items():
        np.savez_compressed(out / f"{name}.npz", **sc.env_trajectory(**kwargs))
        print(f"recorded {name}")

    np.savez_compressed(out / "goal_events.npz", **sc.goal_events_trajectory())
    print("recorded goal events")

    np.savez_compressed(
        out / "action_repeat_discrete.npz", **sc.action_repeat_trajectory(discrete=True, seed=21, decisions=250)
    )
    np.savez_compressed(
        out / "action_repeat_factored.npz", **sc.action_repeat_trajectory(discrete=False, seed=22, decisions=250)
    )
    np.savez_compressed(out / "action_repeat_both.npz", **sc.action_repeat_both_trajectory(seed=23, decisions=250))
    print("recorded action repeat")

    np.savez_compressed(out / "sim_bots_fixed.npz", **sc.sim_bots_trajectory(frames=1800, random_spawn=False))
    np.savez_compressed(out / "sim_bots_random.npz", **sc.sim_bots_trajectory(frames=1800, random_spawn=True, seed=5))
    print("recorded sim bots")

    # Architectures: full-size parameter counts / state_dict keys, tiny checkpoints + their greedy actions
    arch = {}
    for name, build in sc.ARCHITECTURES.items():
        net = build(256)
        arch[name] = {
            "params": sum(p.numel() for p in net.parameters() if p.requires_grad),
            "keys": list(net.state_dict().keys()),
            "shapes": {k: list(v.shape) for k, v in net.state_dict().items()},
        }
    (out / "architectures.json").write_text(json.dumps(arch, indent=2) + "\n")

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    probe_obs = np.load(out / "env_blue_bot_random.npz")["obs"][::9][:100]
    greedy = {}
    for name, state_dict in sc.build_tiny_checkpoints().items():
        torch.save(state_dict, ckpt_dir / f"{name}.pt")
        net = sc.ARCHITECTURES[name](sc.TINY_HIDDEN)
        net.load_state_dict(state_dict)
        net.eval()
        greedy[name] = [sc.greedy_env_action(net, o) for o in probe_obs]
    np.save(out / "probe_obs.npy", probe_obs)
    (out / "greedy_actions.json").write_text(json.dumps(greedy) + "\n")
    print("recorded architectures and checkpoints")

    for algo in ("dueling_ddqn", "ppo"):
        with tempfile.TemporaryDirectory() as tmp:
            metrics, state_dict = sc.short_training_run(algo, Path(tmp))
        (out / f"train_{algo}_metrics.csv").write_text(metrics)
        torch.save(state_dict, out / f"train_{algo}_model.pt")
        print(f"recorded {algo} training run")


if __name__ == "__main__":
    main()
