"""Network architectures and checkpoints: parameter counts, state_dict keys, loading, greedy actions."""

import json

import numpy as np
import pytest
import scenarios as sc
import torch

ARCH = json.loads((sc.GOLDEN_DIR / "architectures.json").read_text())
GREEDY = json.loads((sc.GOLDEN_DIR / "greedy_actions.json").read_text())


@pytest.mark.parametrize("name", sorted(sc.ARCHITECTURES))
def test_state_dict_layout_matches_golden(name):
    net = sc.ARCHITECTURES[name](256)
    state_dict = net.state_dict()
    assert list(state_dict.keys()) == ARCH[name]["keys"]
    assert {k: list(v.shape) for k, v in state_dict.items()} == ARCH[name]["shapes"]
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == ARCH[name]["params"]


def test_reported_parameter_count():
    # The report compares agents at 150,949 trainable parameters (flat 36-way heads); the factored
    # PPO head (9 + 2 + 2 outputs instead of 36) has 23 x 129 fewer.
    assert ARCH["dueling_dqn"]["params"] == 150_949
    assert ARCH["ppo_flat"]["params"] == 150_949
    assert ARCH["ppo_factored"]["params"] == 150_949 - 23 * 129
    assert ARCH["dqn"]["params"] == 89_636


@pytest.mark.parametrize("name", sorted(sc.ARCHITECTURES))
def test_tiny_checkpoint_loads_and_acts_like_golden(name):
    state_dict = torch.load(sc.GOLDEN_DIR / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=True)
    net = sc.ARCHITECTURES[name](sc.TINY_HIDDEN)
    net.load_state_dict(state_dict)
    net.eval()
    probe = np.load(sc.GOLDEN_DIR / "probe_obs.npy")
    assert [sc.greedy_env_action(net, o) for o in probe] == GREEDY[name]


@pytest.mark.parametrize("name", sorted(sc.ARCHITECTURES))
def test_checkpoint_round_trip(name, tmp_path):
    net = sc.ARCHITECTURES[name](sc.TINY_HIDDEN)
    path = tmp_path / "model.pt"
    torch.save(net.state_dict(), path)
    loaded = sc.ARCHITECTURES[name](sc.TINY_HIDDEN)
    loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    for (k1, v1), (k2, v2) in zip(net.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert k1 == k2 and torch.equal(v1, v2)


@pytest.mark.parametrize("name", sorted(sc.ARCHITECTURES))
def test_load_policy_infers_architecture(name):
    from src.neural_nets.checkpoint import load_policy
    from src.neural_nets.policy import greedy_action

    net, kind = load_policy(sc.GOLDEN_DIR / "checkpoints" / f"{name}.pt")
    assert kind == ("ppo" if name.startswith("ppo") else "dqn")
    assert not net.training
    probe = np.load(sc.GOLDEN_DIR / "probe_obs.npy")
    assert [greedy_action(net, torch.tensor(o)) for o in probe] == GREEDY[name]


def test_infer_spec_matches_builders():
    from src.neural_nets.checkpoint import infer_spec

    expected = {
        "dqn": ("dqn", 36, False),
        "dueling_dqn": ("dqn", 36, True),
        "ppo_flat": ("ppo", 36, False),
        "ppo_factored": ("ppo", [9, 2, 2], False),
    }
    for name, (kind, action_dim, dueling) in expected.items():
        spec = infer_spec(sc.ARCHITECTURES[name](64).state_dict())
        assert (spec.kind, spec.state_dim, spec.hidden_dim, spec.action_dim, spec.dueling) == (
            kind,
            52,
            64,
            action_dim,
            dueling,
        )


def test_load_policy_missing_file(tmp_path):
    from src.neural_nets.checkpoint import load_policy, load_state_dict

    assert load_state_dict(tmp_path / "missing.pt") is None
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "missing.pt")


def test_network_modules_do_not_import_the_env():
    import subprocess
    import sys

    code = (
        "import sys; import src.neural_nets.dueling_ddqn_torch, src.neural_nets.ppo_torch; "
        "bad = [m for m in sys.modules if m.startswith(('gymnasium', 'src.gym_env', 'src.config'))]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=sc.GOLDEN_DIR.parents[1])
