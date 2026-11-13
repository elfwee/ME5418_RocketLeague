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
