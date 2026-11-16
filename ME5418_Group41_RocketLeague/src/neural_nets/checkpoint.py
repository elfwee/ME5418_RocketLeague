"""Checkpoint loading: the only place that infers a network's architecture from its state_dict.

Every saved model is a plain ``state_dict``. Layer names tell the network type apart:

- PPO ``ActorCritic``: ``fc_actor`` / ``fc_critic`` streams, and either one flat ``actor`` head or
  factored ``actor_heads.N`` heads (one per action axis).
- ``DQN``: dueling ``fc_value`` / ``advantages`` streams, or a single ``fc3`` Q head.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.neural_nets.dueling_ddqn_torch import DQN
from src.neural_nets.ppo_torch import ActorCritic

PPO = "ppo"
DQN_KIND = "dqn"
_PPO_PREFIXES = ("actor", "actor_heads", "critic", "fc_actor", "fc_critic")


@dataclass(frozen=True)
class PolicySpec:
    """Everything needed to rebuild a network: its kind and layer sizes.

    ``action_dim`` is an int for a flat head, a list for factored PPO heads, or None when a PPO
    checkpoint has no actor layer (the caller then takes it from the env's action space).
    """

    kind: str
    state_dim: int
    hidden_dim: int
    action_dim: int | list[int] | None
    dueling: bool = False

    @property
    def factored(self) -> bool:
        return isinstance(self.action_dim, list)


def is_ppo_state_dict(state_dict: dict) -> bool:
    return any(key.split(".")[0] in _PPO_PREFIXES for key in state_dict)


def infer_spec(state_dict: dict) -> PolicySpec:
    """Architecture of a saved ``ActorCritic`` or ``DQN`` from its layer names and shapes."""
    hidden_dim, state_dim = state_dict["fc1.weight"].shape
    if is_ppo_state_dict(state_dict):
        if "actor_heads.0.weight" in state_dict:
            n_heads = len({k.split(".")[1] for k in state_dict if k.startswith("actor_heads.")})
            action_dim = [state_dict[f"actor_heads.{i}.weight"].shape[0] for i in range(n_heads)]
        elif "actor.weight" in state_dict:
            action_dim = state_dict["actor.weight"].shape[0]
        else:
            action_dim = None
        return PolicySpec(PPO, state_dim, hidden_dim, action_dim)
    dueling = "advantages.weight" in state_dict
    out_key = "advantages.weight" if dueling else "fc3.weight"
    return PolicySpec(DQN_KIND, state_dim, hidden_dim, state_dict[out_key].shape[0], dueling=dueling)


def build_network(spec: PolicySpec) -> nn.Module:
    """A freshly initialised network with the given architecture."""
    if spec.kind == PPO:
        return ActorCritic(spec.state_dim, spec.action_dim, spec.hidden_dim)
    return DQN(spec.state_dim, spec.action_dim, spec.hidden_dim, enable_dueling_dqn=spec.dueling)


def load_state_dict(path: str | Path, device: str | torch.device = "cpu") -> dict | None:
    """A ``.pt`` state_dict mapped onto ``device``, or None when the file does not exist."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=True)


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, str]:
    """Rebuild and load the network saved at ``path`` in eval mode; returns ``(network, kind)``.

    ``kind`` is ``"ppo"`` or ``"dqn"``. Raises ``FileNotFoundError`` when the file is missing.
    """
    state_dict = load_state_dict(path, device)
    if state_dict is None:
        raise FileNotFoundError(f"No checkpoint at '{path}'")
    spec = infer_spec(state_dict)
    net = build_network(spec).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net, spec.kind
