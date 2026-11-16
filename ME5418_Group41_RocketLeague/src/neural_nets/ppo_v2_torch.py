"""Separate-network PPO actor-critic architecture (independent policy and value MLPs)."""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class SeparateActorCritic(nn.Module):
    """Independent Actor and Critic MLPs with factored MultiDiscrete policy heads.

    Architecture:
        - Actor (pi): ``Linear(state_dim, hidden) -> LayerNorm -> ReLU -> Linear(hidden, hidden)
          -> LayerNorm -> ReLU -> ModuleList`` of per-axis linear heads (default ``[9, 2, 2]``).
        - Critic (V): independent ``Linear(state_dim, hidden) -> LayerNorm -> ReLU
          -> Linear(hidden, hidden) -> LayerNorm -> ReLU -> Linear(hidden, 1)``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: Sequence[int] = (9, 2, 2),
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.is_factored = True
        self.action_dims = list(action_dims)

        def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
            nn.init.orthogonal_(layer.weight, gain=std)
            nn.init.constant_(layer.bias, bias_const)
            return layer

        # Independent Actor MLP
        self.pi_fc1 = layer_init(nn.Linear(state_dim, hidden_dim))
        self.pi_ln1 = nn.LayerNorm(hidden_dim)
        self.pi_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.pi_ln2 = nn.LayerNorm(hidden_dim)
        self.pi_heads = nn.ModuleList([layer_init(nn.Linear(hidden_dim, d), std=0.01) for d in self.action_dims])

        # Independent Critic MLP
        self.vf_fc1 = layer_init(nn.Linear(state_dim, hidden_dim))
        self.vf_ln1 = nn.LayerNorm(hidden_dim)
        self.vf_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.vf_ln2 = nn.LayerNorm(hidden_dim)
        self.vf_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Compute factored policy logits list and scalar state value V(s) of shape (B,)."""
        pi = F.relu(self.pi_ln1(self.pi_fc1(x)))
        pi = F.relu(self.pi_ln2(self.pi_fc2(pi)))
        logits = [head(pi) for head in self.pi_heads]

        vf = F.relu(self.vf_ln1(self.vf_fc1(x)))
        vf = F.relu(self.vf_ln2(self.vf_fc2(vf)))
        V = self.vf_out(vf).squeeze(-1)
        return logits, V

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """Sample (or evaluate a given) factored action, returning (action, log pi, entropy, V(s))."""
        logits, V = self(x)
        dists = [Categorical(logits=l) for l in logits]
        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=-1)
        log_prob = torch.stack([d.log_prob(action[..., i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
        entropy = torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1)
        return action, log_prob, entropy, V
