"""Separate-network PPO actor-critic architecture (independent policy and value MLPs)."""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class SeparateActorCritic(nn.Module):
    """Independent Actor and Critic MLPs with a flat ``Discrete(n)`` or factored ``MultiDiscrete`` policy.

    Architecture:
        - Actor (pi): ``Linear(state_dim, hidden) -> LayerNorm -> ReLU -> Linear(hidden, hidden)
          -> LayerNorm -> ReLU -> ModuleList`` of linear heads: one head (``action_dims=[36]``, a joint
          softmax over direction x jump x boost, like the DQN) or one per action axis (``[9, 2, 2]``).
          ``forward`` returns a logits tensor for one head and a list of tensors for several.
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
        self.action_dims = list(action_dims)
        self.is_factored = len(self.action_dims) > 1

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]:
        """Policy logits (a tensor for one head, a list for factored heads) and V(s) of shape (B,)."""
        pi = F.relu(self.pi_ln1(self.pi_fc1(x)))
        pi = F.relu(self.pi_ln2(self.pi_fc2(pi)))
        logits = [head(pi) for head in self.pi_heads] if self.is_factored else self.pi_heads[0](pi)

        vf = F.relu(self.vf_ln1(self.vf_fc1(x)))
        vf = F.relu(self.vf_ln2(self.vf_fc2(vf)))
        V = self.vf_out(vf).squeeze(-1)
        return logits, V

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """Sample (or evaluate a given) action, returning (action, log pi, entropy, V(s)).

        Actions are ``(B,)`` indices for one head and ``(B, n_heads)`` for factored heads.
        """
        logits, V = self(x)
        if not self.is_factored:
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), V
        dists = [Categorical(logits=l) for l in logits]
        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=-1)
        log_prob = torch.stack([d.log_prob(action[..., i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
        entropy = torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1)
        return action, log_prob, entropy, V
