"""Proximal Policy Optimization (PPO) actor-critic architecture."""

import math

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO supporting discrete and factored MultiDiscrete action spaces.

    Architecture:
        - Input Layer: 52-dimensional normalized state vector in [-1, 1].
        - Shared Feature Extractor: 2 fully-connected layers (256 units) with LayerNorm and ReLU.
        - Two Heads:
            1. Actor (Policy) Head:
               - Discrete: Single Linear layer outputting logits over all discrete actions.
               - Factored MultiDiscrete: ModuleList of independent Linear heads for each action axis
                 (e.g., 9 directions, 2 jumps, 2 boosts).
            2. Critic (Value) Head V(s): estimates scalar value of being in state s.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int | list[int] | tuple[int, ...],
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.is_factored = isinstance(action_dim, (list, tuple))
        self.action_dims = list(action_dim) if self.is_factored else [action_dim]

        def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
            nn.init.orthogonal_(layer.weight, gain=std)
            nn.init.constant_(layer.bias, bias_const)
            return layer

        # Shared feature backbone with LayerNorm (hidden: gain = sqrt(2))
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Actor stream (hidden: gain = sqrt(2), policy output: gain = 0.01 for near-uniform exploration)
        self.fc_actor = layer_init(nn.Linear(hidden_dim, hidden_dim // 2))
        if self.is_factored:
            self.actor_heads = nn.ModuleList(
                [layer_init(nn.Linear(hidden_dim // 2, d), std=0.01) for d in self.action_dims]
            )
        else:
            self.actor = layer_init(nn.Linear(hidden_dim // 2, self.action_dims[0]), std=0.01)

        # Critic stream (hidden: gain = sqrt(2), value output: gain = 1.0)
        self.fc_critic = layer_init(nn.Linear(hidden_dim, hidden_dim // 2))
        self.critic = layer_init(nn.Linear(hidden_dim // 2, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]:
        """Compute action logits and state value V(s) for input state tensor x."""
        # Shared feature extraction
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))

        # Compute action logits for pi(a | s)
        a = F.relu(self.fc_actor(x))
        if self.is_factored:
            logits = [head(a) for head in self.actor_heads]
        else:
            logits = self.actor(a)

        # Compute state value V(s)
        v = F.relu(self.fc_critic(x))
        V = self.critic(v).squeeze(-1)
        return logits, V

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """Sample (or evaluate a given) action, returning (action, log pi(a | s), entropy, V(s))."""
        logits, V = self(x)
        if self.is_factored:
            dists = [Categorical(logits=l) for l in logits]
            if action is None:
                action = torch.stack([d.sample() for d in dists], dim=-1)
            # Sum log probabilities and entropies across independent action dimensions
            log_prob = torch.stack([d.log_prob(action[..., i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
            entropy = torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1)
            return action, log_prob, entropy, V
        else:
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), V
