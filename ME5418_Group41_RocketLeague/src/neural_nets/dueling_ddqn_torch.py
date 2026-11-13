"""Dueling Deep Q-Network (DQN) architecture."""

import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    """Deep Q-Network supporting both Standard and Dueling DQN architectures.

    Architecture:
        - Input Layer: 52-dimensional normalized state vector in [-1, 1].
        - Shared Feature Extractor: 2 fully-connected layers with LayerNorm and ReLU.
        - Dueling Streams (when enable_dueling_dqn=True):
            1. State-Value Stream V(s): estimates scalar value of being in state s.
            2. Action-Advantage Stream A(s, a): estimates relative advantage of each action a.
            Combined via identifiability-stabilized aggregation:
                Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, enable_dueling_dqn: bool = True):
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # Shared feature backbone with LayerNorm to stabilize hidden activations across varying scales
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        if self.enable_dueling_dqn:
            # State-Value stream V(s): R^(hidden_dim) -> R^1
            self.fc_value = nn.Linear(hidden_dim, hidden_dim // 2)
            self.value = nn.Linear(hidden_dim // 2, 1)

            # Action-Advantage stream A(s, a): R^(hidden_dim) -> R^(action_dim)
            self.fc_advantages = nn.Linear(hidden_dim, hidden_dim // 2)
            self.advantages = nn.Linear(hidden_dim // 2, action_dim)
        else:
            # Standard single-stream Q-value head: R^(hidden_dim) -> R^(action_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action-value estimates Q(s, a) for input state tensor x."""
        # Shared feature extraction
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))

        if self.enable_dueling_dqn:
            # Compute state value V(s)
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Compute action advantages A(s, a)
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Combine streams: subtracting mean advantage ensures V(s) and A(s, a) are uniquely identifiable
            Q = V + A - torch.mean(A, dim=-1, keepdim=True)
            return Q
        else:
            return self.fc3(x)
