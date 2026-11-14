"""Dueling Deep Q-Network (DQN) architecture."""

import torch
from torch import nn
from torch.nn import functional as F

from src.neural_nets.trunk import Trunk


class DQN(nn.Module):
    """Deep Q-Network supporting both Standard and Dueling DQN architectures.

    Architecture:
        - Input Layer: 52-dimensional normalized state vector in [-1, 1].
        - Feature Extractor: 2 fully-connected layers with LayerNorm and ReLU. With dueling and
          ``split_trunk=True`` the value and advantage streams each get their own trunk
          (``value_trunk`` / ``advantage_trunk``); otherwise one shared trunk (``fc1/ln1 -> fc2/ln2``)
          feeds the head(s). A single ``fc3`` head has nothing to split, so the flag is ignored there.
        - Dueling Streams (when enable_dueling_dqn=True):
            1. State-Value Stream V(s): estimates scalar value of being in state s.
            2. Action-Advantage Stream A(s, a): estimates relative advantage of each action a.
            Combined via identifiability-stabilized aggregation:
                Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        enable_dueling_dqn: bool = True,
        split_trunk: bool = True,
    ):
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn
        self.split_trunk = split_trunk and enable_dueling_dqn

        # Feature backbone(s) with LayerNorm to stabilize hidden activations across varying scales
        if self.split_trunk:
            self.value_trunk = Trunk(state_dim, hidden_dim)
            self.advantage_trunk = Trunk(state_dim, hidden_dim)
        else:
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
        # Feature extraction: separate value / advantage trunks, or one shared trunk
        if self.split_trunk:
            x_value, x_advantage = self.value_trunk(x), self.advantage_trunk(x)
        else:
            x = F.relu(self.ln1(self.fc1(x)))
            x = x_value = x_advantage = F.relu(self.ln2(self.fc2(x)))

        if self.enable_dueling_dqn:
            # Compute state value V(s)
            v = F.relu(self.fc_value(x_value))
            V = self.value(v)

            # Compute action advantages A(s, a)
            a = F.relu(self.fc_advantages(x_advantage))
            A = self.advantages(a)

            # Combine streams: subtracting mean advantage ensures V(s) and A(s, a) are uniquely identifiable
            Q = V + A - torch.mean(A, dim=-1, keepdim=True)
            return Q
        else:
            return self.fc3(x)
