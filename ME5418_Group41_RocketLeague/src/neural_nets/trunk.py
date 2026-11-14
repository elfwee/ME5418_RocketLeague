"""The LayerNorm feature trunk used by every network: ``fc1/ln1 -> fc2/ln2`` with ReLU."""

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class Trunk(nn.Module):
    """Two fully-connected layers with LayerNorm and ReLU: R^(state_dim) -> R^(hidden_dim).

    ``init`` (e.g. PPO's orthogonal ``layer_init``) is applied to both Linear layers; None keeps
    PyTorch's default initialisation.
    """

    def __init__(self, state_dim: int, hidden_dim: int, init: Callable[[nn.Linear], nn.Linear] | None = None):
        super().__init__()
        init = init or (lambda layer: layer)
        self.fc1 = init(nn.Linear(state_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        return F.relu(self.ln2(self.fc2(x)))
