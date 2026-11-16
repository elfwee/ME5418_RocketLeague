"""Greedy action selection shared by evaluation, head-to-head play and the agents."""

import torch
from torch import nn

from src.neural_nets.ppo_torch import ActorCritic


def greedy_action(net: nn.Module, obs: torch.Tensor) -> int | list[int]:
    """Most likely env action for a single observation (shape ``(52,)``).

    - ``DQN``: ``argmax_a Q(s, a)`` as a flat index.
    - ``ActorCritic`` with a flat head: ``argmax`` of the logits as a flat index.
    - ``ActorCritic`` with factored heads: ``[direction, jump, boost]``, one argmax per head.
    """
    with torch.no_grad():
        out = net(obs.unsqueeze(0))
    if isinstance(net, ActorCritic):
        logits, _ = out
        if net.is_factored:
            return [int(l.squeeze(0).argmax().item()) for l in logits]
        return int(logits.squeeze(0).argmax().item())
    return int(out.squeeze(0).argmax().item())
