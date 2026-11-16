"""Greedy and stochastic action selection shared by evaluation, head-to-head play, self-play and agents."""

import random

import torch
from torch import nn

from src.neural_nets.ppo_torch import ActorCritic
from src.neural_nets.ppo_v2_torch import SeparateActorCritic


def greedy_action(net: nn.Module, obs: torch.Tensor) -> int | list[int]:
    """Most likely env action for a single observation (shape ``(52,)``).

    - ``DQN``: ``argmax_a Q(s, a)`` as a flat index.
    - ``ActorCritic`` with a flat head: ``argmax`` of the logits as a flat index.
    - ``ActorCritic`` / ``SeparateActorCritic`` with factored heads: ``[direction, jump, boost]``.
    """
    with torch.no_grad():
        out = net(obs.unsqueeze(0))
    if isinstance(net, (ActorCritic, SeparateActorCritic)):
        logits, _ = out
        if net.is_factored:
            return [int(l.squeeze(0).argmax().item()) for l in logits]
        return int(logits.squeeze(0).argmax().item())
    return int(out.squeeze(0).argmax().item())


def sample_action(net: nn.Module, obs: torch.Tensor, epsilon: float = 0.05) -> int | list[int]:
    """Sample an env action for a single observation (shape ``(52,)``).

    - ``ActorCritic`` / ``SeparateActorCritic``: draw from the categorical policy distribution.
    - ``DQN``: ``epsilon``-greedy around ``argmax_a Q(s, a)``.
    """
    if isinstance(net, (ActorCritic, SeparateActorCritic)):
        with torch.no_grad():
            action, _, _, _ = net.get_action_and_value(obs.unsqueeze(0))
        action = action.squeeze(0)
        return action.cpu().numpy().tolist() if net.is_factored else int(action.item())
    with torch.no_grad():
        out = net(obs.unsqueeze(0)).squeeze(0)
    if random.random() < epsilon:
        return random.randrange(int(out.shape[-1]))
    return int(out.argmax().item())
