"""On-policy rollout storage for PPO."""

import torch

from src.learning_agents.env_factory import device


class RolloutBuffer:
    """On-policy storage for one rollout of decision steps.

    `next_value` is the bootstrap target V(s_{t+1}) for each step. For a step that does not end an
    episode it is left as None and filled in by `backfill_next_value` once the next step's forward
    pass has produced V(s_{t+1}), which saves one critic evaluation per step.
    """

    def __init__(self):
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.next_values: list[torch.Tensor | None] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.states)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor | None,
        reward: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.next_values.append(next_value)
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        self.dones.append(torch.tensor(float(done), dtype=torch.float32, device=device))

    def needs_next_value(self) -> bool:
        """True when the most recent step is still waiting for its bootstrap value."""
        return len(self.next_values) > 0 and self.next_values[-1] is None

    def backfill_next_value(self, value: torch.Tensor):
        """Give the most recent step its V(s_{t+1}), if it is still waiting for one."""
        if self.needs_next_value():
            self.next_values[-1] = value

    def as_tensors(self) -> tuple[torch.Tensor, ...]:
        """(states, actions, log_probs, values, next_values, rewards, dones), each stacked over time."""
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.stack(self.next_values),
            torch.stack(self.rewards),
            torch.stack(self.dones),
        )

    def clear(self):
        for buf in (
            self.states,
            self.actions,
            self.log_probs,
            self.values,
            self.next_values,
            self.rewards,
            self.dones,
        ):
            buf.clear()
