"""Actor-critic network and A2C agent."""

import copy
import os

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class ActorCritic(nn.Module):
    """MLP actor-critic: a 2-layer Tanh trunk feeding a policy head and a value head.

    Args:
        obs_dim: Observation size.
        n_actions: Number of discrete actions.
        hidden_size: Hidden layer width.
        trunk_type: ``"independent"`` gives the critic its own copy of the trunk;
            ``"shared"`` makes actor and critic share one trunk.
    """

    def __init__(self, obs_dim, n_actions, hidden_size=64, trunk_type="independent"):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        if trunk_type == "independent":
            self.value_trunk = copy.deepcopy(self.trunk)
        elif trunk_type == "shared":
            self.value_trunk = None
        self.policy_head = nn.Linear(hidden_size, n_actions)  # Actor
        self.value_head = nn.Linear(hidden_size, 1)  # Critic

    def forward(self, observations):
        """Return ``(logits, values)`` for a batch of observations."""
        logits = self.policy_head(self.trunk(observations))
        if self.value_trunk is None:
            value = self.value_head(self.trunk(observations))
        else:
            value = self.value_head(self.value_trunk(observations))
        return logits, value.squeeze(-1)


class A2C:
    """Advantage Actor-Critic for discrete actions.

    The caller collects a rollout with :meth:`act`, computes n-step returns
    (bootstrapped from V(s_last)) and passes them to :meth:`_backprop`.

    Loss: ``actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy``.

    Args:
        obs_dim: Observation size.
        n_actions: Number of discrete actions.
        hidden_size: Hidden layer width.
        learning_rate: Adam learning rate.
        discount_factor: Gamma (used by the caller to compute returns).
        value_loss_coef: Critic loss weight.
        entropy_coef: Entropy bonus weight.
        max_norm: Gradient norm clip.
        trunk_type: ``"independent"`` or ``"shared"`` (see ActorCritic).
    """

    def __init__(
        self,
        obs_dim=52,
        n_actions=36,
        hidden_size=256,
        learning_rate=7e-4,
        discount_factor=0.99,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_norm=0.5,
        trunk_type="independent",
    ):
        self.model = ActorCritic(obs_dim, n_actions, hidden_size, trunk_type)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_norm = max_norm

    def _forwardpass(self, states):
        """Run the network; returns ``(logits, values)``."""
        return self.model(states)

    @torch.no_grad()
    def act(self, observations):
        """Sample actions; returns ``(actions, values)``."""
        logits, values = self._forwardpass(observations)
        return Categorical(logits=logits).sample(), values

    def _backprop(self, states, actions, returns):
        """One A2C gradient step on a rollout batch. Returns the loss."""
        logits, values = self._forwardpass(states)
        dist = Categorical(logits=logits)

        advantages = (returns - values).detach()  # A = R - V(s)
        actor_loss = -(dist.log_prob(actions) * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, observation):
        """Return the greedy action for a single observation."""
        logits, _ = self._forwardpass(torch.as_tensor(observation, dtype=torch.float32))
        return int(logits.argmax())

    def save(self, path):
        """Save network and optimizer state to ``path``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path):
        """Load network and optimizer state saved by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def test_a2c_torch(num_updates=3000, num_envs=8, rollout_length=16):
    """Smoke test on a dummy env (constant reward, random obs, ~5% episode end)."""
    agent = A2C(obs_dim=52, n_actions=36)
    obs = torch.randn(num_envs, 52)
    for update in range(1, num_updates + 1):
        rollout_states, rollout_actions, rollout_dones = [], [], []
        for _ in range(rollout_length):
            states = obs
            actions, _ = agent.act(states)
            obs = torch.randn(num_envs, 52)
            dones = (torch.rand(num_envs) < 0.05).float()
            rollout_states.append(states)
            rollout_actions.append(actions)
            rollout_dones.append(dones)
        # Bootstrap from V(s_last), then build n-step returns backwards (reward = 1.0)
        _, discounted_return = agent.act(obs)
        returns = []
        for t in reversed(range(rollout_length)):
            discounted_return = 1.0 + agent.discount_factor * (1.0 - rollout_dones[t]) * discounted_return
            returns.append(discounted_return)
        returns.reverse()
        loss = agent._backprop(torch.cat(rollout_states), torch.cat(rollout_actions), torch.cat(returns))
        if update % 300 == 0:
            print(f"  update: {update:5d}  loss: {loss:6.1f}")


if __name__ == "__main__":
    test_a2c_torch()
