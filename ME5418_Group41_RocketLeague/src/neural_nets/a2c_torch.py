"""
Actor-Critic Architecture
"""

import sys
from pathlib import Path
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
import copy


class ActorCritic(nn.Module):
    """
    ActorCritic framework
    Attributes:
        obs_dim: observation space dimension
        n_actions: action space dimension
        hidden_size: units within one hidden layer
        trunk_type: independent or shared
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
        logits = self.policy_head(self.trunk(observations))
        if self.value_trunk is None:
            value = self.value_head(self.trunk(observations))
        else:
            value = self.value_head(self.value_trunk(observations))
        return logits, value.squeeze(-1)


# ---------------------------------------------------------------
# A2C: actor-critic, on-policy, n-step returns
# ---------------------------------------------------------------
class A2C:
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
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.max_norm = max_norm

    def _forwardpass(self, states):
        return self.model(states)  # (policy logits, V(s))

    @torch.no_grad()
    def act(self, observations):
        logits, values = self._forwardpass(observations)
        return (
            Categorical(logits=logits).sample(),
            values,
        )  # applies softmax i.e. all values are between 0,1 and add up to 1.

    def _backprop(self, states, actions, returns):
        logits, values = self._forwardpass(states)
        policy_distribution = Categorical(
            logits=logits
        )  # applies softmax i.e. all values are between 0,1 and add up to 1.
        advantages = (returns - values).detach()  # advantage = R - V(s)
        score_function = policy_distribution.log_prob(actions)
        actor_loss = -(score_function * advantages).mean()  # A2C
        critic_loss = F.mse_loss(values, returns)  # mean square error between estimated value and actual value
        entropy_bonus = policy_distribution.entropy().mean()  # exploration
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_bonus
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_norm
        )  # Clip the gradient norm of an iterable of parameters.
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, observation):
        return int(self._forwardpass(torch.as_tensor(observation, dtype=torch.float32))[0].argmax())


# AGENT
def test_a2c_torch(num_updates=3000, num_envs=8, rollout_length=16):
    agent = A2C(obs_dim=52, n_actions=36)
    obs = torch.randn(num_envs, 52)
    for update in range(1, num_updates + 1):
        rollout_states, rollout_actions, rollout_dones = [], [], []
        for _ in range(rollout_length):
            states = obs
            actions, _ = agent.act(states)
            # Dummy env step: random next observation, ~5% chance of episode end
            obs = torch.randn(num_envs, 52)
            dones = (torch.rand(num_envs) < 0.05).float()
            rollout_states.append(states)
            rollout_actions.append(actions)
            rollout_dones.append(dones)
        # Bootstrap from V(s_last), then build n-step discounted returns backwards
        _, discounted_return = agent.act(obs)
        returns = []
        for t in reversed(range(rollout_length)):
            # reward is always 1.0
            discounted_return = 1.0 + agent.discount_factor * (1.0 - rollout_dones[t]) * discounted_return
            returns.append(discounted_return)
        returns.reverse()
        loss = agent._backprop(torch.cat(rollout_states), torch.cat(rollout_actions), torch.cat(returns))
        if update % 300 == 0:
            print(f"  update: {update:5d}  loss: {loss:6.1f}")


if __name__ == "__main__":
    test_a2c_torch()
