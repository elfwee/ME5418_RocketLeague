"""
Advantage Actor-Critic (A2C) — Algorithm & Training Loop Comparison Version.
Compare this file directly with `ppo_torch_compare.py` to see the exact
few-line differences that turn A2C into PPO (marked with [DIFF #1] to [DIFF #4]).
"""

import copy
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.config.gym_config import GymConfig


class ActorCritic(nn.Module):
    """MLP actor-critic: a 2-layer Tanh trunk feeding a policy head and a value head."""

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


class A2C:
    """Advantage Actor-Critic (A2C) algorithm for discrete action spaces."""

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
        # === [DIFF #1: Hyperparameters] ===
        # A2C performs 1 gradient update per rollout, so it does NOT need
        # `clip_coef`, `update_epochs`, or `minibatch_size`.
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
        """Sample actions from the current policy during rollout collection."""
        logits, values = self._forwardpass(observations)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        # === [DIFF #2: Rollout Outputs] ===
        # A2C only returns (actions, values) because it updates once immediately
        # after the rollout without comparing against an old policy's log_prob.
        return actions, values

    def _backprop(self, states, actions, returns, advantages=None):
        """Perform one A2C gradient update on a rollout batch."""
        # === [DIFF #3: Single Update vs. Multi-Epoch Minibatch Loop] ===
        # A2C evaluates states once and takes 1 gradient step on the rollout batch.
        logits, values = self._forwardpass(states)
        policy_distribution = Categorical(logits=logits)
        if advantages is None:
            advantages = (returns - values).detach()  # advantage = R - V(s)
        else:
            advantages = advantages.detach()
        score_function = policy_distribution.log_prob(actions)

        # === [DIFF #4: Vanilla Policy Gradient vs. Clipped Surrogate Objective] ===
        # A2C directly weights log pi(a|s) by the advantage:
        actor_loss = -(score_function * advantages).mean()  # A2C

        critic_loss = F.mse_loss(values, returns)
        entropy_bonus = policy_distribution.entropy().mean()
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, observation):
        """Return the greedy (argmax-logit) action."""
        return int(self._forwardpass(torch.as_tensor(observation, dtype=torch.float32))[0].argmax())


def compute_returns_and_advantages(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95, use_gae=True):
    """Phase 1 (Sequential): Compute rollout returns and advantages backwards in time via GAE or n-step returns."""
    T = rewards.shape[0]
    if use_gae:
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(next_value)
        for t in reversed(range(T)):
            next_v = next_value if t == T - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * not_done * next_v - values[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            advantages[t] = last_gae
        returns = advantages + values
    else:
        returns = torch.zeros_like(rewards)
        discounted_return = next_value
        for t in reversed(range(T)):
            discounted_return = rewards[t] + gamma * (1.0 - dones[t]) * discounted_return
            returns[t] = discounted_return
        advantages = returns - values
    return returns, advantages


def train_a2c(total_timesteps=500_000, num_envs=8, rollout_steps=16, bot_type="heuristic_bot", use_gae=True):
    """Train A2C by collecting on-policy rollouts, computing GAE/returns, and running 1 update per rollout."""
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.make(GymConfig.gym_id, render_mode=None, bot_type=bot_type, discrete_action=True)
            for _ in range(num_envs)
        ]
    )
    agent = A2C(obs_dim=envs.single_observation_space.shape[0], n_actions=envs.single_action_space.n)

    obs, _ = envs.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    num_updates = max(1, total_timesteps // (num_envs * rollout_steps))

    for update in range(1, num_updates + 1):
        # === [DIFF #2 (cont.): Rollout Buffer] ===
        # A2C does not need to store `rollout_log_probs`.
        rollout_states, rollout_actions, rollout_rewards, rollout_dones, rollout_values = [], [], [], [], []

        for _ in range(rollout_steps):
            states = obs_t
            actions, values = agent.act(states)
            next_obs, rewards, terminateds, truncateds, _ = envs.step(actions.numpy())
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)

            rollout_states.append(states)
            rollout_actions.append(actions)
            rollout_values.append(values)
            rollout_rewards.append(torch.as_tensor(rewards, dtype=torch.float32))
            rollout_dones.append(torch.as_tensor(dones, dtype=torch.float32))

            obs_t = torch.as_tensor(next_obs, dtype=torch.float32)

        # Phase 1: Bootstrap from V(s_last) and compute returns & advantages sequentially in time
        _, next_value = agent.act(obs_t)
        returns_t, advantages_t = compute_returns_and_advantages(
            torch.stack(rollout_rewards),
            torch.stack(rollout_values),
            torch.stack(rollout_dones),
            next_value,
            gamma=agent.discount_factor,
            use_gae=use_gae,
        )

        # Phase 2: Perform 1 gradient update on the flattened rollout batch
        loss = agent._backprop(
            torch.cat(rollout_states),
            torch.cat(rollout_actions),
            returns_t.reshape(-1),
            advantages=advantages_t.reshape(-1),
        )
        if update % 5 == 0 or update == num_updates:
            print(f"[A2C] Update {update}/{num_updates} | Loss: {loss:.4f}")

    envs.close()
    return agent


if __name__ == "__main__":
    train_a2c(total_timesteps=1_280, num_envs=4, rollout_steps=16)
