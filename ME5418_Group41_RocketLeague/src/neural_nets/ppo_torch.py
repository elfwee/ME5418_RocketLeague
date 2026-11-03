"""PPO agent built on the shared ActorCritic network."""

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from src.neural_nets.a2c_torch import ActorCritic


class PPO:
    """Proximal Policy Optimization with GAE and an on-policy rollout buffer.

    Usage::

        while not agent.buffer_full:
            actions, values, log_probs = agent.act(obs)
            next_obs, rewards, dones = env.step(actions)
            agent.store(obs, actions, rewards, dones, values, log_probs)
            obs = next_obs
        _, last_value, _ = agent.act(obs)
        agent.update(last_value)

    Loss: ``actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy``.

    Args:
        obs_dim: Observation size.
        n_actions: Number of discrete actions.
        hidden_size: Hidden layer width.
        learning_rate: Adam learning rate.
        discount_factor: Gamma.
        clip_range: PPO ratio clip epsilon.
        value_loss_coef: Critic loss weight.
        entropy_coef: Entropy bonus weight.
        max_norm: Gradient norm clip.
        trunk_type: ``"independent"`` or ``"shared"`` (see ActorCritic).
        gae_lambda: GAE lambda (1.0 = Monte Carlo, 0.0 = one-step TD).
        rollout_length: Steps per env before each update.
        num_envs: Number of parallel envs.
        n_epochs: Passes over the rollout per update.
        batch_size: Minibatch size; ``None`` uses the full rollout.
    """

    def __init__(
        self,
        obs_dim=52,
        n_actions=36,
        hidden_size=256,
        learning_rate=7e-4,
        discount_factor=0.99,
        clip_range=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_norm=0.5,
        trunk_type="independent",
        gae_lambda=0.95,
        rollout_length=16,
        num_envs=8,
        n_epochs=4,
        batch_size=32,
    ):
        self.model = ActorCritic(obs_dim, n_actions, hidden_size, trunk_type)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_norm = max_norm
        self.obs_dim = obs_dim
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reset_buffer()

    # ---------------- rollout buffer ----------------
    def reset_buffer(self):
        """Clear the rollout buffer."""
        shape = (self.rollout_length, self.num_envs)
        self.buf_states = torch.zeros(*shape, self.obs_dim)
        self.buf_actions = torch.zeros(shape, dtype=torch.long)
        self.buf_rewards = torch.zeros(shape)
        self.buf_dones = torch.zeros(shape)
        self.buf_values = torch.zeros(shape)
        self.buf_log_probs = torch.zeros(shape)
        self.buf_advantages = torch.zeros(shape)
        self.buf_returns = torch.zeros(shape)
        self.buf_pos = 0

    @property
    def buffer_full(self):
        """True once ``rollout_length`` steps are stored."""
        return self.buf_pos == self.rollout_length

    def store(self, state, action, reward, done, value, log_prob):
        """Store one step for all envs (each arg has leading dim ``num_envs``)."""
        if self.buffer_full:
            raise RuntimeError("Rollout buffer is full; call update() before storing more.")
        t = self.buf_pos
        self.buf_states[t] = torch.as_tensor(state, dtype=torch.float32)
        self.buf_actions[t] = torch.as_tensor(action, dtype=torch.long)
        self.buf_rewards[t] = torch.as_tensor(reward, dtype=torch.float32)
        self.buf_dones[t] = torch.as_tensor(done, dtype=torch.float32)
        self.buf_values[t] = torch.as_tensor(value, dtype=torch.float32)
        self.buf_log_probs[t] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.buf_pos += 1

    @torch.no_grad()
    def _compute_returns_and_advantages(self, last_value):
        """Compute GAE advantages and returns, bootstrapping from ``last_value``."""
        next_value = torch.as_tensor(last_value, dtype=torch.float32)
        gae = torch.zeros(self.num_envs)
        for t in reversed(range(self.rollout_length)):
            not_done = 1.0 - self.buf_dones[t]
            delta = self.buf_rewards[t] + self.discount_factor * next_value * not_done - self.buf_values[t]
            gae = delta + self.discount_factor * self.gae_lambda * not_done * gae
            self.buf_advantages[t] = gae
            next_value = self.buf_values[t]
        self.buf_returns = self.buf_advantages + self.buf_values

    def _minibatches(self):
        """Yield shuffled ``(states, actions, returns, old_log_probs, advantages)`` minibatches."""
        n = self.rollout_length * self.num_envs
        flat = (
            self.buf_states.reshape(n, self.obs_dim),
            self.buf_actions.reshape(n),
            self.buf_returns.reshape(n),
            self.buf_log_probs.reshape(n),
            self.buf_advantages.reshape(n),
        )
        batch_size = self.batch_size or n
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield tuple(x[idx] for x in flat)

    def update(self, last_value):
        """Run ``n_epochs`` of minibatch updates on the rollout, then clear it.

        Returns the mean loss.
        """
        if not self.buffer_full:
            raise RuntimeError(f"Rollout incomplete ({self.buf_pos}/{self.rollout_length} steps).")
        self._compute_returns_and_advantages(last_value)
        losses = [self._backprop(*batch) for _ in range(self.n_epochs) for batch in self._minibatches()]
        self.reset_buffer()
        return sum(losses) / len(losses)

    # ---------------- network ----------------
    def _forwardpass(self, states):
        """Run the network; returns ``(logits, values)``."""
        return self.model(states)

    @torch.no_grad()
    def act(self, observations):
        """Sample actions; returns ``(actions, values, log_probs)``."""
        logits, values = self._forwardpass(observations)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return actions, values, dist.log_prob(actions)

    def _backprop(self, states, actions, returns, old_log_probs, advantages):
        """One clipped-surrogate PPO gradient step on a minibatch. Returns the loss."""
        logits, values = self._forwardpass(states)
        dist = Categorical(logits=logits)

        # Normalise advantages per minibatch
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(dist.log_prob(actions) - old_log_probs)  # pi / pi_old
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
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


def test_ppo_torch(num_updates=300, num_envs=8, rollout_length=16):
    """Smoke test on a dummy env (constant reward, random obs, ~5% episode end)."""
    agent = PPO(obs_dim=52, n_actions=36, rollout_length=rollout_length, num_envs=num_envs)
    obs = torch.randn(num_envs, 52)
    for update in range(1, num_updates + 1):
        while not agent.buffer_full:
            actions, values, log_probs = agent.act(obs)
            next_obs = torch.randn(num_envs, 52)
            rewards = torch.ones(num_envs)
            dones = (torch.rand(num_envs) < 0.05).float()
            agent.store(obs, actions, rewards, dones, values, log_probs)
            obs = next_obs
        _, last_value, _ = agent.act(obs)
        loss = agent.update(last_value)
        if update % 30 == 0:
            print(f"  update: {update:5d}  loss: {loss:6.1f}")


if __name__ == "__main__":
    test_ppo_torch()
