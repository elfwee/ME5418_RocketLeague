"""
Actor-Critic Architecture
"""

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from src.neural_nets.a2c_torch import ActorCritic


# ---------------------------------------------------------------
# PPO: actor-critic, on-policy, n-step returns
# ---------------------------------------------------------------
class PPO:
    """Proximal Policy Optimization

    Wraps an :class:`ActorCritic` network, an Adam optimiser and an on-policy
    rollout buffer. Training loop::

        while not agent.buffer_full:
            actions, values, log_probs = agent.act(obs)
            next_obs, rewards, dones = env.step(actions)
            agent.store(obs, actions, rewards, dones, values, log_probs)
            obs = next_obs
        _, last_value, _ = agent.act(obs)
        agent.update(last_value)  # GAE -> n_epochs of minibatch updates -> clear buffer

    The buffer holds ``rollout_length`` steps for ``num_envs`` parallel envs.
    Advantages use GAE (Schulman et al., 2016)::

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        R_t     = A_t + V(s_t)

    where ``done_t`` marks that the episode ended *after* taking ``a_t``.

    The loss minimised is::

        loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy

    where ``actor_loss = -mean(min(r * A, clip(r, 1-eps, 1+eps) * A))`` with ``r = pi/pi_old`` and GAE advantages ``A``,
    ``critic_loss = MSE(V(s), R)``, and ``entropy`` is the mean policy entropy.

    Args:
        obs_dim (int): Dimension of the observation vector. Defaults to 52.
        n_actions (int): Number of discrete actions. Defaults to 36.
        hidden_size (int): Hidden layer width of the network. Defaults to 256.
        learning_rate (float): Adam learning rate. Defaults to 7e-4.
        discount_factor (float): Discount factor gamma. Defaults to 0.99.
        gae_lambda (float): GAE lambda; 1.0 gives Monte-Carlo-style n-step
            returns, 0.0 gives one-step TD. Defaults to 0.95.
        clip_range (float): PPO clip epsilon on the probability ratio. Defaults to 0.2.
        value_loss_coef (float): Weight of the critic loss. Defaults to 0.5.
        entropy_coef (float): Weight of the entropy bonus (encourages
            exploration). Defaults to 0.01.
        max_norm (float): Max global gradient norm for clipping. Defaults to 0.5.
        trunk_type (str): ``"independent"`` or ``"shared"``; see
            :class:`ActorCritic`. Defaults to ``"independent"``.
        rollout_length (int): Steps collected per env before an update. Defaults to 16.
        num_envs (int): Number of parallel environments. Defaults to 8.
        n_epochs (int): Passes over the rollout per update. Defaults to 4.
        batch_size (int | None): Minibatch size; ``None`` uses the whole
            rollout as one batch. Defaults to 32.

    Attributes:
        model (ActorCritic): The actor-critic network.
        optimizer (torch.optim.Adam): Optimiser over ``model`` parameters.
        discount_factor (float): Discount factor gamma.
        value_loss_coef (float): Critic loss weight.
        entropy_coef (float): Entropy bonus weight.
        max_norm (float): Gradient clipping threshold.
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
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.max_norm = max_norm
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
        self.obs_dim, self.rollout_length, self.num_envs = obs_dim, rollout_length, num_envs
        self.n_epochs, self.batch_size = n_epochs, batch_size
        self.reset_buffer()

    # ---------------- rollout buffer ----------------
    def reset_buffer(self):
        """Clear the rollout buffer and set the write pointer to 0."""
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
        """bool: True once ``rollout_length`` steps have been stored."""
        return self.buf_pos == self.rollout_length

    def store(self, state, action, reward, done, value, log_prob):
        """Store one vectorised transition; every argument has leading dim ``num_envs``.

        Args:
            state: Observation s_t, shape ``(num_envs, obs_dim)``.
            action: Action a_t, shape ``(num_envs,)``.
            reward: Reward r_t received after a_t, shape ``(num_envs,)``.
            done: 1.0 if the episode ended after a_t, shape ``(num_envs,)``.
            value: Critic estimate V(s_t), shape ``(num_envs,)``.
            log_prob: log pi_old(a_t | s_t) at collection time, shape ``(num_envs,)``.
        """
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
        """Fill the advantage and return buffers with GAE, working backwards in time.

        Args:
            last_value: V(s_T) for the observation following the final stored
                step, shape ``(num_envs,)``. Bootstraps envs whose episode did
                not end on the last step.
        """
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
        """Yield shuffled minibatches over the flattened ``rollout_length * num_envs`` samples.

        Yields:
            tuple[torch.Tensor, ...]: ``(states, actions, returns, old_log_probs,
            advantages)`` in the argument order of :meth:`_backprop`.
        """
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
        """Run a full PPO update on the stored rollout, then clear the buffer.

        Computes GAE advantages and returns, runs ``n_epochs`` passes of
        shuffled minibatch updates via :meth:`_backprop`, and resets the buffer.

        Args:
            last_value: V(s_T) for the observation after the final stored step,
                shape ``(num_envs,)`` (the ``values`` output of :meth:`act`).

        Returns:
            float: Mean loss over all minibatch updates.
        """
        if not self.buffer_full:
            raise RuntimeError(f"Rollout incomplete ({self.buf_pos}/{self.rollout_length} steps).")
        self._compute_returns_and_advantages(last_value)
        losses = [self._backprop(*batch) for _ in range(self.n_epochs) for batch in self._minibatches()]
        self.reset_buffer()
        return sum(losses) / len(losses)

    # ---------------- network ----------------

    def _forwardpass(self, states):
        """Run the network.

        Args:
            states (torch.Tensor): Observations of shape ``(..., obs_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(logits, values)`` as returned
            by :meth:`ActorCritic.forward`.
        """
        return self.model(states)  # (policy logits, V(s))

    @torch.no_grad()
    def act(self, observations):
        """Sample actions from the current policy (no gradients tracked).

        Used during rollout collection and to bootstrap returns from V(s_last).

        Args:
            observations (torch.Tensor): Float tensor of shape
                ``(batch, obs_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - actions: shape ``(batch,)``, int64 actions sampled from
                  ``Categorical(logits)``.
                - values: shape ``(batch,)``, critic estimates V(s).
                - log_probs: shape ``(batch,)``, log pi(a|s) of the sampled
                  actions (stored in the rollout buffer as pi_old).
        """
        logits, values = self._forwardpass(observations)
        dist = Categorical(logits=logits)  # applies softmax i.e. all values are between 0,1 and add up to 1.
        actions = dist.sample()
        return actions, values, dist.log_prob(actions)

    def _backprop(self, states, actions, returns, old_log_probs, advantages):
        """Perform one PPO gradient update on a minibatch (called by :meth:`update`).

        Uses the clipped surrogate objective::

            ratio      = pi(a|s) / pi_old(a|s) = exp(log pi(a|s) - log pi_old(a|s))
            actor_loss = -mean(min(ratio * A, clamp(ratio, 1 - eps, 1 + eps) * A))

        Advantages come from the buffer (GAE, computed once with the network
        that collected the rollout) and are normalised per minibatch. Actor,
        critic and entropy terms are combined into a single loss, the global
        gradient norm is clipped to ``max_norm`` and one optimiser step is taken.

        Args:
            states (torch.Tensor): Observations, shape ``(N, obs_dim)``.
            actions (torch.Tensor): Actions taken in those states, shape ``(N,)``.
            returns (torch.Tensor): Critic targets R = A + V_old(s), shape ``(N,)``.
            old_log_probs (torch.Tensor): log pi_old(a|s) recorded at collection
                time, shape ``(N,)``.
            advantages (torch.Tensor): GAE advantages from the buffer, shape ``(N,)``.

        Returns:
            float: The total scalar loss for this update.
        """
        logits, values = self._forwardpass(states)
        policy_distribution = Categorical(
            logits=logits
        )  # applies softmax i.e. all values are between 0,1 and add up to 1.
        # Normalise advantages per minibatch (keeps the actor gradient scale stable)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_probs = policy_distribution.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)  # pi(a|s) / pi_old(a|s)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)  # PPO
        # Pessimistic bound: take the smaller objective so large policy steps aren't rewarded
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
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
        """Return the greedy (argmax-logit) action for evaluation/deployment.

        Args:
            observation (array-like | torch.Tensor): A single observation of
                shape ``(obs_dim,)``; converted to a float32 tensor.

        Returns:
            int: Index of the highest-scoring action.
        """
        return int(self._forwardpass(torch.as_tensor(observation, dtype=torch.float32))[0].argmax())


# AGENT
def test_ppo_torch(num_updates=300, num_envs=8, rollout_length=16):
    agent = PPO(obs_dim=52, n_actions=36, rollout_length=rollout_length, num_envs=num_envs)
    obs = torch.randn(num_envs, 52)
    for update in range(1, num_updates + 1):
        while not agent.buffer_full:
            actions, values, log_probs = agent.act(obs)
            # Dummy env step: reward always 1.0, random next obs, ~5% chance of episode end
            next_obs = torch.randn(num_envs, 52)
            rewards = torch.ones(num_envs)
            dones = (torch.rand(num_envs) < 0.05).float()
            agent.store(obs, actions, rewards, dones, values, log_probs)
            obs = next_obs
        _, last_value, _ = agent.act(obs)  # bootstrap from V(s_T)
        loss = agent.update(last_value)
        if update % 30 == 0:
            print(f"  update: {update:5d}  loss: {loss:6.1f}")


if __name__ == "__main__":
    test_ppo_torch()
