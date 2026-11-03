"""
Actor-Critic Architecture
"""

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
import copy


class ActorCritic(nn.Module):
    """Actor-critic network with a policy (actor) head and a value (critic) head.

    The trunk is a 2-layer MLP (Linear -> Tanh -> Linear -> Tanh). The actor
    head maps trunk features to unnormalised action logits; the critic head maps
    trunk features to a scalar state-value estimate V(s).

    Args:
        obs_dim (int): Dimension of the (flat) observation vector.
        n_actions (int): Number of discrete actions.
        hidden_size (int): Number of units in each hidden layer. Defaults to 64.
        trunk_type (str): How the actor and critic share features.
            ``"independent"``: the critic gets its own copy of the trunk
            (separate weights, initialised identically to the actor trunk).
            ``"shared"``: actor and critic use the same trunk, so critic
            gradients also update the actor's features.
            Defaults to ``"independent"``.

    Attributes:
        trunk (nn.Sequential): Feature extractor used by the actor (and by the
            critic when ``trunk_type == "shared"``).
        value_trunk (nn.Sequential | None): Separate critic feature extractor,
            or ``None`` when the trunk is shared.
        policy_head (nn.Linear): Maps features to ``n_actions`` logits.
        value_head (nn.Linear): Maps features to a single value estimate.
    """

    def __init__(self, obs_dim, n_actions, hidden_size=64, trunk_type="independent"):
        """Build the trunk(s) and the actor/critic heads. See class docstring for args."""
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
        """Compute policy logits and state values for a batch of observations.

        Args:
            observations (torch.Tensor): Float tensor of shape ``(..., obs_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - logits: shape ``(..., n_actions)``, unnormalised action scores
                  (pass to ``Categorical(logits=...)`` to get a distribution).
                - value: shape ``(...)``, estimated V(s) with the trailing
                  singleton dimension squeezed out.
        """
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
    """Advantage Actor-Critic (A2C) agent for discrete action spaces.

    Wraps an :class:`ActorCritic` network and an Adam optimiser. Training is
    on-policy: the caller collects a rollout with :meth:`act`, computes n-step
    discounted returns (bootstrapped from V(s_last)), and passes them to
    :meth:`_backprop` for a single gradient update.

    The loss minimised is::

        loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy

    where ``actor_loss = -mean(log pi(a|s) * A)``, ``A = R - V(s)`` (detached),
    ``critic_loss = MSE(V(s), R)``, and ``entropy`` is the mean policy entropy.

    Args:
        obs_dim (int): Dimension of the observation vector. Defaults to 52.
        n_actions (int): Number of discrete actions. Defaults to 36.
        hidden_size (int): Hidden layer width of the network. Defaults to 256.
        learning_rate (float): Adam learning rate. Defaults to 7e-4.
        discount_factor (float): Discount factor gamma used when computing
            returns. Stored for the caller's use; not applied inside this
            class. Defaults to 0.99.
        value_loss_coef (float): Weight of the critic loss. Defaults to 0.5.
        entropy_coef (float): Weight of the entropy bonus (encourages
            exploration). Defaults to 0.01.
        max_norm (float): Max global gradient norm for clipping. Defaults to 0.5.
        trunk_type (str): ``"independent"`` or ``"shared"``; see
            :class:`ActorCritic`. Defaults to ``"independent"``.

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
            tuple[torch.Tensor, torch.Tensor]:
                - actions: shape ``(batch,)``, int64 actions sampled from
                  ``Categorical(logits)``.
                - values: shape ``(batch,)``, critic estimates V(s).
        """
        logits, values = self._forwardpass(observations)
        return (
            Categorical(logits=logits).sample(),
            values,
        )  # applies softmax i.e. all values are between 0,1 and add up to 1.

    def _backprop(self, states, actions, returns):
        """Perform one A2C gradient update on a rollout batch.

        Computes advantages ``A = R - V(s)`` (detached so the actor loss does
        not update the critic), combines actor, critic and entropy terms into a
        single loss, clips the global gradient norm to ``max_norm`` and takes
        one optimiser step.

        Args:
            states (torch.Tensor): Observations, shape ``(N, obs_dim)``.
            actions (torch.Tensor): Actions taken in those states, shape
                ``(N,)``.
            returns (torch.Tensor): n-step discounted returns R for each
                state, shape ``(N,)``.

        Returns:
            float: The total scalar loss for this update.
        """
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
        """Return the greedy (argmax-logit) action for evaluation/deployment.

        Args:
            observation (array-like | torch.Tensor): A single observation of
                shape ``(obs_dim,)``; converted to a float32 tensor.

        Returns:
            int: Index of the highest-scoring action.
        """
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
