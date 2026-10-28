"""Shared building blocks for the low-level TensorFlow (no Keras) agents."""

import numpy as np
import tensorflow as tf
import gymnasium as gym


def discrete_action_nvec(action_space: gym.spaces.Space):
    """Return (nvec, is_multi_discrete) for a Discrete or MultiDiscrete action space."""
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return [int(n) for n in action_space.nvec], True
    if isinstance(action_space, gym.spaces.Discrete):
        return [int(action_space.n)], False
    raise TypeError(f"Requires a discrete action space, got {action_space}")


class MLP(tf.Module):
    """Fully connected network built from raw tf.Variables."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple, activation=tf.nn.relu, name=None):
        super().__init__(name=name)
        self.activation = activation
        sizes = [input_dim, *hidden_sizes, output_dim]
        self.weights = []
        self.biases = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            # He uniform initialisation for ReLU layers
            limit = np.sqrt(6.0 / fan_in)
            w = tf.random.uniform((fan_in, fan_out), -limit, limit, dtype=tf.float32)
            self.weights.append(tf.Variable(w, name=f"w{i}"))
            self.biases.append(tf.Variable(tf.zeros((fan_out,), dtype=tf.float32), name=f"b{i}"))

    def __call__(self, x):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(tf.matmul(x, w) + b)
        return tf.matmul(x, self.weights[-1]) + self.biases[-1]


class Adam(tf.Module):
    """Minimal Adam optimiser (tf.optimizers is part of Keras)."""

    def __init__(self, variables, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = tf.Variable(0.0, trainable=False)
        self.m = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
        self.v = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]

    def apply_gradients(self, grads, variables):
        self.t.assign_add(1.0)
        lr_t = self.lr * tf.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)
        for g, var, m, v in zip(grads, variables, self.m, self.v):
            m.assign(self.beta1 * m + (1.0 - self.beta1) * g)
            v.assign(self.beta2 * v + (1.0 - self.beta2) * tf.square(g))
            var.assign_sub(lr_t * m / (tf.sqrt(v) + self.eps))


class RMSProp(tf.Module):
    """Minimal RMSProp optimiser (tf.optimizers is part of Keras)."""

    def __init__(self, variables, learning_rate=7e-4, alpha=0.99, eps=1e-5):
        super().__init__()
        self.lr = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.sq = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]

    def apply_gradients(self, grads, variables):
        for g, var, sq in zip(grads, variables, self.sq):
            sq.assign(self.alpha * sq + (1.0 - self.alpha) * tf.square(g))
            var.assign_sub(self.lr * g / (tf.sqrt(sq) + self.eps))


class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, n_action_dims: int):
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_action_dims), dtype=np.int32)
        self.rewards = np.zeros((n_steps,), dtype=np.float32)
        self.dones = np.zeros((n_steps,), dtype=np.float32)
        self.values = np.zeros((n_steps,), dtype=np.float32)
        self.log_probs = np.zeros((n_steps,), dtype=np.float32)
        self.advantages = np.zeros((n_steps,), dtype=np.float32)
        self.returns = np.zeros((n_steps,), dtype=np.float32)
        self.n_steps = n_steps

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        # dones[t] marks that the episode ended after step t
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values


class ActorCritic(tf.Module):
    """Base for on-policy actor-critic agents (A2C, PPO).

    MultiDiscrete action spaces use one independent categorical head per action dimension.
    Subclasses implement `_make_optimizer` and `learn`.
    Config must provide: hidden_sizes, learning_rate, gamma, gae_lambda, n_steps, seed.
    """

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, config):
        super().__init__()
        self.config = config
        tf.random.set_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        self.action_nvec, self.is_multi_discrete = discrete_action_nvec(action_space)
        self.obs_dim = int(np.prod(observation_space.shape))

        self.actor = MLP(self.obs_dim, sum(self.action_nvec), config.hidden_sizes, activation=tf.nn.tanh, name="actor")
        self.critic = MLP(self.obs_dim, 1, config.hidden_sizes, activation=tf.nn.tanh, name="critic")
        # Small policy output layer keeps the initial policy close to uniform
        self.actor.weights[-1].assign(self.actor.weights[-1] * 0.01)
        self.variables_to_train = self.actor.trainable_variables + self.critic.trainable_variables
        self.optimizer = self._make_optimizer()
        self.num_timesteps = 0

    def _make_optimizer(self):
        raise NotImplementedError

    def learn(self, env: gym.Env, total_timesteps: int, log_dir: str = None):
        raise NotImplementedError

    # --- distribution helpers ---
    def _split_logits(self, obs):
        return tf.split(self.actor(obs), self.action_nvec, axis=1)

    def _log_prob_entropy(self, obs, actions):
        log_prob, entropy = 0.0, 0.0
        for i, logits in enumerate(self._split_logits(obs)):
            log_p = tf.nn.log_softmax(logits)
            log_prob += tf.gather(log_p, actions[:, i], axis=1, batch_dims=1)
            entropy += -tf.reduce_sum(tf.exp(log_p) * log_p, axis=1)
        return log_prob, entropy

    def _value(self, obs):
        return tf.squeeze(self.critic(obs), axis=1)

    @tf.function
    def _act(self, obs, deterministic):
        heads = self._split_logits(obs)
        if deterministic:
            actions = [tf.argmax(l, axis=1, output_type=tf.int32) for l in heads]
        else:
            actions = [tf.squeeze(tf.random.categorical(l, 1, dtype=tf.int32), axis=1) for l in heads]
        actions = tf.stack(actions, axis=1)
        log_prob, _ = self._log_prob_entropy(obs, actions)
        return actions, log_prob, self._value(obs)

    def _to_env_action(self, action):
        return action.astype(np.int64) if self.is_multi_discrete else int(action[0])

    def predict(self, observation, deterministic: bool = True):
        obs = tf.convert_to_tensor(np.reshape(observation, (1, -1)), dtype=tf.float32)
        action, _, _ = self._act(obs, tf.constant(deterministic))
        return self._to_env_action(action.numpy()[0]), None

    def value(self, observation) -> float:
        obs = tf.convert_to_tensor(np.reshape(observation, (1, -1)), dtype=tf.float32)
        return float(self._value(obs)[0])

    # --- rollout ---
    def _collect_rollout(self, env, buffer, obs, episode_rewards, episode_reward):
        c = self.config
        for t in range(c.n_steps):
            obs_t = tf.convert_to_tensor(np.reshape(obs, (1, -1)), dtype=tf.float32)
            action, log_prob, value = self._act(obs_t, tf.constant(False))
            action = action.numpy()[0]
            next_obs, reward, terminated, truncated, _ = env.step(self._to_env_action(action))
            episode_reward += reward
            self.num_timesteps += 1

            # Bootstrap through time-limit truncation instead of treating it as a terminal state
            if truncated and not terminated:
                reward += c.gamma * self.value(next_obs)

            buffer.obs[t] = obs
            buffer.actions[t] = action
            buffer.rewards[t] = reward
            buffer.dones[t] = float(terminated or truncated)
            buffer.values[t] = float(value[0])
            buffer.log_probs[t] = float(log_prob[0])

            obs = next_obs
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()

        buffer.compute_gae(self.value(obs), c.gamma, c.gae_lambda)
        return obs, episode_reward

    # --- persistence ---
    def save(self, path: str):
        tf.train.Checkpoint(actor=self.actor, critic=self.critic).write(path)

    def load(self, path: str):
        tf.train.Checkpoint(actor=self.actor, critic=self.critic).read(path).expect_partial()
        return self
