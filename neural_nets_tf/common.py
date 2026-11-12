import random
import numpy as np
import tensorflow as tf
from collections import deque

# Define a helper function to create Dense layers configured with identical activation and kernel_initializer
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# Definer helper function to create an MLP
def multi_layer_perceptron(input_dim : tuple[int, ...], 
                                 output_layer:tf.keras.layers.Dense, 
                                 hidden_layer:int = 2, 
                                 hidden_size:int = 64, ) -> tf.keras.Sequential:
    return tf.keras.Sequential([tf.keras.Input(shape=input_dim)] + 
        [dense_layer(hidden_size) for i in range(hidden_layer)] + 
        [output_layer])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)   # old data falls off the end

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)   # uniform random
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (state.astype(np.float32), action.astype(np.int32), reward.astype(np.float32),
                next_state.astype(np.float32), done.astype(np.float32))

    def __len__(self):
        return len(self.buffer)


class RolloutBuffer:
    """Stores one batch of experience and turns it into training data."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
        self.advantages = self.returns = None

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.values.append(value)

    def __len__(self):
        return len(self.rewards)

    def finish(self, last_value, gamma=0.99, lam=0.95):
        """Compute advantages (GAE) and returns once the rollout is complete."""
        rewards = np.array(self.rewards, np.float32)
        values = np.array(self.values, np.float32)
        dones = np.array(self.dones, np.float32)
        T = len(rewards)

        adv = np.zeros(T, np.float32)
        running = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            running = delta + gamma * lam * not_done * running
            adv[t] = running

        self.returns = adv + values
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    def minibatches(self, batch_size):
        """Yield shuffled minibatches as tensors."""
        obs = np.array(self.obs, np.float32)
        actions = np.array(self.actions, np.int32)
        log_probs = np.array(self.log_probs, np.float32)
        n = len(self)
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            yield (tf.constant(obs[mb]), tf.constant(actions[mb]),
                   tf.constant(log_probs[mb]), tf.constant(self.advantages[mb]),
                   tf.constant(self.returns[mb]))