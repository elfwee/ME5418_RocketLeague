import random
import numpy as np
import tensorflow as tf
from collections import deque

from ME5418_RocketLeague.config.nn_config import DQNConfig
from ME5418_RocketLeague.gym_environment.bentjh_gym import RocketLeagueSideSwipeGym

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

class DQN():
    def __init__(self):
        # BUILD NEURAL NETWORK
        q_output = tf.keras.layers.Dense(DQNConfig.output_dim, activation="softmax")
        target_output = tf.keras.layers.Dense(DQNConfig.output_dim, activation="softmax")
        self.q_net = multi_layer_perceptron(DQNConfig.input_dim, q_output, DQNConfig.hidden_layer, DQNConfig.hidden_size)
        self.target_net = multi_layer_perceptron(DQNConfig.input_dim, target_output, DQNConfig.hidden_layer, DQNConfig.hidden_size)
        self.target_net.set_weights(self.q_net.get_weights()) # make target and q networks identical

        self.optimiser = tf.keras.optimizers.SGD(DQNConfig.learning_rate)
        self.discount_factor = tf.constant(DQNConfig.discount_factor)
        
        self.experience_buffer = ReplayBuffer(DQNConfig.buffer_size)

    def _forward_pass(self, state):
        return self.q_net(state)

    @tf.function
    def _back_propagation(self, state, action, reward, next_state, done):
        # Bellman target from the slowly-updated target network
        max_next_q_values = tf.reduce_max(self.target_net(next_state), axis=1)
        temporal_difference_targets = reward + self.discount_factor * (1.0 - done) * max_next_q_values
        with tf.GradientTape() as tape:
            q_all = self._forward_pass(state) # Q(s, ·; θ), shape (input_dim, output_dim)
            action_mask = tf.one_hot(action, DQNConfig.output_dim) # shape (input_dim, output_dim)
            q_taken = tf.reduce_sum(q_all * action_mask, axis=1) # Q(s, a; θ), shape (input_dim,)
            loss = tf.reduce_mean(tf.keras.losses.huber(temporal_difference_targets, q_taken))
        gradients = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.q_net.trainable_variables))

def test_nn():
    # Quick smoke test with random data
    env = RocketLeagueSideSwipeGym()
    net = DQN()
    replay_buffer = ReplayBuffer(DQNConfig.buffer_size)
    observation, info = env.reset()
    observation = tf.transpose(tf.convert_to_tensor(observation))
    episode_return = 0.0

    action = net._forward_pass(observation)
    print(action)

    recent_episode_returns = deque(maxlen=20)
    for step in range(30_000):
        break
        epsilon = max(0.05, 1.0 - step / DQNConfig.epsilon_decay_steps)  # epsilon-greedy
        if random.random() < epsilon:
            action = random.randrange(DQNConfig.output_dim)
        else:
            action = net._forward_pass(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = tf.transpose(tf.convert_to_tensor(next_observation))
        replay_buffer.add(observation, action, reward, next_observation, terminated or truncated)
        observation = next_observation
        episode_return += reward

        if len(replay_buffer) >= DQNConfig.learning_start:
            (batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals) = replay_buffer.sample(DQNConfig.batch_size)
            net._back_propagation(tf.constant(batch_states), tf.constant(batch_actions),
                               tf.constant(batch_rewards), tf.constant(batch_next_states),
                               tf.constant(batch_terminals))
            if step % DQNConfig.target_update_interval == 0:
                net.target_net.set_weights(net.q_net.get_weights())
            if step % 2000 == 0:
                mean_return = sum(recent_episode_returns) / max(len(recent_episode_returns), 1)
                print(f"  step {step:6d}  eps {epsilon:.2f}  train return {mean_return:6.1f}")
