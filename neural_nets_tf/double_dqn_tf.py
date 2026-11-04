import random
import numpy as np
import tensorflow as tf
from collections import deque

from src.config.nn_config import DQNConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv

from src.neural_nets.common import multi_layer_perceptron, ReplayBuffer

class DoubleDQN():
    def __init__(self):
        # BUILD NEURAL NETWORK
        q_output = tf.keras.layers.Dense(DQNConfig.output_dim, activation=None)
        target_output = tf.keras.layers.Dense(DQNConfig.output_dim, activation=None)
        self.q_net = multi_layer_perceptron(DQNConfig.input_dim, q_output, DQNConfig.hidden_layer, DQNConfig.hidden_size)
        self.target_net = multi_layer_perceptron(DQNConfig.input_dim, target_output, DQNConfig.hidden_layer, DQNConfig.hidden_size)
        self.target_net.set_weights(self.q_net.get_weights()) # make target and q networks identical
        # SET OPTIMISER
        self.optimiser = tf.keras.optimizers.SGD(DQNConfig.learning_rate)
        # LOSS FUNTION
        self.loss_fn = tf.keras.losses.Huber()
        # BUFFER
        self.experience_buffer = ReplayBuffer(DQNConfig.buffer_size)

    def _forward_pass(self, state):
        return self.q_net(state)

    def greedy_action(self, observation):
        q_values = self._forward_pass(observation[np.newaxis, :])  # add batch dim -> (1, input_dim)
        return int(tf.argmax(q_values, axis=1)[0])

    @tf.function
    def _back_propagation(self, state, action, reward, next_state, done):
        # ONLINE NETWORK
        best_action = tf.argmax(self.q_net(next_state, training=False), axis=1, output_type=tf.int32)
        # TARGET NETWORK
        next_q = tf.gather(self.target_net(next_state, training=False), best_action, batch_dims=1)

        td_target = reward + DQNConfig.discount_factor * (1.0 - done) * next_q
        with tf.GradientTape() as tape:
            q_all = self.q_net(state, training=True)
            q_sa = tf.gather(q_all, action, batch_dims=1)
            loss = self.loss_fn(td_target, q_sa)
        gradients = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.q_net.trainable_variables))
        return loss

    def sync_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

def test_nn():
    # Quick smoke test on the gym environment
    env = RocketLeagueEnv()
    net = DoubleDQN()
    observation, info = env.reset()
    observation = np.asarray(observation, dtype=np.float32)
    episode_return = 0.0

    print("Q-values:", net._forward_pass(observation[np.newaxis, :]).numpy())
    recent_episode_returns = deque(maxlen=20)
    for step in range(DQNConfig.demo_max_steps):
        epsilon = max(0.05, 1.0 - step / DQNConfig.epsilon_decay_steps)  # epsilon-greedy
        if random.random() < epsilon:
            action = random.randrange(DQNConfig.output_dim)
        else:
            action = net.greedy_action(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)  # env uses Discrete(36) and decodes the flat index itself
        next_observation = np.asarray(next_observation, dtype=np.float32)
        # only true termination cuts off bootstrapping; truncation still bootstraps
        net.experience_buffer.add(observation, action, reward, next_observation, terminated)
        observation = next_observation
        episode_return += reward

        if terminated or truncated:
            recent_episode_returns.append(episode_return)
            episode_return = 0.0
            observation, info = env.reset()
            observation = np.asarray(observation, dtype=np.float32)

        if len(net.experience_buffer) >= DQNConfig.learning_start:
            (batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals) = net.experience_buffer.sample(DQNConfig.batch_size)
            net._back_propagation(tf.constant(batch_states), tf.constant(batch_actions),
                               tf.constant(batch_rewards), tf.constant(batch_next_states),
                               tf.constant(batch_terminals))
            if step % DQNConfig.target_update_interval == 0:
                net.sync_target()
            if step % int(DQNConfig.demo_max_steps//10) == 0:
                mean_return = sum(recent_episode_returns) / max(len(recent_episode_returns), 1)
                print(f"  step: {step:6d} | eps: {epsilon:.2f} | train return: {mean_return:6.1f} | episodes: {len(recent_episode_returns)}")
