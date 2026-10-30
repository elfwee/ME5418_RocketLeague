"""Double Deep Q-Network (Double DQN) Architecture & Forward/Backward Pass Demo (TensorFlow/Keras)."""
import random
import sys
from collections import deque
from pathlib import Path
import gymnasium as gym
import numpy as np
import tensorflow as tf

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.gym_config import GymConfig
from src.config.nn_config import DQNConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.neural_nets.common import multi_layer_perceptron, ReplayBuffer


class DoubleDQN:
    """Double Deep Q-Network decoupling action selection and action evaluation in TensorFlow/Keras.

    Target formulation:
        a* = argmax_{a'} Q_online(s', a')
        y  = r + gamma * (1 - done) * Q_target(s', a*)
    """

    def __init__(self):
        # Build online Q-network and target Q-network with linear output head (36 discrete actions)
        q_output = tf.keras.layers.Dense(DQNConfig.output_dim, activation=None)
        target_output = tf.keras.layers.Dense(DQNConfig.output_dim, activation=None)
        self.q_net = multi_layer_perceptron(
            DQNConfig.input_dim, q_output, DQNConfig.hidden_layer, DQNConfig.hidden_size
        )
        self.target_net = multi_layer_perceptron(
            DQNConfig.input_dim, target_output, DQNConfig.hidden_layer, DQNConfig.hidden_size
        )
        self.target_net.set_weights(self.q_net.get_weights())  # synchronize initial weights

        # Optimizer, Huber TD loss function, and experience replay buffer
        self.optimiser = tf.keras.optimizers.SGD(DQNConfig.learning_rate)
        self.loss_fn = tf.keras.losses.Huber()
        self.experience_buffer = ReplayBuffer(DQNConfig.buffer_size)

    def _forward_pass(self, state, training: bool = False):
        """Compute action-value estimates Q(s, .) for input state batch."""
        return self.q_net(state, training=training)

    def greedy_action(self, observation: np.ndarray) -> int:
        """Select greedy action argmax_a Q(s, a) for a single 1D state observation."""
        q_values = self._forward_pass(observation[np.newaxis, :])  # add batch dim -> (1, 52)
        return int(tf.argmax(q_values, axis=1)[0])

    @tf.function
    def _back_propagation(self, state, action, reward, next_state, done):
        """Compute Double-DQN TD Huber loss, backpropagate gradients, and update online network."""
        # Online network selects best next action a*
        best_action = tf.argmax(self.q_net(next_state, training=False), axis=1, output_type=tf.int32)
        # Target network evaluates Q_target(s', a*) to mitigate maximization bias
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
        """Copy online Q-network weights to the target Q-network."""
        self.target_net.set_weights(self.q_net.get_weights())


def test_nn():
    """Assignment 2 Demo: Double DQN Instantiation, Forward/Backward Pass & Smoke Test."""
    env = gym.make(GymConfig.gym_id, render_mode=None, bot_type="heuristic_bot", discrete_action=True)
    net = DoubleDQN()

    print("=" * 65)
    print("1. NETWORK ARCHITECTURE (DOUBLE DQN - TENSORFLOW/KERAS)")
    print("=" * 65)
    print(f"State Dim: {DQNConfig.input_dim[0]} | Action Dim: {DQNConfig.output_dim} | Hidden: {DQNConfig.hidden_layer}x{DQNConfig.hidden_size}")
    net.q_net.summary()

    # 2. Single-Step Forward Pass Verification
    observation, info = env.reset()
    observation = np.asarray(observation, dtype=np.float32)
    state_batch = observation[np.newaxis, :]  # Shape: (1, 52)

    q_values = net._forward_pass(state_batch).numpy()  # Shape: (1, 36)
    action_idx = net.greedy_action(observation)
    dir_idx, jump_bit, boost_bit = action_idx // 4, (action_idx % 4) // 2, action_idx % 2

    print("\n" + "=" * 65)
    print("2. FORWARD PASS (INPUT -> OUTPUT)")
    print("=" * 65)
    print(f"Input State Shape : {state_batch.shape}")
    print(f"Input State (first 8 features): {state_batch[0, :8].round(4)}")
    print(f"Output Q(s, .) Shape: {q_values.shape}")
    print(f"Output Q(s, .) Sample (first 6 actions): {q_values[0, :6].round(4)}")
    print(f"Selected Action   : {action_idx} -> dir={GymConfig.directions[dir_idx]}, jump={bool(jump_bit)}, boost={bool(boost_bit)}")

    # 3. Single-Step Backward Pass Verification (Double-DQN TD Target, Loss & Gradients)
    next_obs, reward, terminated, truncated, info = env.step(action_idx)
    next_obs = np.asarray(next_obs, dtype=np.float32)
    next_state_batch = next_obs[np.newaxis, :]

    best_next_act = tf.argmax(net.q_net(next_state_batch, training=False), axis=1, output_type=tf.int32)
    next_q = tf.gather(net.target_net(next_state_batch, training=False), best_next_act, batch_dims=1)
    td_target = tf.constant([reward], dtype=tf.float32) + DQNConfig.discount_factor * (1.0 - float(terminated)) * next_q

    with tf.GradientTape() as tape:
        q_all = net.q_net(state_batch, training=True)
        q_pred = tf.gather(q_all, tf.constant([action_idx], dtype=tf.int32), batch_dims=1)
        single_loss = net.loss_fn(td_target, q_pred)
    grads = tape.gradient(single_loss, net.q_net.trainable_variables)
    net.optimiser.apply_gradients(zip(grads, net.q_net.trainable_variables))
    q_pred_after = float(net._forward_pass(state_batch).numpy()[0, action_idx])

    print("\n" + "=" * 65)
    print("3. BACKWARD PASS (LOSS & GRADIENTS)")
    print("=" * 65)
    print(f"Env Step Reward     : {reward:.4f} | Terminated: {terminated}")
    print(f"Predicted Q(s, a={action_idx}) : {float(q_pred.numpy()[0]):.6f} (before update)")
    print(f"Target Q-value (y)  : {float(td_target.numpy()[0]):.6f}")
    print(f"TD Huber Loss       : {float(single_loss.numpy()):.6f}")
    print("\nLayer Gradient Norms (L2):")
    for var, grad in zip(net.q_net.trainable_variables, grads):
        if grad is not None:
            var_label = getattr(var, "path", var.name)
            print(f"  {var_label:<22} | shape={str(list(var.shape)):<12} | grad_norm={float(tf.norm(grad).numpy()):.6f}")
    print(f"\nPredicted Q(s, a={action_idx}) : {q_pred_after:.6f} (after 1 optimizer step)")
    print("=" * 65)

    # 4. Experience Replay & Batch Backpropagation Smoke Test
    print(f"\n4. RUNNING {DQNConfig.demo_max_steps}-STEP REPLAY & BACKPROPAGATION SMOKE TEST...")
    observation, info = env.reset()
    observation = np.asarray(observation, dtype=np.float32)
    episode_return = 0.0
    recent_episode_returns = deque(maxlen=20)
    last_loss = 0.0

    for step in range(1, DQNConfig.demo_max_steps + 1):
        epsilon = max(0.05, 1.0 - step / DQNConfig.epsilon_decay_steps)  # epsilon-greedy exploration
        if random.random() < epsilon:
            action = random.randrange(DQNConfig.output_dim)
        else:
            action = net.greedy_action(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        # Only true termination cuts off bootstrapping; time-limit truncation still bootstraps
        net.experience_buffer.add(observation, action, reward, next_observation, terminated)
        observation = next_observation
        episode_return += reward

        if terminated or truncated:
            recent_episode_returns.append(episode_return)
            episode_return = 0.0
            observation, info = env.reset()
            observation = np.asarray(observation, dtype=np.float32)

        if len(net.experience_buffer) >= DQNConfig.learning_start:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals = (
                net.experience_buffer.sample(DQNConfig.batch_size)
            )
            loss_tensor = net._back_propagation(
                tf.constant(batch_states),
                tf.constant(batch_actions),
                tf.constant(batch_rewards),
                tf.constant(batch_next_states),
                tf.constant(batch_terminals),
            )
            last_loss = float(loss_tensor.numpy())
            if step % DQNConfig.target_update_interval == 0:
                net.sync_target()
            if step % int(DQNConfig.demo_max_steps // 10) == 0:
                mean_return = (
                    sum(recent_episode_returns) / len(recent_episode_returns)
                    if recent_episode_returns
                    else episode_return
                )
                print(
                    f"  step: {step:6d} | eps: {epsilon:.2f} | batch TD loss: {last_loss:.5f} | "
                    f"return: {mean_return:6.1f} | completed episodes: {len(recent_episode_returns)}"
                )

    env.close()


if __name__ == "__main__":
    test_nn()
