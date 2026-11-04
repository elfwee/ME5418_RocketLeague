"""Shared neural network utilities and experience replay buffer for TensorFlow/Keras DQN models."""
import random
from collections import deque
import numpy as np
import tensorflow as tf


def dense_layer(num_units: int) -> tf.keras.layers.Dense:
    """Create a fully-connected Dense layer with ReLU activation and He variance-scaling initialization."""
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal"
        ),
    )


def multi_layer_perceptron(
    input_dim: tuple[int, ...],
    output_layer: tf.keras.layers.Dense,
    hidden_layer: int = 2,
    hidden_size: int = 64,
) -> tf.keras.Sequential:
    """Build a sequential Multi-Layer Perceptron (MLP) mapping state observations to action Q-values."""
    return tf.keras.Sequential(
        [tf.keras.Input(shape=input_dim)]
        + [dense_layer(hidden_size) for _ in range(hidden_layer)]
        + [output_layer]
    )


class ReplayBuffer:
    """Fixed-capacity FIFO experience replay buffer for off-policy TD learning."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)  # oldest transitions fall off when full

    def add(self, state, action, reward, next_state, done):
        """Store a single transition tuple (s, a, r, s', done)."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Uniformly sample a mini-batch of transitions as typed NumPy arrays."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            state.astype(np.float32),
            action.astype(np.int32),
            reward.astype(np.float32),
            next_state.astype(np.float32),
            done.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
