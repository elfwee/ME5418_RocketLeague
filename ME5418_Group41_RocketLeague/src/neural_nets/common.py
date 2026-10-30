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