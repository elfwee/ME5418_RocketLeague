import random
import numpy as np
import tensorflow as tf
from collections import deque

from src.config.nn_config import PPOConfig

from src.neural_nets.common import RolloutBuffer

class PPO:
    def __init__(self):
        # BUILD NEURAL NETWORK
        self.actor_net = self._build_mlp(PPOConfig.input_dim, (PPOConfig.hidden_size, PPOConfig.hidden_size), PPOConfig.output_dim)
        self.critic_net = self._build_mlp(PPOConfig.input_dim, (PPOConfig.hidden_size, PPOConfig.hidden_size), 1)
        # SET OPTIMISER
        self.optimiser = tf.keras.optimizers.Adam(PPOConfig.learning_rate)

    @staticmethod
    def _build_mlp(input_dim, hidden, out_dim):
        net = tf.keras.Sequential()
        net.add(tf.keras.Input(input_dim))
        for units in hidden:
            net.add(tf.keras.layers.Dense(units, activation="tanh"))
        net.add(tf.keras.layers.Dense(out_dim))
        return net

    @property
    def trainable_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables
        
    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(self, obs, actions=None):
        """Returns (actions, log_prob, entropy, value). Samples actions if none are given."""
        obs = tf.convert_to_tensor(obs, tf.float32)
        if obs.shape.rank == 1:
            obs = obs[None]

        logits = self.actor_net(obs)
        value = tf.squeeze(self.critic_net(obs), axis=-1)

        if actions is None:
            actions = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        actions = tf.cast(actions, tf.int32)

        log_probs_all = tf.nn.log_softmax(logits)
        log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
        entropy = -tf.reduce_sum(tf.exp(log_probs_all) * log_probs_all, axis=-1)
        return actions, log_prob, entropy, value

    # ------------------------------------------------------------------
    # Agent-facing helpers
    # ------------------------------------------------------------------
    def act(self, obs, deterministic=False):
        """Pick an action for a single observation. Returns plain Python/numpy values."""
        if deterministic:
            logits = self.actor_net(tf.convert_to_tensor(obs, tf.float32)[None])
            return int(tf.argmax(logits, axis=-1)[0]), 0.0, 0.0
        action, log_prob, _, value = self.forward(obs)
        return int(action[0]), float(log_prob[0]), float(value[0])

    def get_value(self, obs):
        """State value for a single observation (used to bootstrap at the end of a rollout)."""
        obs = tf.convert_to_tensor(obs, tf.float32)[None]
        return float(tf.squeeze(self.critic_net(obs)))

    # ------------------------------------------------------------------
    # Back propagation
    # ------------------------------------------------------------------
    @tf.function
    def backward(self, obs, actions, old_log_prob, advantages, returns):
        """One gradient step on a minibatch using the PPO clipped loss."""
        with tf.GradientTape() as tape:
            _, new_log_prob, entropy, value = self.forward(obs, actions)

            ratio = tf.exp(new_log_prob - old_log_prob)
            unclipped = ratio * advantages
            clipped = tf.clip_by_value(ratio, 1.0 - PPOConfig.clipping_epsilon, 1.0 + PPOConfig.clipping_epsilon) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

            value_loss = tf.reduce_mean(tf.square(returns - value))
            entropy_mean = tf.reduce_mean(entropy)

            loss = policy_loss + PPOConfig.value_function_coefficient * value_loss - PPOConfig.entropy_coefficient * entropy_mean

        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        approx_kl = tf.reduce_mean(old_log_prob - new_log_prob)
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "approx_kl": approx_kl,
        }

    def update(self, buffer, epochs=PPOConfig.epochs, batch_size=PPOConfig.batch_size, target_kl=PPOConfig.target_kl):
        """Learn from a finished RolloutBuffer. The Agent calls this once per rollout."""
        stats = {}
        for _ in range(epochs):
            kls = []
            for mb in buffer.minibatches(batch_size):
                stats = self.backward(*mb)
                kls.append(float(stats["approx_kl"]))
            if np.mean(kls) > target_kl:  # policy moved far enough; stop early
                break
        return {k: float(v) for k, v in stats.items()}

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def save(self, path_prefix):
        self.actor.save_weights(f"{path_prefix}_actor.weights.h5")
        self.critic.save_weights(f"{path_prefix}_critic.weights.h5")

    def load(self, path_prefix):
        self.actor.load_weights(f"{path_prefix}_actor.weights.h5")
        self.critic.load_weights(f"{path_prefix}_critic.weights.h5")

def test_nn():
    # Quick smoke test on the gym environment
    net = PPO()