"""Proximal Policy Optimisation implemented with low-level TensorFlow (no Keras)."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import gymnasium as gym

from ME5418_RocketLeague.net.common import ActorCritic, Adam, RolloutBuffer


@dataclass
class PPOConfig:
    hidden_sizes: tuple = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    seed: int = 0


class PPO(ActorCritic):
    """PPO-Clip agent with separate actor and critic networks and GAE."""

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, config=PPOConfig()):
        super().__init__(observation_space, action_space, config)

    def _make_optimizer(self):
        return Adam(self.variables_to_train, learning_rate=self.config.learning_rate)

    @tf.function
    def _train_step(self, obs, actions, old_log_probs, advantages, returns):
        c = self.config
        if c.normalize_advantage:
            mean, var = tf.nn.moments(advantages, axes=[0])
            advantages = (advantages - mean) / (tf.sqrt(var) + 1e-8)

        with tf.GradientTape() as tape:
            log_prob, entropy = self._log_prob_entropy(obs, actions)
            ratio = tf.exp(log_prob - old_log_probs)
            clipped = tf.clip_by_value(ratio, 1.0 - c.clip_range, 1.0 + c.clip_range)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
            value_loss = tf.reduce_mean(tf.square(returns - self._value(obs)))
            entropy_loss = -tf.reduce_mean(entropy)
            loss = policy_loss + c.vf_coef * value_loss + c.ent_coef * entropy_loss
        grads = tape.gradient(loss, self.variables_to_train)
        grads, _ = tf.clip_by_global_norm(grads, c.max_grad_norm)
        self.optimizer.apply_gradients(grads, self.variables_to_train)

        approx_kl = tf.reduce_mean(old_log_probs - log_prob)
        clip_frac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > c.clip_range, tf.float32))
        return policy_loss, value_loss, -entropy_loss, approx_kl, clip_frac

    def learn(self, env: gym.Env, total_timesteps: int, log_dir: str = None):
        c = self.config
        writer = tf.summary.create_file_writer(log_dir) if log_dir else None
        buffer = RolloutBuffer(c.n_steps, self.obs_dim, len(self.action_nvec))
        obs, _ = env.reset(seed=c.seed)
        episode_reward, episode_rewards = 0.0, []

        while self.num_timesteps < total_timesteps:
            obs, episode_reward = self._collect_rollout(env, buffer, obs, episode_rewards, episode_reward)

            stats = []
            for _ in range(c.n_epochs):
                indices = self.rng.permutation(c.n_steps)
                for start in range(0, c.n_steps, c.batch_size):
                    i = indices[start : start + c.batch_size]
                    batch = (buffer.obs[i], buffer.actions[i], buffer.log_probs[i], buffer.advantages[i], buffer.returns[i])
                    stats.append([float(s) for s in self._train_step(*(tf.convert_to_tensor(b) for b in batch))])
            policy_loss, value_loss, entropy, approx_kl, clip_frac = np.mean(stats, axis=0)

            mean_rew = np.mean(episode_rewards[-10:]) if episode_rewards else float("nan")
            print(
                f"step {self.num_timesteps:>8d} | pi_loss {policy_loss:.4f} | v_loss {value_loss:.4f} | "
                f"entropy {entropy:.3f} | kl {approx_kl:.4f} | clip {clip_frac:.3f} | mean_ep_rew(10) {mean_rew:.2f}"
            )
            if writer:
                with writer.as_default():
                    step = self.num_timesteps
                    tf.summary.scalar("train/policy_loss", policy_loss, step=step)
                    tf.summary.scalar("train/value_loss", value_loss, step=step)
                    tf.summary.scalar("train/entropy", entropy, step=step)
                    tf.summary.scalar("train/approx_kl", approx_kl, step=step)
                    tf.summary.scalar("train/clip_fraction", clip_frac, step=step)
                    if episode_rewards:
                        tf.summary.scalar("rollout/ep_rew_mean", mean_rew, step=step)
        if writer:
            writer.flush()
        return self
