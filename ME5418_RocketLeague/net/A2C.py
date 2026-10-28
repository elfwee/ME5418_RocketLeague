"""Advantage Actor-Critic implemented with low-level TensorFlow (no Keras)."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import gymnasium as gym

from ME5418_RocketLeague.net.common import ActorCritic, RMSProp, RolloutBuffer


@dataclass
class A2CConfig:
    hidden_sizes: tuple = (64, 64)
    learning_rate: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 1.0
    n_steps: int = 5
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rms_alpha: float = 0.99
    rms_eps: float = 1e-5
    normalize_advantage: bool = False
    seed: int = 0


class A2C(ActorCritic):
    """Synchronous A2C: one unclipped policy-gradient step per short rollout."""

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, config=A2CConfig()):
        super().__init__(observation_space, action_space, config)

    def _make_optimizer(self):
        c = self.config
        return RMSProp(self.variables_to_train, learning_rate=c.learning_rate, alpha=c.rms_alpha, eps=c.rms_eps)

    @tf.function
    def _train_step(self, obs, actions, advantages, returns):
        c = self.config
        if c.normalize_advantage:
            mean, var = tf.nn.moments(advantages, axes=[0])
            advantages = (advantages - mean) / (tf.sqrt(var) + 1e-8)

        with tf.GradientTape() as tape:
            log_prob, entropy = self._log_prob_entropy(obs, actions)
            policy_loss = -tf.reduce_mean(log_prob * advantages)
            value_loss = tf.reduce_mean(tf.square(returns - self._value(obs)))
            entropy_loss = -tf.reduce_mean(entropy)
            loss = policy_loss + c.vf_coef * value_loss + c.ent_coef * entropy_loss
        grads = tape.gradient(loss, self.variables_to_train)
        grads, _ = tf.clip_by_global_norm(grads, c.max_grad_norm)
        self.optimizer.apply_gradients(grads, self.variables_to_train)
        return policy_loss, value_loss, -entropy_loss

    def learn(self, env: gym.Env, total_timesteps: int, log_dir: str = None, log_interval: int = 100):
        c = self.config
        writer = tf.summary.create_file_writer(log_dir) if log_dir else None
        buffer = RolloutBuffer(c.n_steps, self.obs_dim, len(self.action_nvec))
        obs, _ = env.reset(seed=c.seed)
        episode_reward, episode_rewards = 0.0, []
        n_updates = 0

        while self.num_timesteps < total_timesteps:
            obs, episode_reward = self._collect_rollout(env, buffer, obs, episode_rewards, episode_reward)
            batch = (buffer.obs, buffer.actions, buffer.advantages, buffer.returns)
            policy_loss, value_loss, entropy = (
                float(s) for s in self._train_step(*(tf.convert_to_tensor(b) for b in batch))
            )
            n_updates += 1

            if n_updates % log_interval == 0:
                mean_rew = np.mean(episode_rewards[-10:]) if episode_rewards else float("nan")
                print(
                    f"step {self.num_timesteps:>8d} | pi_loss {policy_loss:.4f} | v_loss {value_loss:.4f} | "
                    f"entropy {entropy:.3f} | mean_ep_rew(10) {mean_rew:.2f}"
                )
                if writer:
                    with writer.as_default():
                        step = self.num_timesteps
                        tf.summary.scalar("train/policy_loss", policy_loss, step=step)
                        tf.summary.scalar("train/value_loss", value_loss, step=step)
                        tf.summary.scalar("train/entropy", entropy, step=step)
                        if episode_rewards:
                            tf.summary.scalar("rollout/ep_rew_mean", mean_rew, step=step)
        if writer:
            writer.flush()
        return self
