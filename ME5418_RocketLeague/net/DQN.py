"""Deep Q-Network implemented with low-level TensorFlow (no Keras)."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import gymnasium as gym

from ME5418_RocketLeague.net.common import MLP, Adam, discrete_action_nvec


@dataclass
class DQNConfig:
    hidden_sizes: tuple = (256, 256)
    learning_rate: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update_freq: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    max_grad_norm: float = 10.0
    seed: int = 0


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        i = rng.integers(0, self.size, size=batch_size)
        return self.obs[i], self.actions[i], self.rewards[i], self.next_obs[i], self.dones[i]


class DQN(tf.Module):
    """DQN agent with target network, experience replay and epsilon-greedy exploration.

    MultiDiscrete action spaces are flattened into a single Discrete index.
    """

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, config=DQNConfig()):
        super().__init__()
        self.config = config
        tf.random.set_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        nvec, _ = discrete_action_nvec(action_space)
        self.action_nvec = tuple(nvec)
        self.n_actions = int(np.prod(self.action_nvec))
        self.obs_dim = int(np.prod(observation_space.shape))

        self.q_net = MLP(self.obs_dim, self.n_actions, config.hidden_sizes, name="q_net")
        self.target_net = MLP(self.obs_dim, self.n_actions, config.hidden_sizes, name="target_net")
        self.update_target()
        self.optimizer = Adam(self.q_net.trainable_variables, learning_rate=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_size, self.obs_dim)
        self.num_timesteps = 0

    # --- action conversion ---
    def index_to_action(self, index: int):
        if len(self.action_nvec) == 1:
            return int(index)
        return np.array(np.unravel_index(int(index), self.action_nvec), dtype=np.int64)

    def action_to_index(self, action) -> int:
        if len(self.action_nvec) == 1:
            return int(action)
        return int(np.ravel_multi_index(tuple(action), self.action_nvec))

    # --- core ---
    def update_target(self):
        for t, s in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
            t.assign(s)

    def epsilon(self) -> float:
        c = self.config
        frac = min(1.0, self.num_timesteps / c.epsilon_decay_steps)
        return c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start)

    @tf.function
    def _greedy(self, obs):
        return tf.argmax(self.q_net(obs), axis=1)

    def predict(self, observation, deterministic: bool = True):
        if not deterministic and self.rng.random() < self.epsilon():
            index = self.rng.integers(self.n_actions)
        else:
            obs = tf.convert_to_tensor(np.reshape(observation, (1, -1)), dtype=tf.float32)
            index = int(self._greedy(obs)[0])
        return self.index_to_action(index), index

    @tf.function
    def _train_step(self, obs, actions, rewards, next_obs, dones):
        # Double DQN target: action chosen by online net, evaluated by target net
        next_actions = tf.argmax(self.q_net(next_obs), axis=1, output_type=tf.int32)
        next_q = tf.gather(self.target_net(next_obs), next_actions, axis=1, batch_dims=1)
        targets = tf.stop_gradient(rewards + self.config.gamma * (1.0 - dones) * next_q)

        variables = self.q_net.trainable_variables
        with tf.GradientTape() as tape:
            q = tf.gather(self.q_net(obs), actions, axis=1, batch_dims=1)
            # Huber loss
            err = targets - q
            abs_err = tf.abs(err)
            loss = tf.reduce_mean(tf.where(abs_err < 1.0, 0.5 * tf.square(err), abs_err - 0.5))
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        self.optimizer.apply_gradients(grads, variables)
        return loss

    def learn(self, env: gym.Env, total_timesteps: int, log_dir: str = None, log_interval: int = 1_000):
        c = self.config
        writer = tf.summary.create_file_writer(log_dir) if log_dir else None
        obs, _ = env.reset(seed=c.seed)
        episode_reward, episode_rewards = 0.0, []
        loss = None

        for _ in range(total_timesteps):
            action, index = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            # Only bootstrap-cut on true termination, not time-limit truncation
            self.buffer.add(obs, index, reward, next_obs, float(terminated))
            obs = next_obs
            episode_reward += reward
            self.num_timesteps += 1

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                if writer:
                    with writer.as_default():
                        tf.summary.scalar("rollout/ep_rew", episode_reward, step=self.num_timesteps)
                episode_reward = 0.0
                obs, _ = env.reset()

            if self.num_timesteps >= c.learning_starts and self.num_timesteps % c.train_freq == 0:
                batch = self.buffer.sample(c.batch_size, self.rng)
                loss = self._train_step(*(tf.convert_to_tensor(b) for b in batch))

            if self.num_timesteps % c.target_update_freq == 0:
                self.update_target()

            if self.num_timesteps % log_interval == 0:
                mean_rew = np.mean(episode_rewards[-10:]) if episode_rewards else float("nan")
                loss_val = float(loss) if loss is not None else float("nan")
                print(
                    f"step {self.num_timesteps:>8d} | eps {self.epsilon():.3f} | "
                    f"loss {loss_val:.4f} | mean_ep_rew(10) {mean_rew:.2f}"
                )
                if writer and loss is not None:
                    with writer.as_default():
                        tf.summary.scalar("train/loss", loss, step=self.num_timesteps)
                        tf.summary.scalar("train/epsilon", self.epsilon(), step=self.num_timesteps)
        if writer:
            writer.flush()
        return self

    # --- persistence ---
    def save(self, path: str):
        tf.train.Checkpoint(q_net=self.q_net, target_net=self.target_net).write(path)

    def load(self, path: str):
        tf.train.Checkpoint(q_net=self.q_net, target_net=self.target_net).read(path).expect_partial()
        return self
