"""Experience Replay Buffer for Off-Policy Deep Q-Network (DQN) Agents."""

import random
from collections import deque


class ReplayMemory:
    """Fixed-capacity circular experience replay buffer for off-policy RL.

    Stores transitions ``(state, action, next_state, reward, terminated)`` and samples uniform
    minibatches to break temporal correlations during training. Sampling uses the buffer's own
    generator, seeded with ``seed`` (fresh entropy when None).
    """

    def __init__(self, maxlen: int, seed: int | None = None):
        self.memory = deque([], maxlen=maxlen)
        self.rng = random.Random(seed)

    def append(self, transition: tuple):
        """Append a single transition tuple to the replay memory."""
        self.memory.append(transition)

    def sample(self, sample_size: int) -> list:
        """Uniformly sample a minibatch of transitions of size ``sample_size``."""
        return self.rng.sample(self.memory, sample_size)

    def __len__(self) -> int:
        return len(self.memory)
