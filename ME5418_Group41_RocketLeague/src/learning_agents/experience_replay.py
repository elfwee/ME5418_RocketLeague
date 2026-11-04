"""Experience Replay Buffer for Off-Policy Deep Q-Network (DQN) Agents."""

from collections import deque
import random


class ReplayMemory:
    """Fixed-capacity circular experience replay buffer for off-policy RL.

    Stores transitions ``(state, action, next_state, reward, terminated, truncated)``
    and samples uniform minibatches to break temporal correlations during training.
    """

    def __init__(self, maxlen: int, seed: int | None = None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition: tuple):
        """Append a single transition tuple to the replay memory."""
        self.memory.append(transition)

    def sample(self, sample_size: int) -> list:
        """Uniformly sample a minibatch of transitions of size ``sample_size``."""
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        return len(self.memory)
