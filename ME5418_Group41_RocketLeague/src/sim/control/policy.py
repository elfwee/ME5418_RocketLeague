"""A trained policy driving one car at sim level (head-to-head play)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.config.nn_config import PPOConfig
from src.gym_env import action_codec
from src.neural_nets.checkpoint import load_policy
from src.neural_nets.policy import greedy_action
from src.sim.core.actions import CarAction
from src.sim.core.observation import build_observation

if TYPE_CHECKING:
    from src.sim.core.simulation import Simulation


class PolicyController:
    """Greedy actions from a PPO / (Dueling) DQN checkpoint, decided every `action_repeat` frames.

    Plays exactly like the env's `ActionRepeat` path: the jump bit is only held on the decision
    frame, and on Orange the observation and the action are mirrored so a model trained on Blue
    plays from its own perspective. `action_repeat` must match training (4 for both agents).
    """

    def __init__(
        self,
        team: str,
        sim: Simulation,
        model_path: str | Path,
        action_repeat: int = PPOConfig.action_repeat,
    ):
        if team not in ("blue", "orange"):
            raise ValueError(f"Invalid team for agent: {team}. Must be 'blue' or 'orange'.")
        self.team = team
        self.sim = sim
        self.action_repeat = max(1, action_repeat)
        self.policy, _ = load_policy(model_path, device="cpu")
        print(f"[PolicyController] Loaded {type(self.policy).__name__} agent from: {model_path}")
        self._frame = 0
        self._env_action = 0

    def get_action(self) -> CarAction:
        if self.team == "orange" and self.sim.orange_car is None:  # orange toggled off in-game
            self._frame = 0
            return CarAction()

        is_decision_frame = self._frame % self.action_repeat == 0
        if is_decision_frame:
            obs = torch.tensor(build_observation(self.sim, self.team), dtype=torch.float32)
            self._env_action = greedy_action(self.policy, obs)
        self._frame += 1
        env_action = self._env_action if is_decision_frame else action_codec.clear_jump(self._env_action)
        return action_codec.decode(env_action, mirror_x=self.team == "orange")

    def reset(self):
        self._frame = 0
        self._env_action = 0
