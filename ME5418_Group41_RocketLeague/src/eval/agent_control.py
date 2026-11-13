from pathlib import Path

import numpy as np
import torch

from src.config.nn_config import PPOConfig
from src.eval.orange_perspective import get_orange_perspective_obs
from src.gym_env import action_codec
from src.neural_nets.checkpoint import load_policy
from src.neural_nets.policy import greedy_action
from src.sim.core.actions import CarAction
from src.sim.core.simulation import Simulation


class AgentControl:
    """Controller wrapper dispatching car actions by learned policy
    - team:
        - orange
        - blue
        (agent on orange uses a blue-trained model via the mirrored perspective in orange_perspective.py)
    - sim: Simulation needed for HeuristicBot and agent
    - model_path: .pt state_dict needed for agent
    - action_repeat: decision interval for agent; must match training (PPOConfig/DQNConfig action_repeat)
    """

    def __init__(
        self,
        team: str = "blue",
        sim: Simulation | None = None,
        model_path: str | Path | None = None,
        action_repeat: int = PPOConfig.action_repeat,
    ):
        self.sim = sim
        self.team = team
        self.action = CarAction()

        self.policy = None
        self.action_repeat = max(1, action_repeat)
        self._agent_frame = 0
        self._agent_action = 0
        if self.team not in ("blue", "orange"):
            raise ValueError(f"Invalid team for agent: {self.team}. Must be 'blue' or 'orange'.")
        if self.sim is None or model_path is None:
            raise ValueError("control_type='agent' requires both sim and model_path.")
        self.policy = self._load_policy(model_path)

    @staticmethod
    def _load_policy(model_path: str | Path):
        """Load a PPO actor-critic or (Dueling) DQN checkpoint for CPU inference."""
        net, _ = load_policy(model_path, device="cpu")
        print(f"[AgentControl] Loaded {type(net).__name__} agent from: {model_path}")
        return net

    def get_action(self) -> CarAction:
        """Greedy policy action, re-decided every `action_repeat` frames like in training. The jump
        bit is only held on the decision frame so edge-triggered jumping can re-fire. On orange, the
        observation and action are mirrored so a blue-trained model plays from its own perspective."""
        if self.team == "orange" and self.sim.orange_car is None:  # orange toggled off in-game
            self._agent_frame = 0
            return CarAction()

        is_decision_frame = self._agent_frame % self.action_repeat == 0
        if is_decision_frame:
            raw_obs = (
                get_orange_perspective_obs(self.sim)
                if self.team == "orange"
                else self.sim.get_state_norm(as_flat_array=True)
            )
            obs = torch.from_numpy(np.asarray(raw_obs, dtype=np.float32))
            self._agent_action = greedy_action(self.policy, obs)
        self._agent_frame += 1
        env_action = self._agent_action if is_decision_frame else action_codec.clear_jump(self._agent_action)
        action = action_codec.decode(env_action)
        return action_codec.mirror(action) if self.team == "orange" else action

    def reset(self):
        self.action = CarAction(dir_x=0, dir_y=0, jump=False, boost=False)
        self._agent_frame = 0
        self._agent_action = 0
