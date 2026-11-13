from pathlib import Path

import numpy as np
from src.config.gym_config import GymConfig
from src.config.nn_config import PPOConfig
from src.sim.core.actions import CarAction
from src.sim.core.simulation import Simulation
from src.eval.orange_perspective import get_orange_perspective_obs, mirror_action
from typing import Optional


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
        sim: Optional[Simulation] = None,
        model_path: Optional[str | Path] = None,
        action_repeat: int = PPOConfig.action_repeat,
    ):
        self.sim = sim
        self.team = team
        self.action = CarAction()

        self.policy = None
        self.action_repeat = max(1, action_repeat)
        self._agent_frame = 0
        self._agent_action_idx = 0
        if self.team not in ("blue", "orange"):
            raise ValueError(f"Invalid team for agent: {self.team}. Must be 'blue' or 'orange'.")
        if self.sim is None or model_path is None:
            raise ValueError("control_type='agent' requires both sim and model_path.")
        self.policy = self._load_policy(model_path)

    @staticmethod
    def _load_policy(model_path: str | Path):
        """Load a PPO actor-critic or (Dueling) DQN state_dict for CPU inference, inferring the
        architecture from the checkpoint's layer names and shapes."""
        import torch

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dim = state_dict["fc1.weight"].shape[1]
        hidden_dim = state_dict["fc1.weight"].shape[0]

        if "actor_heads.0.weight" in state_dict or "actor.weight" in state_dict:
            from src.neural_nets.ppo_torch import ActorCritic

            if "actor_heads.0.weight" in state_dict:  # factored MultiDiscrete([9, 2, 2]) heads
                n_heads = len({k.split(".")[1] for k in state_dict if k.startswith("actor_heads.")})
                action_dim = [state_dict[f"actor_heads.{i}.weight"].shape[0] for i in range(n_heads)]
            else:  # flat Discrete(36) head
                action_dim = state_dict["actor.weight"].shape[0]
            net = ActorCritic(state_dim, action_dim, hidden_dim)
        else:
            from src.neural_nets.dueling_ddqn_torch import DQN

            dueling = "advantages.weight" in state_dict
            out_key = "advantages.weight" if dueling else "fc3.weight"
            net = DQN(state_dim, state_dict[out_key].shape[0], hidden_dim, enable_dueling_dqn=dueling)

        net.load_state_dict(state_dict)
        net.eval()
        print(f"[AgentControl] Loaded {type(net).__name__} agent from: {model_path}")
        return net

    @staticmethod
    def _decode_discrete_action(action_idx: int, allow_jump: bool = True) -> CarAction:
        """Map a Discrete(36) index to CarAction (same layout as RocketLeagueEnv.step)."""
        direction, jump, boost = action_idx // 4, (action_idx % 4) // 2, action_idx % 2
        dir_x, dir_y = GymConfig.directions[direction]
        return CarAction(
            dir_x=float(dir_x), dir_y=float(dir_y), jump=allow_jump and jump == 1, boost=boost == 1
        ).clamp()

    @staticmethod
    def _greedy_action_idx(out) -> int:
        """Greedy Discrete(36) index from a network output: PPO ActorCritic -> (logits, V), where
        logits is one tensor (flat) or [direction, jump, boost] tensors (factored); DQN -> Q-values."""
        scores = out[0] if isinstance(out, tuple) else out
        if isinstance(scores, (list, tuple)):
            direction, jump, boost = (int(s.squeeze(0).argmax().item()) for s in scores)
            return direction * 4 + jump * 2 + boost
        return int(scores.squeeze(0).argmax().item())

    def get_action(self) -> CarAction:
        """Greedy policy action, re-decided every `action_repeat` frames like in training. The jump
        bit is only held on the decision frame so edge-triggered jumping can re-fire. On orange, the
        observation and action are mirrored so a blue-trained model plays from its own perspective."""
        import torch

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
            obs = np.asarray(raw_obs, dtype=np.float32)
            with torch.no_grad():
                out = self.policy(torch.from_numpy(obs).unsqueeze(0))
            self._agent_action_idx = self._greedy_action_idx(out)
        self._agent_frame += 1
        action = self._decode_discrete_action(self._agent_action_idx, allow_jump=is_decision_frame)
        return mirror_action(action) if self.team == "orange" else action

    def reset(self):
        self.action = CarAction(dir_x=0, dir_y=0, jump=False, boost=False)
        self._agent_frame = 0
        self._agent_action_idx = 0
