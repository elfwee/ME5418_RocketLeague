from pathlib import Path

import pygame
import numpy as np
from src.config.gym_config import GymConfig
from src.config.nn_config import PPOConfig
from src.sim.core.actions import CarAction
from src.sim.ai.heuristic_bot import HeuristicBot
from src.sim.core.simulation import Simulation
from src.sim.core.orange_perspective import get_orange_perspective_obs, mirror_action
from typing import Dict, Any, Optional, Tuple


class CarControl:
    """Controller wrapper dispatching car actions by control source.

    - control_type:
        - passive: returns empty action
        - human: enables keyboard control
        - random: samples a random action every 0.5s
        - heuristic_bot: uses HeuristicBot (HZISM)
        - agent: greedy inference from a trained PPO / (Dueling) DQN checkpoint (no training)
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
        control_type: str = "none",
        team: str = "blue",
        sim: Optional[Simulation] = None,
        model_path: Optional[str | Path] = None,
        action_repeat: int = PPOConfig.action_repeat,
    ):
        self.control_type = control_type
        self.sim = sim
        self.team = team
        self.bot = HeuristicBot(team=self.team) if self.control_type == "heuristic_bot" else None
        self.action = CarAction()

        self.policy = None
        self.action_repeat = max(1, action_repeat)
        self._agent_frame = 0
        self._agent_action_idx = 0
        if self.control_type == "agent":
            if self.team not in ("blue", "orange"):
                raise ValueError(f"Invalid team for agent: {self.team}. Must be 'blue' or 'orange'.")
            if self.sim is None or model_path is None:
                raise ValueError("control_type='agent' requires both sim and model_path.")
            self.policy = self._load_policy(model_path)

    @staticmethod
    def _load_policy(model_path: str | Path):
        """Load a PPO ActorCritic or (Dueling) DQN state_dict for CPU inference, inferring the
        architecture from the checkpoint's layer names and shapes."""
        import torch

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dim = state_dict["fc1.weight"].shape[1]
        hidden_dim = state_dict["fc1.weight"].shape[0]

        if "actor.weight" in state_dict:
            from src.neural_nets.ppo_torch import ActorCritic

            net = ActorCritic(state_dim, state_dict["actor.weight"].shape[0], hidden_dim)
        else:
            from src.neural_nets.dueling_ddqn_torch import DQN

            dueling = "advantages.weight" in state_dict
            out_key = "advantages.weight" if dueling else "fc3.weight"
            net = DQN(state_dim, state_dict[out_key].shape[0], hidden_dim, enable_dueling_dqn=dueling)

        net.load_state_dict(state_dict)
        net.eval()
        print(f"[CarControl] Loaded {type(net).__name__} agent from: {model_path}")
        return net

    @staticmethod
    def _decode_discrete_action(action_idx: int, allow_jump: bool = True) -> CarAction:
        """Map a Discrete(36) index to CarAction (same layout as RocketLeagueEnv.step)."""
        direction, jump, boost = action_idx // 4, (action_idx % 4) // 2, action_idx % 2
        dir_x, dir_y = GymConfig.directions[direction]
        return CarAction(
            dir_x=float(dir_x), dir_y=float(dir_y), jump=allow_jump and jump == 1, boost=boost == 1
        ).clamp()

    def _agent_action(self) -> CarAction:
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
                scores = out[0] if isinstance(out, tuple) else out  # PPO -> (logits, V); DQN -> Q
                self._agent_action_idx = int(scores.squeeze(0).argmax().item())
        self._agent_frame += 1
        action = self._decode_discrete_action(self._agent_action_idx, allow_jump=is_decision_frame)
        return mirror_action(action) if self.team == "orange" else action

    def get_action(self):
        if self.control_type == "passive":
            self.action = CarAction(dir_x=0, dir_y=0, jump=False, boost=False)

        elif self.control_type == "human":
            # gets keyboard input into CarAction
            keys = pygame.key.get_pressed()

            dir_x = 0.0
            dir_y = 0.0

            # Keyboard directional input: WASD or Arrow Keys
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                dir_x += 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                dir_x -= 1.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                dir_y += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                dir_y -= 1.0

            self.action.dir_x = dir_x
            self.action.dir_y = dir_y

            # Jump: Space bar or Gamepad Button A (button 0)
            self.action.jump = bool(keys[pygame.K_SPACE])

            # Rocket Boost: Shift, O, J, or Gamepad Button B / Trigger (buttons 1, 5)
            self.action.boost = bool(
                keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_o] or keys[pygame.K_j]
            )

        elif self.control_type == "random":
            if self.sim is not None and self.sim.match_time % 0.5 < 0.001:
                self.random_action = CarAction(
                    dir_x={0: 0.0, 1: 1.0, 2: -1.0}[np.random.randint(3)],
                    dir_y={0: 0.0, 1: 1.0, 2: -1.0}[np.random.randint(3)],
                    jump={0: False, 1: True}[np.random.randint(2)],
                    boost={0: False, 1: True}[np.random.randint(2)],
                )
            self.action = self.random_action

        elif self.control_type == "heuristic_bot":
            if self.bot is None:
                self.bot = HeuristicBot(team=self.team)
            self.action = self.bot.compute_action(self.sim)

        elif self.control_type == "agent":
            self.action = self._agent_action()

        return self.action

    def reset(self):
        self.action = CarAction(dir_x=0, dir_y=0, jump=False, boost=False)
        if self.control_type == "heuristic_bot" and self.bot is not None:
            self.bot.reset()
        self._agent_frame = 0
        self._agent_action_idx = 0
