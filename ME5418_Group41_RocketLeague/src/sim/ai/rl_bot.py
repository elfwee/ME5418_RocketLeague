"""RL Policy Bot controlling a vehicle using a trained PyTorch model (PPO or Dueling DDQN)."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config.gym_config import GymConfig
from src.config.nn_config import DQNConfig, PPOConfig
from src.neural_nets.dueling_ddqn_torch import DQN
from src.neural_nets.ppo_torch import ActorCritic
from src.sim.core.actions import CarAction

ROOT_DIR = Path(__file__).resolve().parents[3]


class RLBot:
    """Agent policy bot that queries a trained neural network to generate CarAction.

    Supports:
        - Agent types: 'ppo', 'dueling_ddqn' (or 'ddqn'), 'double_dqn', 'dqn'
        - Teams: 'blue' (attacks Right) or 'orange' (attacks Left, with observation & action mirroring)
        - Action repeat: 4 frames with edge-triggered jump release, matching training
        - Auto-detection of factored vs discrete action heads for PPO
    """

    def __init__(
        self,
        agent_type: str = "ppo",
        team: str = "blue",
        model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        action_repeat: int = 4,
    ):
        self.team = team.lower()
        self.agent_type = agent_type.lower()
        self.action_repeat = action_repeat
        self.repeat_counter = 0
        self.current_action = CarAction()
        self.is_factored = False

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve default checkpoint path if not explicitly provided
        if model_path is None:
            runs_dir = ROOT_DIR / "runs"
            if self.agent_type == "ppo":
                self.model_path = runs_dir / "model_ppo.pt"
            elif self.agent_type in ("dueling_ddqn", "ddqn"):
                self.model_path = runs_dir / "model_dueling_ddqn.pt"
            elif self.agent_type == "double_dqn":
                self.model_path = runs_dir / "model_double_dqn.pt"
            elif self.agent_type == "dqn":
                self.model_path = runs_dir / "model_dqn.pt"
            else:
                self.model_path = runs_dir / f"model_{self.agent_type}.pt"
        else:
            self.model_path = Path(model_path)

        num_states = 52
        num_actions = 36

        # Check if checkpoint exists and inspect keys
        state_dict = None
        if os.path.exists(self.model_path):
            state_dict = torch.load(str(self.model_path), map_location=self.device, weights_only=True)

        if self.agent_type == "ppo":
            # Auto-detect whether checkpoint was trained with factored MultiDiscrete or discrete heads
            if state_dict is not None and "actor_heads.0.weight" in state_dict:
                self.is_factored = True
                action_dims = PPOConfig.action_dims if hasattr(PPOConfig, "action_dims") else (9, 2, 2)
            elif getattr(PPOConfig, "factored_action", False):
                self.is_factored = True
                action_dims = PPOConfig.action_dims
            else:
                self.is_factored = False
                action_dims = 36

            self.model = ActorCritic(num_states, action_dims, PPOConfig.hidden_size).to(self.device)
        elif self.agent_type in ("dueling_ddqn", "ddqn"):
            self.model = DQN(num_states, num_actions, DQNConfig.hidden_size, enable_dueling_dqn=True).to(self.device)
        elif self.agent_type == "double_dqn":
            self.model = DQN(num_states, num_actions, DQNConfig.hidden_size, enable_dueling_dqn=False).to(self.device)
        elif self.agent_type == "dqn":
            self.model = DQN(num_states, num_actions, DQNConfig.hidden_size, enable_dueling_dqn=False).to(self.device)
        else:
            raise ValueError(f"Unsupported agent_type '{self.agent_type}'. Must be 'ppo', 'dueling_ddqn', or 'dqn'.")

        # Load weights if available
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            mode_str = f"factored={self.is_factored}" if self.agent_type == "ppo" else "discrete"
            print(f"[RLBot] Successfully loaded {self.agent_type.upper()} ({self.team.upper()} car, {mode_str}) from: {self.model_path}")
        else:
            print(f"[RLBot] Warning: Checkpoint '{self.model_path}' not found! Using uninitialized weights.")

        self.model.eval()

    def reset(self):
        """Reset step repeat counters upon kickoff."""
        self.repeat_counter = 0
        self.current_action = CarAction()

    def compute_action(self, sim) -> CarAction:
        """Query policy network every `action_repeat` frames and return CarAction."""
        if self.repeat_counter % self.action_repeat == 0:
            obs = sim.get_state_norm(as_flat_array=True, team=self.team)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                if self.agent_type == "ppo":
                    logits, _ = self.model(obs_tensor)
                    if self.is_factored:
                        # logits is a list of [direction (9), jump (2), boost (2)]
                        direction = int(logits[0].squeeze(0).argmax().item())
                        jump = int(logits[1].squeeze(0).argmax().item())
                        boost = int(logits[2].squeeze(0).argmax().item())
                    else:
                        act_idx = int(logits.squeeze(0).argmax().item())
                        direction = act_idx // 4
                        jump = (act_idx % 4) // 2
                        boost = act_idx % 2
                else:
                    q_values = self.model(obs_tensor)
                    act_idx = int(q_values.squeeze(0).argmax().item())
                    direction = act_idx // 4
                    jump = (act_idx % 4) // 2
                    boost = act_idx % 2

            dir_x, dir_y = GymConfig.directions[direction]

            # Invert horizontal steering if Orange car (since Orange attacks towards Left)
            if self.team == "orange":
                dir_x = -dir_x

            self.current_action = CarAction(
                dir_x=float(dir_x),
                dir_y=float(dir_y),
                jump=(jump == 1),
                boost=(boost == 1),
            )
        else:
            # Action repeat edge-trigger: release jump after frame 0 so double-jump can trigger cleanly
            if self.current_action.jump:
                self.current_action = CarAction(
                    dir_x=self.current_action.dir_x,
                    dir_y=self.current_action.dir_y,
                    jump=False,
                    boost=self.current_action.boost,
                )

        self.repeat_counter += 1
        return self.current_action
