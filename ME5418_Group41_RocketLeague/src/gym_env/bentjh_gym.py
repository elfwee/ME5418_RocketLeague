import numpy as np
import pygame

import gymnasium as gym

from src.config.gym_config import BentjhGymConfig
from src.config.sim_config import SIM_HZ, SCREEN_WIDTH, SCREEN_HEIGHT

from src.sim.core.simulation import Simulation
from src.sim.core.control import CarControl
from src.sim.visualization.renderer import Renderer
from src.gym_env.gif_saver import GifSaver
from src.gym_env.rocket_league_env import decode_action
from src.gym_env.sim2gym_layer import sim_state_to_gym_observation, sim_action_to_gym_action
from src.gym_env.reward import GymReward

if BentjhGymConfig.gym_id not in gym.envs.registry:
    gym.register(
        id=BentjhGymConfig.gym_id,
        entry_point="src.gym_env.bentjh_gym:RocketLeagueSideSwipeGym",
        max_episode_steps=BentjhGymConfig.max_steps,
    )


class RocketLeagueSideSwipeGym(gym.Env):
    metadata = {
        "render_modes": ["human", "terminal"],
        "render_fps": SIM_HZ,
        "bot_type": ["none", "heuristic_bot", "random", "human"],
    }

    def __init__(self, render_mode=None, bot_type: str = "none", gif_path=None):
        """
        Args:
            render_mode: None, "terminal" or "human".
            bot_type: Orange opponent: "none", "heuristic_bot", "random" or "human".
            gif_path: If set, each episode is recorded and saved as a GIF to this path when a goal
                is scored. May contain "{episode}". Works with or without render_mode="human".
        """
        super().__init__()
        if bot_type not in self.metadata["bot_type"]:
            raise ValueError(f"Invalid bot_type: {bot_type}. Must be one of {self.metadata['bot_type']}.")

        # Setting up simulation
        self.sim = Simulation(enable_orange=(bot_type != "none"))
        control_type = "passive" if bot_type == "none" else bot_type
        self.adversary = CarControl(control_type=control_type, team="orange", sim=self.sim)
        self.render_mode = render_mode
        self.step_count = 0
        self.isopen = True

        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Rocket League Sideswipe 2D (bentjh)")
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(screen)

        self.gif_path = gif_path
        self.gif_saver = GifSaver() if gif_path else None
        self.episode = 0
        if self.gif_saver is not None and self.render_mode != "human":
            # Offscreen surface: frames are rendered without opening a window
            pygame.init()
            self.renderer = Renderer(pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)))

        # Setting up Environment
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
        # direction (9) x jump (2) x boost (2), decoded by rocket_league_env.decode_action
        self.action_space = gym.spaces.Discrete(36)
        self.gym_reward = GymReward(self.sim)

    @staticmethod
    def observe(sim: Simulation) -> np.ndarray:
        """Observation of ``sim`` as this env sees it (used by the agent to play on any sim)."""
        return sim_state_to_gym_observation(sim.get_state())

    def _get_obs(self):
        return self.observe(self.sim)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.sim.reset(reset_scores=True)
        self.adversary.reset()
        self.gym_reward.reset()
        self.observed_state = self._get_obs()
        if self.gif_saver is not None:
            self.episode += 1
            self.gif_saver.clear()
            self._capture_frame(force=True)
        return self.observed_state, {}

    def step(self, action):
        if not self.isopen:
            return self.observed_state, 0.0, False, True, {}

        # Apply Action
        self.sim.step(decode_action(action), 1.0 / SIM_HZ, action_orange=self.adversary.get_action())
        self.step_count += 1

        # Updating State
        sim_state = self.sim.get_state()
        self.observed_state = sim_state_to_gym_observation(sim_state)
        reward = self.gym_reward.calculate_reward()

        terminated = sim_state["score"]["blue"] != sim_state["score"]["orange"]
        truncated = self.step_count >= BentjhGymConfig.max_steps
        info = {}
        if terminated:
            info["goal_scored"] = self.step_count

        if self.render_mode is not None:
            self.render()

        if self.gif_saver is not None:
            if self.sim.goal_scored_this_step is not None:
                # The sim resets to kickoff in the same step a goal is scored, so the last
                # buffered frame is the goal.
                path = self.gif_path.format(episode=self.episode)
                if self.gif_saver.save(path):
                    info["gif_path"] = path
                self.gif_saver.clear()
            else:
                self._capture_frame()

        return self.observed_state, reward, terminated, truncated or not self.isopen, info

    def _capture_frame(self, force=False):
        if not self.isopen:
            return
        if self.render_mode != "human":
            self.renderer.render(self.sim)  # human mode already drew this frame in render()
        self.gif_saver.capture(self.renderer.screen, force=force)

    def render(self):
        if self.render_mode is None:
            return
        elif self.render_mode == "terminal":
            print(f"Step: {self.step_count}")
            print("Observed State:")
            print(self.observed_state)
        elif self.render_mode == "human":
            if not self.isopen:
                return
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.isopen = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.sim.reset()
                    elif event.key == pygame.K_b:
                        self.sim.set_orange_enabled(not self.sim.enable_orange, is_bot=True)
                    elif event.key == pygame.K_l:
                        # Toggle 45-degree boundary raycasts on/off
                        self.renderer.show_raycasts = not self.renderer.show_raycasts
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.renderer.handle_click(event.pos)

            self.renderer.render(self.sim)
            pygame.display.flip()
            self.clock.tick(SIM_HZ)

    def close(self):
        self.isopen = False
        if (self.render_mode == "human" or self.gif_saver is not None) and pygame.get_init():
            pygame.quit()


def check_environment():
    from gymnasium.utils.env_checker import check_env

    gym_env = gym.make(id=BentjhGymConfig.gym_id, render_mode=None, disable_env_checker=True)
    check_env(gym_env.unwrapped)
    print("Environment check passed. gym.make successful")


def test_gym(render_mode: str = "human", bot_type: str = "none"):
    """Drive the blue car with the keyboard (human mode) and print the shaped reward."""
    gym_env = gym.make(id=BentjhGymConfig.gym_id, render_mode=render_mode, bot_type=bot_type)
    car_control = CarControl(control_type="human" if render_mode == "human" else "passive", team="blue")

    # Normalises rewards to make training more stable
    if BentjhGymConfig.normalize_reward:
        gym_env = gym.wrappers.NormalizeReward(gym_env, gamma=0.99, epsilon=1e-8)
    print("OBSERVATION SPACE:")
    observed_state, info = gym_env.reset()
    print(observed_state)
    print("ACTION SPACE:")
    print(gym_env.action_space.sample())

    while gym_env.unwrapped.isopen:
        action = sim_action_to_gym_action(car_control.get_action())
        observed_state, reward, terminated, truncated, info = gym_env.step(action)
        print(f"{reward:0.1f}")

        if terminated or truncated:
            print("Terminated")
            gym_env.reset()
    gym_env.close()
