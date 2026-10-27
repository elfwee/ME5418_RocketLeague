import numpy as np
import pygame

import gymnasium as gym

from ME5418_RocketLeague.config.gym_config import GymConfig
from ME5418_RocketLeague.config.sim_config import SIM_HZ, SCREEN_WIDTH, SCREEN_HEIGHT, FIELD_WIDTH, FIELD_HEIGHT

from ME5418_RocketLeague.simulation.simulation import Simulation
from ME5418_RocketLeague.simulation.control import CarControl
from ME5418_RocketLeague.visualization.renderer import Renderer
from ME5418_RocketLeague.gym_environment.sim2gym_layer import (
    gym_action_to_sim_action,
    sim_state_to_gym_observation,
    sim_action_to_gym_action,
)
from ME5418_RocketLeague.gym_environment.reward import GymReward

if GymConfig.gym_id not in gym.envs.registry:
    gym.register(
        id=GymConfig.gym_id,
        entry_point="ME5418_RocketLeague.gym_environment.bentjh_gym:RocketLeagueSideSwipeGym",
        max_episode_steps=GymConfig.max_steps,
    )


class RocketLeagueSideSwipeGym(gym.Env):
    metadata = {
        "render_modes": ["human", "terminal"],
        "render_fps": SIM_HZ,
        "adversary_type": ["none", "human", "level_3"],
    }

    def __init__(self, render_mode: str = "terminal", adversary_type: str = "none"):
        super().__init__()

        # Setting up simulation
        self.sim = Simulation(enable_orange=False)
        self.adversary = CarControl(control_type=adversary_type, team="orange", sim=self.sim)
        self.render_mode = render_mode
        self.step_count = 0
        self.is_open = True

        if render_mode == "human":
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(screen)

        # Setting up Environment
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
        # select x-direction, y-direction, jump, and boost
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 2])  # size = 36
        self.gym_reward = GymReward(self.sim)
        self.info = {}

    def _get_observation(self):
        return self.observed_state

    def _get_info(self):
        return self.info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.sim.reset(reset_scores=True)
        self.observed_state = sim_state_to_gym_observation(self.sim.get_state())
        info = {}
        return self.observed_state, info

    def step(self, action):
        """ """
        if not self.is_open:
            return self.observed_state, 0.0, False, True, {"is_success": False}

        # Apply Action
        self.sim.step(
            action_blue=gym_action_to_sim_action(action), action_orange=self.adversary.get_action(), dt=1.0 / SIM_HZ
        )
        # Updating State
        sim_state = self.sim.get_state()
        self.observed_state = sim_state_to_gym_observation(sim_state)

        reward = self.gym_reward.calculate_reward()

        terminated = True if abs(sim_state["score"]["blue"] - sim_state["score"]["orange"]) > 0 else False
        info = {}
        info["is_success"] = True
        if terminated:
            info["goal_scored"] = self.step_count
        truncated = True if self.step_count >= GymConfig.max_steps else False

        if self.render_mode == "human":
            self.render()

        self.step_count += 1

        return self.observed_state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == None:
            return
        elif self.render_mode == "terminal":
            print(f"Step: {self.step_count}")
            print("Observed State:")
            print(self.observed_state)
        elif self.render_mode == "human":
            if not self.is_open:
                return
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.is_open = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.sim.reset()
                    elif event.key == pygame.K_b:
                        if self.bot_type == "none":
                            self.bot_type == "level_3"
                            self.sim.set_orange_enabled(True, is_bot=True)
                        else:
                            self.bot_type = "none"
                            self.sim.set_orange_enabled(False, is_bot=True)
                    elif event.key == pygame.K_l:
                        # Toggle 45-degree boundary raycasts on/off
                        self.renderer.show_raycasts = not self.renderer.show_raycasts
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.renderer.handle_click(event.pos)
                    # --- Input Mapping: 2D Direction Vector ---

            self.renderer.render(self.sim)
            pygame.display.flip()
            self.clock.tick(SIM_HZ)

    def close(self):
        if self.is_open:
            self.is_open = False
        if self.render_mode == "human" and pygame.get_init():
            pygame.display.quit()
            pygame.quit()


def check_environment():
    from gymnasium.utils.env_checker import check_env

    gym_env = gym.make(id=GymConfig.gym_id, render_mode=None, disable_env_checker=True)
    check_env(gym_env.unwrapped)
    print("Environment check passed. gym.make successful")


def test_gym(render_mode: str = "human", adversary_type: str = "none"):
    """Run headless simulation loop for automated benchmark or verification."""
    # Create Gym Environment
    gym_env = gym.make(id=GymConfig.gym_id, render_mode=render_mode, adversary_type=adversary_type)
    if render_mode == "human":
        car_control = CarControl(control_type="human", team="blue")
    else:
        car_control = CarControl(control_type="none")

    # Affinely (linearly) rescales a Box action space of the environment to within the range of [min_action, max_action].
    # gym_env = gym.wrappers.RescaleAction(gym_env, min_action=-1.0, max_action=1.0)

    # Normalises observation, rewards to make training more stable
    if GymConfig.normalize_reward:
        gym_env = gym.wrappers.NormalizeReward(gym_env, gamma=0.99, epsilon=1e-8)
    print("OBSERVATION SPACE:")
    observed_state, info = gym_env.reset()
    print(observed_state)
    print("ACTION SPACE:")
    print(gym_env.action_space.sample())

    reward = 0
    count = 0
    while gym_env.unwrapped.is_open:
        action = sim_action_to_gym_action(car_control.get_action())
        observed_state, reward, terminated, truncated, info = gym_env.step(action)
        print(f"{reward:0.1f}")

        if terminated or truncated:
            print("Terminated")
            gym_env.reset()
        count += 1
        # if count > max_episode:
        #     break
