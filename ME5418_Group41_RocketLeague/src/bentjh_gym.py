import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import pygame
from src.config import (
    SIM_HZ,
    FIELD_WIDTH,
    FIELD_HEIGHT,
    GOAL_DEPTH,
    CAR_MAX_SPEED,
    CAR_MAX_ANGULAR_SPEED,
    BALL_MAX_SPEED,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SIM_HZ,
    ENABLE_ORANGE_CAR,
)
from src.core.actions import CarAction
from src.core.simulation import Simulation
from src.visualization.renderer import Renderer

import gymnasium as gym

local_config = {
    "rl_id": "bentjh01_2d_rocketleague_v1", 
    "max_steps": 10000, 
    "action_gain": 1.0,
    "progress_gain":5.0,
    "score_gain":10.0
                }

if local_config["rl_id"] not in gym.envs.registry:
    gym.register(
        id=local_config["rl_id"],
        entry_point="src.bentjh_gym:RocketLeagueSideSwipeGym",
        max_episode_steps=local_config["max_steps"],
    )


# Define the wrapper to swap 'truncated' for 'terminated'
class TerminateOnTimeout(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If the 6000 step limit is hit, force terminated to True
        if truncated:
            terminated = True

        return obs, reward, terminated, truncated, info


def training_viz(sim: Simulation):
    """Launch the Pygame interactive sandbox with 2D vector controls and Octane physics."""
    pygame.init()

    pygame.display.set_caption("Rocket League Sideswipe 2D - Suspension Physics Sandbox")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    renderer = Renderer(screen)
    renderer.render(sim)
    pygame.display.flip()
    clock.tick(SIM_HZ)

    pygame.quit()


class RocketLeagueSideSwipeGym(gym.Env):
    meta_data = {"render_mode": ["human", "terminal"], "render_fps": SIM_HZ, "bot_type": ["none", "level_3"]}

    def __init__(self, render_mode: str = "terminal", bot_type: str = "zeros", car_type:str = "manual"):
        super().__init__()

        # Setting up simulation
        self.bot_type = bot_type
        if bot_type == "none":
            self.sim = Simulation(enable_orange=False)
        elif bot_type == "level_3":
            self.sim = Simulation(enable_orange=True)
        self.car_type = car_type
        self.render_mode = render_mode
        self.previous_score = {"blue": 0, "orange": 0}
        self.prev_action = (0, 0, 0, 0)
        self.step_count = 0
        self.is_open = True
        if render_mode == "human":
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(screen)

        # Setting up Environment
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        # select x-direction, y-direction, jump, and boost
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 2])  # size = 36
        # select one of 9 possible directions, jump, and boost
        # self.action_space = gym.spaces.MultiDiscrete([9,2,2]) # size = 36
        # press any of 6 buttons (left, right, up, down, jump, boost)
        # self.action_space = gym.spaces.MultiBinary(6) # size = 64
        # has redundant actions e.g. up + down. This makes the probability of choosing a valid action lower
        self.reward_range = (-1, 1)
        self.info = {}
        # self.observed_state =

    def calculate_reward(self, state, action):
        """
        takes the sate from sim and returns a reward
        """
        # score reward = [-1, 1]
        score_reward = (state["score"]["blue"] - self.previous_score["blue"]) - (
            state["score"]["orange"] - self.previous_score["orange"]
        )
        score_reward = local_config["score_gain"] * score_reward

        # Action reward is to encourage moving towards ball
        car_vel = np.array(state["car"]["velocity"]) / max(np.linalg.norm(state["car"]["velocity"]),1)
        car2ball = np.array(state["ball"]["position"]) - np.array(state["car"]["position"])
        car2ball = car2ball / np.linalg.norm(car2ball)
        action_reward = np.dot(car_vel, car2ball)
        # print(f"car_vel: ({car_vel[0]:.1f}, {car_vel[1]:.1f}), car2ball: ({car2ball[0]:.1f}, {car2ball[1]:.1f}), reward: {action_reward:.1f}")
        action_reward = local_config["action_gain"] * action_reward

        # Progress is defined as the velocity of the ball towards the goal
        ball_vel = np.array(state["ball"]["velocity"]) / max(np.linalg.norm(state["ball"]["velocity"]),1)
        ball2goal = np.array(self.sim.arena.goal_right_center) - np.array(state["ball"]["position"])
        ball2goal = ball2goal / np.linalg.norm(ball2goal)
        progress_reward = np.dot(ball2goal, ball_vel)
        # print(f"ball_vel: ({ball_vel[0]:.1f}, {ball_vel[1]:.1f}), ball2goal: ({ball2goal[0]:.1f}, {ball2goal[1]:.1f}), reward: {progress_reward:.1f}")
        progress_reward = local_config["progress_gain"] * progress_reward

        return score_reward + action_reward + progress_reward

    def check_terminated(self, state):
        """
        takes the state from sim and returns True if any goal is scored, otherwise False
        """
        terminated = False
        if (
            state["score"]["blue"] != self.previous_score["blue"]
            or state["score"]["orange"] != self.previous_score["orange"]
        ):
            terminated = True
        self.previous_score["blue"] = state["score"]["blue"]
        self.previous_score["orange"] = state["score"]["orange"]
        return terminated

    def action_to_sim_action(self, action):
        """
        takes the action from the environment and returns the CarAction object that defines the controls of the car
        """
        x, y, jump, boost = action
        dir_x = {0: 0.0, 1: 1.0, 2: -1.0}[x]  # 0: neutral, 1: right, 2:left
        dir_y = {0: 0.0, 1: 1.0, 2: -1.0}[y]  # 0: neutral, 1: up, 2: down
        jump = {0: False, 1: True}[jump]  # 0:no jump, 1: jump
        boost = {0: False, 1: True}[boost]  # 0:no boost, 1: boost
        return CarAction(dir_x=dir_x, dir_y=dir_y, jump=jump, boost=boost).clamp()

    def sim_state_to_observation(self, state: dict) -> np.ndarray:
        """
        takes the sim state returns the pose and twist of the car, car_orange and ball as a np.ndarray with shape (18,)
        """
        obs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0]
        # car
        obs[0] = state["car"]["position"][0] / (FIELD_WIDTH + 2*GOAL_DEPTH) * 2 - 1
        obs[1] = state["car"]["position"][1] / FIELD_HEIGHT * 2 - 1
        obs[2] = state["car"]["angle"] / np.pi
        obs[2] = obs[2] % 2
        if 1.0 < abs(obs[2]):
            obs[2] = obs[2] - 2 * np.sign(obs[2])
        obs[3] = state["car"]["velocity"][0] / CAR_MAX_SPEED
        obs[4] = state["car"]["velocity"][1] / CAR_MAX_SPEED
        obs[5] = state["car"]["angular_velocity"] / CAR_MAX_ANGULAR_SPEED
        # print(f"car: [{obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}], [{obs[3]:.1f}, {obs[4]:.1f}], {obs[5]:.1f}")
        # car_orange
        if self.bot_type != "none":
            obs[6] = state["car_orange"]["position"][0] / (FIELD_WIDTH + 2*GOAL_DEPTH) * 2 - 1
            obs[7] = state["car_orange"]["position"][1] / FIELD_WIDTH * 2 - 1
            obs[8] = state["car_orange"]["angle"] / np.pi
            obs[8] = obs[8] % 2
            if 1.0 < abs(obs[8]):
                obs[8] = obs[8] - 2 * np.sign(obs[8])
            obs[9] = state["car_orange"]["velocity"][0] / CAR_MAX_SPEED
            obs[10] = state["car_orange"]["velocity"][1] /CAR_MAX_SPEED
            obs[11] = state["car_orange"]["angular_velocity"] / CAR_MAX_ANGULAR_SPEED
            # print(f"car_orange: [{obs[6]:.1f}, {obs[7]:.1f}, {obs[8]:.1f}], [{obs[9]:.1f}, {obs[10]:.1f}, {obs[11]:.1f}]")
        # ball
        obs[12] = state["ball"]["position"][0] / (FIELD_WIDTH + 2*GOAL_DEPTH) * 2 - 1
        obs[13] = state["ball"]["position"][1] / FIELD_HEIGHT * 2 - 1
        obs[14] = state["ball"]["velocity"][0] / BALL_MAX_SPEED
        obs[15] = state["ball"]["velocity"][1] / BALL_MAX_SPEED
        # print(f"ball: [{obs[12]:.1f}, {obs[13]:.1f}], [{obs[14]:.1f}, {obs[15]:.1f}]")

        #Score
        # obs.extend([state["score"]["blue"], state["score"]["orange"]])

        # TODO observation conversion not returning -1 to 1 conversion
        return np.asarray(obs, dtype=np.float32)

    def _get_observation(self):
        return self.observed_state

    def _get_info(self):
        return self.info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_score = {"blue": 0, "orange": 0}
        self.step_count = 0
        self.sim.reset(reset_scores=True)
        self.observed_state = self.sim_state_to_observation(self.sim.get_state())
        self.info = {}
        # self.render()
        return self.observed_state, self.info

    def step(self, action):
        """ """
        if not self.is_open:
            return self.observed_state, 0.0, False, True, {"is_success": False}

        # Apply Action
        car_action = self.action_to_sim_action(action)
        if self.car_type == "manual" and self.render_mode == "human":
            car_action = self.manual_action()
        # Updating State
        self.sim.step(car_action, 1.0 / SIM_HZ)
        self.step_count += 1
        state = self.sim.get_state()
        self.observed_state = self.sim_state_to_observation(state)

        reward = self.calculate_reward(state, action)
        terminated = self.check_terminated(state)
        self.info["is_success"] = True
        if terminated:
            self.info["goal_scored"] = self.step_count
        truncated = not self.is_open

        if self.render_mode == "human":
            self.render()

        self.prev_action = action
        self.step_count += 1

        return self.observed_state, reward, terminated, truncated, self.info

    def manual_action(self):
        car_action = CarAction()
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
        car_action.dir_x = dir_x
        car_action.dir_y = dir_y
        car_action.jump = bool(keys[pygame.K_SPACE])
        car_action.boost = bool(keys[pygame.K_LSHIFT])
        return car_action.clamp()

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

    gym_env = gym.make(id=local_config["rl_id"], render_mode=None, disable_env_checker=True)
    check_env(gym_env.unwrapped)
    print("Environment check passed. gym.make successful")


def test_gym(max_episode: int = 1000):
    """Run headless simulation loop for automated benchmark or verification."""
    if ENABLE_ORANGE_CAR:
        bot_type = "level_3"
    else:
        bot_type = "none"
    # Create Gym Environment
    gym_env = gym.make(id=local_config["rl_id"], render_mode="human", bot_type="none", car_type="manual")
    # gym_env = gym.make(id=local_config["rl_id"], render_mode="terminal", bot_type=bot_type)

    # Affinely (linearly) rescales a Box action space of the environment to within the range of [min_action, max_action].
    # gym_env = gym.wrappers.RescaleAction(gym_env, min_action=-1.0, max_action=1.0)

    # Normalises observation, rewards to make training more stable
    # gym_env = gym.wrappers.NormalizeReward(gym_env, gamma=0.99, epsilon=1e-8)
    print("OBSERVATION SPACE:")
    observed_state, info = gym_env.reset()
    # print(observed_state)
    print(gym_env.observation_space.sample())
    print("ACTION SPACE:")
    print(gym_env.action_space.sample())

    observed_state, info = gym_env.reset()
    total_reward = 0
    reward = 0
    min_reward = 0
    max_reward = 0
    count = 0
    while gym_env.unwrapped.is_open:
        if count % 120 == 0:
            action = gym_env.action_space.sample()
        # print(f"reward: {reward}")
        if reward < min_reward:
            min_reward = reward
            # print(f"min_reward: {min_reward}")
        if max_reward < reward:
            max_reward = reward
            # print(f"max_reward: {max_reward}")

        observed_state, reward, terminated, truncated, info = gym_env.step(action)
        total_reward += reward

        if not gym_env.unwrapped.is_open:
            break
        if terminated or truncated:
            print(reward)
            # print(f"Last Action:\n{action}\nRollout Reward:{total_reward}")
            gym_env.reset()
            pass
        count += 1
        # if count > max_episode:
        #     break


def main():
    # max_episode = int(sys.argv[1])
    # test_gym(max_episode)
    test_gym()


if __name__ == "__main__":
    main()
