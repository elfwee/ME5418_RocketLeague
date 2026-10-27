import numpy as np
import pygame

from ME5418_RocketLeague.config.sim_config import (
    FIELD_HEIGHT,
    FIELD_WIDTH,
    GOAL_DEPTH,
    CAR_MAX_SPEED,
    CAR_MAX_ANGULAR_SPEED,
    BALL_MAX_SPEED,
    CAR_MAX_BOOST,
)
from ME5418_RocketLeague.simulation.actions import CarAction
from ME5418_RocketLeague.simulation.simulation import Simulation


def gym_action_to_sim_action(action: tuple) -> CarAction:
    """
    takes the action from the environment and returns the CarAction object that defines the controls of the car
    """
    x, y, jump, boost = action
    dir_x = {0: 0.0, 1: 1.0, 2: -1.0}[x]  # 0: neutral, 1: right, 2:left
    dir_y = {0: 0.0, 1: 1.0, 2: -1.0}[y]  # 0: neutral, 1: up, 2: down
    jump = {0: False, 1: True}[jump]  # 0:no jump, 1: jump
    boost = {0: False, 1: True}[boost]  # 0:no boost, 1: boost
    return CarAction(dir_x=dir_x, dir_y=dir_y, jump=jump, boost=boost).clamp()


def sim_state_to_gym_observation(sim_state: dict) -> np.ndarray:
    """
    takes the sim_state returns the pose and twist of the car, car_orange and ball as a np.ndarray with shape (18,)
    """
    obs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # car
    obs[0] = sim_state["car"]["position"][0] / (FIELD_WIDTH + 2 * GOAL_DEPTH) * 2 - 1
    obs[1] = sim_state["car"]["position"][1] / FIELD_HEIGHT * 2 - 1
    obs[2] = sim_state["car"]["angle"] / np.pi
    obs[2] = obs[2] % 2
    if 1.0 < abs(obs[2]):
        obs[2] = obs[2] - 2 * np.sign(obs[2])
    obs[3] = sim_state["car"]["velocity"][0] / CAR_MAX_SPEED
    obs[4] = sim_state["car"]["velocity"][1] / CAR_MAX_SPEED
    obs[5] = sim_state["car"]["angular_velocity"] / CAR_MAX_ANGULAR_SPEED
    obs[6] = sim_state["car"]["boost"] / CAR_MAX_BOOST
    # print(f"car: [{obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}], [{obs[3]:.1f}, {obs[4]:.1f}], {obs[5]:.1f}")
    # car_orange
    if "car_orange" in sim_state.keys():
        obs[7] = sim_state["car_orange"]["position"][0] / (FIELD_WIDTH + 2 * GOAL_DEPTH) * 2 - 1
        obs[8] = sim_state["car_orange"]["position"][1] / FIELD_HEIGHT * 2 - 1
        obs[9] = sim_state["car_orange"]["angle"] / np.pi
        obs[9] = obs[9] % 2
        if 1.0 < abs(obs[9]):
            obs[9] = obs[9] - 2 * np.sign(obs[9])
        obs[10] = sim_state["car_orange"]["velocity"][0] / CAR_MAX_SPEED
        obs[11] = sim_state["car_orange"]["velocity"][1] / CAR_MAX_SPEED
        obs[12] = sim_state["car_orange"]["angular_velocity"] / CAR_MAX_ANGULAR_SPEED
        # print(f"car_orange: [{obs[6]:.1f}, {obs[7]:.1f}, {obs[8]:.1f}], [{obs[9]:.1f}, {obs[10]:.1f}, {obs[11]:.1f}]")
    # ball
    obs[13] = sim_state["ball"]["position"][0] / (FIELD_WIDTH + 2 * GOAL_DEPTH) * 2 - 1
    obs[14] = sim_state["ball"]["position"][1] / FIELD_HEIGHT * 2 - 1
    obs[15] = sim_state["ball"]["velocity"][0] / BALL_MAX_SPEED
    obs[16] = sim_state["ball"]["velocity"][1] / BALL_MAX_SPEED
    # print(f"ball: [{obs[12]:.1f}, {obs[13]:.1f}], [{obs[14]:.1f}, {obs[15]:.1f}]")

    # Score
    # obs.extend([state["score"]["blue"], sim_state["score"]["orange"]])
    return np.asarray(obs, dtype=np.float32)


def sim_action_to_gym_action(sim_action: CarAction):
    act = [0, 0, 0, 0]
    act[0] = {-1.0: 2, 0.0: 0, 1.0: 1}[sim_action.dir_x]
    act[1] = {-1.0: 2, 0.0: 0, 1.0: 1}[sim_action.dir_y]
    act[2] = {True: 1, False: 0}[sim_action.jump]
    act[3] = {True: 1, False: 0}[sim_action.boost]

    return np.asarray(act, dtype=np.float32)


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
