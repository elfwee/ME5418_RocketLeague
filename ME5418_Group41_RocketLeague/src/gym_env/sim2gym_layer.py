import numpy as np

from src.config.gym_config import GymConfig
from src.config.sim_config import (
    FIELD_HEIGHT,
    FIELD_WIDTH,
    GOAL_DEPTH,
    CAR_MAX_SPEED,
    CAR_MAX_ANGULAR_SPEED,
    BALL_MAX_SPEED,
    CAR_MAX_BOOST,
)
from src.sim.core.actions import CarAction


def sim_state_to_gym_observation(sim_state: dict) -> np.ndarray:
    """
    takes the sim_state returns the pose and twist of the blue car, orange car and ball as a np.ndarray with shape (17,)
    """
    obs = [0.0] * 17
    # blue car
    obs[0] = sim_state["blue_car"]["position"][0] / (FIELD_WIDTH + 2 * GOAL_DEPTH) * 2 - 1
    obs[1] = sim_state["blue_car"]["position"][1] / FIELD_HEIGHT * 2 - 1
    obs[2] = sim_state["blue_car"]["angle"] / np.pi
    obs[2] = obs[2] % 2
    if 1.0 < abs(obs[2]):
        obs[2] = obs[2] - 2 * np.sign(obs[2])
    obs[3] = sim_state["blue_car"]["velocity"][0] / CAR_MAX_SPEED
    obs[4] = sim_state["blue_car"]["velocity"][1] / CAR_MAX_SPEED
    obs[5] = sim_state["blue_car"]["angular_velocity"] / CAR_MAX_ANGULAR_SPEED
    obs[6] = sim_state["blue_car"]["boost"] / CAR_MAX_BOOST
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
    # ball
    obs[13] = sim_state["ball"]["position"][0] / (FIELD_WIDTH + 2 * GOAL_DEPTH) * 2 - 1
    obs[14] = sim_state["ball"]["position"][1] / FIELD_HEIGHT * 2 - 1
    obs[15] = sim_state["ball"]["velocity"][0] / BALL_MAX_SPEED
    obs[16] = sim_state["ball"]["velocity"][1] / BALL_MAX_SPEED
    return np.asarray(obs, dtype=np.float32)


def sim_action_to_gym_action(sim_action: CarAction) -> int:
    """Inverse of rocket_league_env.decode_action: CarAction -> Discrete(36) index."""
    direction = GymConfig.directions.index((int(np.sign(sim_action.dir_x)), int(np.sign(sim_action.dir_y))))
    return direction * 4 + int(bool(sim_action.jump)) * 2 + int(bool(sim_action.boost))
