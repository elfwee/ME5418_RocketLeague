import numpy as np

from src.config.gym_config import GymConfig
from src.sim.core.actions import CarAction


def _car_features(car: dict) -> list:
    """[pos_x, pos_y, sin, cos, vel_x, vel_y, ang_vel] from a get_state_norm() car dict."""
    sin, cos = car["orientation"]
    return [
        car["position"][0],
        car["position"][1],
        sin,
        cos,
        car["velocity"][0],
        car["velocity"][1],
        car["angular_velocity"],
    ]


def sim_state_to_gym_observation(state_norm: dict) -> np.ndarray:
    """
    takes the normalised state from sim.get_state_norm() and returns the pose and twist of the blue car,
    orange car and ball as a np.ndarray with shape (19,), every value in [-1, 1]

    layout: blue car [pos_x, pos_y, sin, cos, vel_x, vel_y, ang_vel, boost] (8),
            orange car [pos_x, pos_y, sin, cos, vel_x, vel_y, ang_vel] (7, zeros if absent),
            ball [pos_x, pos_y, vel_x, vel_y] (4)
    """
    blue = state_norm["blue_car"]
    obs = _car_features(blue) + [blue["boost"]]
    orange = state_norm.get("car_orange")
    obs += _car_features(orange) if orange is not None else [0.0] * 7
    ball = state_norm["ball"]
    obs += [ball["position"][0], ball["position"][1], ball["velocity"][0], ball["velocity"][1]]
    return np.asarray(obs, dtype=np.float32)


def sim_action_to_gym_action(sim_action: CarAction) -> int:
    """Inverse of rocket_league_env.decode_action: CarAction -> Discrete(36) index."""
    direction = GymConfig.directions.index((int(np.sign(sim_action.dir_x)), int(np.sign(sim_action.dir_y))))
    return direction * 4 + int(bool(sim_action.jump)) * 2 + int(bool(sim_action.boost))
