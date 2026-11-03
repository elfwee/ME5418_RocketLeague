import numpy as np

from src.config.gym_config import BentjhGymConfig
from src.sim.core.simulation import Simulation


def calculate_unit_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-8)


class GymReward:
    def __init__(self, sim: Simulation) -> None:
        """
        - sim: Simulation
        """
        self.sim = sim
        self.target_goal = np.array(self.sim.arena.goal_right_center, dtype=np.float32)
        self.reset()

    def reset(self):
        pass

    def calculate_reward(self):
        """
        Ball moving towards the opponent goal:
        dot product of the unit vector from the ball to the target goal and the ball velocity (m/s),
        plus BentjhGymConfig.score_gain on the step blue scores (minus it if orange scores).
        """
        state = self.sim.get_state()
        ball_pos = np.array(state["ball"]["position"], dtype=np.float32)
        ball_vel = np.array(state["ball"]["velocity"], dtype=np.float32)

        ball2goal = self.target_goal - ball_pos
        progress_reward = float(np.dot(calculate_unit_vector(ball2goal), ball_vel))

        score_reward = {"blue": 1.0, "orange": -1.0}.get(state["goal_scored_step"], 0.0)
        score_reward *= BentjhGymConfig.score_gain

        return progress_reward + score_reward
