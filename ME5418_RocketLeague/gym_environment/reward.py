from math import dist

import numpy as np
from ME5418_RocketLeague.config.gym_config import GymConfig
from ME5418_RocketLeague.config.sim_config import FIELD_HEIGHT, FIELD_WIDTH, GOAL_DEPTH
from ME5418_RocketLeague.simulation.simulation import Simulation


def calculate_unit_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-8)


class GymReward:
    def __init__(self, sim: Simulation) -> None:
        """
        - sim: Simulation
        """
        self.sim = sim

    def reset(self):
        pass

    def calculate_reward(self):
        """
        Calculates a potential-based, alignment-aware shaped reward.
        State dict requirements:
        state['car']['position'], state['car']['velocity']
        state['ball']['position'], state['ball']['velocity']
        state['score']['blue'], state['score']['orange']
        """

        eps = 1e-8
        state = self.sim.get_state()
        # Extract kinematics as NumPy arrays
        car_pos = np.array(state["car"]["position"], dtype=np.float32)
        car_vel = np.array(state["car"]["velocity"], dtype=np.float32)
        ball_pos = np.array(state["ball"]["position"], dtype=np.float32)
        ball_vel = np.array(state["ball"]["velocity"], dtype=np.float32)
        target_goal = np.array(self.sim.arena.goal_right_center, dtype=np.float32)
        own_goal = np.array(self.sim.arena.goal_left_center, dtype=np.float32)

        # ------------------------------------------------------------------
        # 1. Sparse Event Reward (Delta Score)
        # ------------------------------------------------------------------
        score_reward = state["score"]["blue"] - state["score"]["orange"]
        score_reward = GymConfig.score_gain * score_reward

        # ------------------------------------------------------------------
        # 2. Agent Moving Towards Ball
        # ------------------------------------------------------------------
        # Action reward is to encourage moving towards ball
        # dot product of the velocity of the car and the vector of the car to the ball
        car2ball = ball_pos - car_pos
        action_reward = np.dot(car_vel, calculate_unit_vector(car2ball))
        # print(f"car_vel: ({car_vel[0]:.1f}, {car_vel[1]:.1f}), car2ball: ({car2ball[0]:.1f}, {car2ball[1]:.1f}), reward: {action_reward:.1f}")
        action_reward = GymConfig.action_gain * action_reward

        # ------------------------------------------------------------------
        # 3. Ball Moving Towards Goal
        # ------------------------------------------------------------------
        ball2goal = target_goal - ball_pos
        progress_reward = np.dot(calculate_unit_vector(ball2goal), ball_vel)
        # progress_reward = max(progress_reward, 0.0)
        # print(f"ball_vel: ({ball_vel[0]:.1f}, {ball_vel[1]:.1f}), ball2goal: ({ball2goal[0]:.1f}, {ball2goal[1]:.1f}), reward: {progress_reward:.1f}")
        progress_reward = GymConfig.progress_gain * progress_reward

        # ------------------------------------------------------------------
        # 4. Agent Facing the Ball
        # ------------------------------------------------------------------
        car_orientation = state["car"]["angle"]
        car_orientation = car_orientation / np.pi
        car_orientation = car_orientation % 2
        if 1.0 < abs(car_orientation):
            car_orientation = car_orientation - 2 * np.sign(car_orientation)
        car_orientation = car_orientation * np.pi
        car_orientaion_vector = (np.cos(car_orientation), np.sin(car_orientation))
        orientation_reward = np.dot(car2ball, car_orientaion_vector)
        orientation_reward = GymConfig.orientation_gain * orientation_reward

        # ------------------------------------------------------------------
        # Ball Near Goal
        # ------------------------------------------------------------------
        y_ball_reward = ball_pos[1] / FIELD_HEIGHT
        y_ball_reward *= 1.0

        x_ball_reward = (ball_pos[0] / (FIELD_WIDTH + 2 * GOAL_DEPTH) - 0.5) * 2
        x_ball_reward *= 1.0

        dist2goal = np.linalg.norm(ball2goal)
        reward = 1 / dist2goal**2

        time_penalty = GymConfig.time_penalty

        # return x_ball_reward + y_ball_reward + score_reward
        return reward + score_reward + action_reward
        # return score_reward + action_reward + progress_reward + orientation_reward + time_penalty
        # return near_ball + 1 / np.linalg.norm(car2ball) + action_reward
