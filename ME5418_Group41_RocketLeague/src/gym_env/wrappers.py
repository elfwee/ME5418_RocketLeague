"""Env wrappers."""

import gymnasium as gym

from src.gym_env.action_codec import clear_jump
from src.gym_env.rocket_league_env import Frame


class ActionRepeat(gym.Wrapper):
    """Hold each decision for `repeat` sim frames (frame skip) and sum the reward.

    The jump bit is released after the first frame, so the edge-triggered jump in `Car._apply_jump`
    can fire again on the next decision. Only the last frame builds an observation (unless the
    episode ends early, in which case the env is asked for one).

    `step(action, action_orange=None)`: with `control_car="both"`, Orange's action goes in
    `action_orange` and Orange's observation comes back in `info["obs_orange"]`.
    """

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        self.repeat = max(1, int(repeat))

    def step(self, action, action_orange=None):
        held_action = clear_jump(action)
        held_orange = clear_jump(action_orange) if action_orange is not None else None

        total_reward = 0.0
        built_obs = False
        for frame_idx in range(self.repeat):
            first = frame_idx == 0
            built_obs = frame_idx == self.repeat - 1
            frame = Frame(
                action=action if first else held_action,
                action_orange=action_orange if first else held_orange,
                build_obs=built_obs,
            )
            obs, reward, terminated, truncated, info = self.env.step(frame)
            total_reward += float(reward)
            if terminated or truncated or not self.env.unwrapped.isopen:
                break

        if not built_obs:
            obs = self.env.unwrapped.get_observation()
            if self.env.unwrapped.control_car == "both":
                info["obs_orange"] = self.env.unwrapped.get_observation("orange")
        return obs, total_reward, terminated, truncated, info
