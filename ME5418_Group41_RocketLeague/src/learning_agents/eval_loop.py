"""Greedy evaluation episodes, run summaries and GIF export."""

import itertools
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.config.sim_config import GIF_FRAME_SIZE
from src.learning_agents.env_factory import obs_to_tensor, scored_goal

MAX_GIF_FRAMES = 1200


def resolve_eval_episodes(num_episodes: int | None, render: bool, save_gif: bool, default: int) -> int | None:
    """Episodes to evaluate: unlimited (None) only when watching a rendered window with no limit given."""
    if render and num_episodes is None and not save_gif:
        return None
    return num_episodes if num_episodes is not None else default


def summarize_episodes(env: gym.Env, rewards: Sequence[float], goals: Sequence[int]) -> dict[str, float]:
    """Run summary returned by every train/eval entry point."""
    return {
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "goal_rate": float(np.mean(goals)) * 100.0 if goals else 0.0,
        "score_blue": int(env.unwrapped.sim.score_blue),
        "score_orange": int(env.unwrapped.sim.score_orange),
    }


def run_eval_episodes(
    env: gym.Env,
    select_blue: Callable[[torch.Tensor], object],
    select_orange: Callable[[torch.Tensor], object] | None = None,
    *,
    num_episodes: int | None,
    gif_path: str | Path | None = None,
    gif_tag: str = "Eval",
) -> dict[str, float]:
    """Play greedy evaluation episodes and return the run summary.

    `select_blue` / `select_orange` map an observation tensor to an env action. `select_orange` is
    only used when the env reports an orange observation (`control_car="both"`). When `gif_path`
    is set, the first episode is recorded and saved as a GIF.
    """
    rewards_per_episode = []
    goals_per_episode = []
    gif_frames = []
    recording = gif_path is not None

    episode_iter = range(num_episodes) if num_episodes is not None else itertools.count()
    for episode in episode_iter:
        state, reset_info = env.reset()
        state = obs_to_tensor(state)
        state_orange = None
        if select_orange is not None and "obs_orange" in reset_info:
            state_orange = obs_to_tensor(reset_info["obs_orange"])

        if not env.unwrapped.isopen:
            break

        terminated = False
        truncated = False
        episode_reward = 0.0
        goal = 0

        while not (terminated or truncated):
            action = select_blue(state)
            action_orange = select_orange(state_orange) if state_orange is not None else None

            next_state, reward, terminated, truncated, info = env.step(action, action_orange)

            if recording and len(gif_frames) < MAX_GIF_FRAMES:
                frame = env.unwrapped.get_frame()
                if frame is not None:
                    gif_frames.append(frame)

            episode_reward += reward
            if scored_goal(info, terminated, reward):
                goal = 1

            state = obs_to_tensor(next_state)
            if select_orange is not None and "obs_orange" in info:
                state_orange = obs_to_tensor(info["obs_orange"])

            if not env.unwrapped.isopen:
                break

        rewards_per_episode.append(episode_reward)
        goals_per_episode.append(goal)

        if recording and len(gif_frames) > 0:
            save_gif(gif_frames, gif_path, tag=gif_tag)
            recording = False

        print(
            f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
            f"Score(Blue-Orange) = {int(env.unwrapped.sim.score_blue)}-{int(env.unwrapped.sim.score_orange)} | "
            f"Goal = {bool(goal)} | Terminated = {terminated} | Truncated = {truncated}"
        )

    return summarize_episodes(env, rewards_per_episode, goals_per_episode)


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 15, tag: str = "Eval"):
    """Compile and save a list of RGB frames as an optimized animated GIF."""
    from PIL import Image

    if not frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Downscale by 50% (GIF_FRAME_SIZE) for smooth playback and lightweight GIF file size
    pil_images = [Image.fromarray(f).resize(GIF_FRAME_SIZE, Image.Resampling.BILINEAR) for f in frames]
    pil_images[0].save(
        str(path),
        save_all=True,
        append_images=pil_images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"[{tag} Eval] Saved evaluation GIF ({len(pil_images)} frames, {fps} FPS) to: {path}")
