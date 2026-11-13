"""Deterministic scenarios shared by `record_goldens.py` and the regression tests.

This module is the only place the tests touch the project's internal API to *produce* data. When
a refactor moves or renames something, update the adapter calls here; the recorded goldens in
`tests/golden/` must not change unless a behaviour change is intended and called out.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Tiny network width for the saved per-architecture checkpoints (keeps the goldens small).
TINY_HIDDEN = 16
OBS_DIM = 52


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scripted_actions(seed: int, n: int, hold: int, n_actions: int = 36) -> list[int]:
    """A repeatable action sequence: a fresh random index every `hold` frames."""
    rng = np.random.default_rng(seed)
    out = []
    current = 0
    for i in range(n):
        if i % hold == 0:
            current = int(rng.integers(n_actions))
        out.append(current)
    return out


# --------------------------------------------------------------------------------------------
# Adapters onto the current code (update these when the API moves)
# --------------------------------------------------------------------------------------------


def make_env(**kwargs):
    from src.gym_env.rocket_league_env import RocketLeagueEnv

    return RocketLeagueEnv(**kwargs)


def orange_view(env) -> np.ndarray:
    """The mirrored Orange-perspective observation of the env's current sim state (zeros without Orange)."""
    if env.sim.orange_car is None:
        return np.zeros(OBS_DIM, dtype=np.float32)
    return np.asarray(env.sim.get_state_norm_mirror(as_flat_array=True), dtype=np.float32)


def place_ball(env, x: float, y: float, vx: float, vy: float):
    env.sim.ball.reset(x, y, vx=vx, vy=vy)


def set_last_touch(env, team: str | None):
    env.sim.last_touch = team


def step_repeat(env, action, action_orange=None, repeat: int = 4):
    from src.gym_env.wrappers import ActionRepeat

    return ActionRepeat(env, repeat).step(action, action_orange)


def make_sim(**kwargs):
    from src.sim.core.simulation import Simulation

    return Simulation(**kwargs)


def make_bot_controller(team: str, sim):
    from src.sim.control import HeuristicController

    return HeuristicController(team, sim)


def bot_state(controller) -> str:
    return str(controller.bot.current_state)


def sim_obs(sim) -> tuple[list[float], list[float]]:
    return sim.get_state_norm(as_flat_array=True), sim.get_state_norm_mirror(as_flat_array=True)


def build_dqn(hidden: int, dueling: bool, state_dim: int = OBS_DIM, action_dim: int = 36):
    from src.neural_nets.dueling_ddqn_torch import DQN

    return DQN(state_dim, action_dim, hidden, enable_dueling_dqn=dueling)


def build_actor_critic(hidden: int, action_dim, state_dim: int = OBS_DIM):
    from src.neural_nets.ppo_torch import ActorCritic

    return ActorCritic(state_dim, action_dim, hidden)


def greedy_env_action(net, obs: np.ndarray):
    """Greedy env action of a DQN / ActorCritic: a Discrete index or [direction, jump, boost]."""
    from src.neural_nets.ppo_torch import ActorCritic

    state = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        if isinstance(net, ActorCritic):
            logits, _ = net(state.unsqueeze(0))
            if net.is_factored:
                return [int(l.squeeze(0).argmax().item()) for l in logits]
            return int(logits.squeeze(0).argmax().item())
        return int(net(state.unsqueeze(0)).squeeze(0).argmax().item())


# --------------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------------


def env_trajectory(
    *,
    control_car: str,
    bot_type: str,
    random_spawn: bool,
    seed: int,
    frames: int,
    action_seed: int,
    orange_action_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Step the env one frame at a time with scripted actions; resets after every goal."""
    env = make_env(bot_type=bot_type, control_car=control_car, random_spawn=random_spawn)
    actions = scripted_actions(action_seed, frames, hold=12)
    actions_orange = scripted_actions(orange_action_seed, frames, hold=9) if orange_action_seed is not None else None

    obs, info = env.reset(seed=seed)
    rec = {"obs": [], "obs_orange_view": [], "reward": [], "terminated": [], "goal": [], "reset_obs": [obs]}
    for i in range(frames):
        if actions_orange is not None:
            obs, reward, terminated, truncated, info = env.step(actions[i], actions_orange[i])
        else:
            obs, reward, terminated, truncated, info = env.step(actions[i])
        rec["obs"].append(obs)
        rec["obs_orange_view"].append(orange_view(env))
        rec["reward"].append(reward)
        rec["terminated"].append(terminated)
        goal = info.get("goal_scored")
        rec["goal"].append({None: 0, "blue": 1, "orange": 2}[goal])
        if terminated:
            obs, info = env.reset()
            rec["reset_obs"].append(obs)
    env.close()
    return {
        "obs": np.asarray(rec["obs"], dtype=np.float32),
        "obs_orange_view": np.asarray(rec["obs_orange_view"], dtype=np.float32),
        "reward": np.asarray(rec["reward"], dtype=np.float64),
        "terminated": np.asarray(rec["terminated"], dtype=bool),
        "goal": np.asarray(rec["goal"], dtype=np.int8),
        "reset_obs": np.asarray(rec["reset_obs"], dtype=np.float32),
    }


ENV_SCENARIOS = {
    # Training configuration: policy on Blue vs the heuristic bot, fixed kickoffs
    "env_blue_bot_fixed": dict(
        control_car="blue", bot_type="heuristic_bot", random_spawn=False, seed=0, frames=900, action_seed=42
    ),
    # Same with random spawns
    "env_blue_bot_random": dict(
        control_car="blue", bot_type="heuristic_bot", random_spawn=True, seed=7, frames=900, action_seed=43
    ),
    # Policy drives both cars (self-play / mixed eval)
    "env_both_fixed": dict(
        control_car="both",
        bot_type="none",
        random_spawn=False,
        seed=1,
        frames=900,
        action_seed=44,
        orange_action_seed=45,
    ),
    # Policy on Orange (mirrored observation and reward); Blue is idle without a keyboard
    "env_orange_random": dict(
        control_car="orange", bot_type="none", random_spawn=True, seed=11, frames=900, action_seed=46
    ),
    # Blue alone in the arena (opponent features zero-filled)
    "env_solo_random": dict(
        control_car="blue", bot_type="none", random_spawn=True, seed=13, frames=600, action_seed=47
    ),
}


# (control_car, ball x, ball y, ball vx, ball vy, last touch) -> one shot at a goal per case
GOAL_CASES = [
    ("blue", 28.0, 8.5, 25.0, 0.0, "blue"),  # Blue scores
    ("blue", 28.0, 8.5, 25.0, 0.0, "orange"),  # Orange own goal: ego scores, opponent touched last
    ("blue", 7.0, 8.5, -25.0, 0.0, "orange"),  # Blue concedes
    ("blue", 7.0, 8.5, -25.0, 0.0, None),  # Blue concedes, nobody touched
    ("orange", 7.0, 8.5, -25.0, 0.0, "orange"),  # Orange (ego) scores
    ("orange", 7.0, 8.5, -25.0, 0.0, "blue"),  # Blue own goal from Orange's perspective
    ("orange", 28.0, 8.5, 25.0, 0.0, "blue"),  # Orange concedes
    ("blue", 17.5, 3.0, 6.0, 9.0, "blue"),  # Ball lofted forward after an ego touch (progress reward)
    ("blue", 17.5, 3.0, -6.0, 9.0, "orange"),  # Ball lofted backward after an opponent touch
]


def goal_events_trajectory(frames_per_case: int = 45) -> dict[str, np.ndarray]:
    """Shoot the ball at a goal from a scripted state and record rewards until the goal (or timeout)."""
    rewards, obs, terminated, goals = [], [], [], []
    for i, (control_car, x, y, vx, vy, touch) in enumerate(GOAL_CASES):
        env = make_env(
            bot_type="heuristic_bot" if control_car == "blue" else "none", control_car=control_car, random_spawn=False
        )
        env.reset(seed=100 + i)
        place_ball(env, x, y, vx, vy)
        set_last_touch(env, touch)
        case_rewards = np.full(frames_per_case, np.nan)
        case_obs = np.zeros((frames_per_case, OBS_DIM), dtype=np.float32)
        case_goal = 0
        case_term = -1
        for f in range(frames_per_case):
            o, r, term, trunc, info = env.step(0)
            case_rewards[f] = r
            case_obs[f] = o
            if term:
                case_goal = {"blue": 1, "orange": 2}[info["goal_scored"]]
                case_term = f
                break
        env.close()
        rewards.append(case_rewards)
        obs.append(case_obs)
        terminated.append(case_term)
        goals.append(case_goal)
    return {
        "reward": np.asarray(rewards),
        "obs": np.asarray(obs),
        "terminated_at": np.asarray(terminated),
        "goal": np.asarray(goals, dtype=np.int8),
    }


def action_repeat_trajectory(*, discrete: bool, seed: int, decisions: int, repeat: int = 4) -> dict[str, np.ndarray]:
    """Decision-level trajectory through the action-repeat path used by training and eval."""
    env = make_env(bot_type="heuristic_bot", control_car="blue", random_spawn=True, discrete_action=discrete)
    flat = scripted_actions(seed, decisions, hold=3)
    obs, _ = env.reset(seed=seed)
    rec = {"obs": [], "reward": [], "terminated": []}
    for i in range(decisions):
        a = flat[i]
        action = a if discrete else [a // 4, (a % 4) // 2, a % 2]
        obs, reward, terminated, truncated, info = step_repeat(env, action, repeat=repeat)
        rec["obs"].append(obs)
        rec["reward"].append(reward)
        rec["terminated"].append(terminated)
        if terminated:
            obs, _ = env.reset()
    env.close()
    return {
        "obs": np.asarray(rec["obs"], dtype=np.float32),
        "reward": np.asarray(rec["reward"], dtype=np.float64),
        "terminated": np.asarray(rec["terminated"], dtype=bool),
    }


def action_repeat_both_trajectory(*, seed: int, decisions: int, repeat: int = 4) -> dict[str, np.ndarray]:
    """Action repeat with the policy on both cars (the `--eval --mix` path)."""
    env = make_env(bot_type="heuristic_bot", control_car="both", random_spawn=True)
    blue = scripted_actions(seed, decisions, hold=2)
    orange = scripted_actions(seed + 100, decisions, hold=3)
    obs, info = env.reset(seed=seed)
    rec = {"obs": [], "obs_orange": [], "reward": []}
    for i in range(decisions):
        obs, reward, terminated, truncated, info = step_repeat(env, blue[i], orange[i], repeat=repeat)
        rec["obs"].append(obs)
        rec["obs_orange"].append(info["obs_orange"])
        rec["reward"].append(reward)
        if terminated:
            obs, info = env.reset()
    env.close()
    return {
        "obs": np.asarray(rec["obs"], dtype=np.float32),
        "obs_orange": np.asarray(rec["obs_orange"], dtype=np.float32),
        "reward": np.asarray(rec["reward"], dtype=np.float64),
    }


def sim_bots_trajectory(*, frames: int = 1800, random_spawn: bool = False, seed: int = 0) -> dict[str, np.ndarray]:
    """Heuristic bot vs heuristic bot at sim level: raw physics state, bot states and both views."""
    seed_everything(seed)
    sim = make_sim(enable_orange=True, random_spawn=random_spawn, seed=seed)
    blue = make_bot_controller("blue", sim)
    orange = make_bot_controller("orange", sim)
    rec = {"state": [], "obs": [], "obs_mirror": [], "bot_states": [], "goal": []}
    for _ in range(frames):
        a_blue = blue.get_action()
        a_orange = orange.get_action()
        sim.step(a_blue, 1.0 / 60.0, action_orange=a_orange)
        if sim.goal_scored_this_step is not None:
            blue.reset()
            orange.reset()
        state = []
        for body in (sim.blue_car.body, sim.orange_car.body, sim.ball.body):
            state.extend([body.position.x, body.position.y, body.velocity.x, body.velocity.y])
            state.extend([body.angle, body.angular_velocity])
        state.extend([sim.blue_car.boost_amount, sim.orange_car.boost_amount, sim.score_blue, sim.score_orange])
        rec["state"].append(state)
        obs, obs_mirror = sim_obs(sim)
        rec["obs"].append(obs)
        rec["obs_mirror"].append(obs_mirror)
        rec["bot_states"].append(f"{bot_state(blue)}|{bot_state(orange)}")
        rec["goal"].append({None: 0, "blue": 1, "orange": 2}[sim.goal_scored_this_step])
    return {
        "state": np.asarray(rec["state"], dtype=np.float64),
        "obs": np.asarray(rec["obs"], dtype=np.float64),
        "obs_mirror": np.asarray(rec["obs_mirror"], dtype=np.float64),
        "bot_states": np.asarray(rec["bot_states"]),
        "goal": np.asarray(rec["goal"], dtype=np.int8),
    }


# Architectures: name -> builder(hidden)
ARCHITECTURES = {
    "dqn": lambda hidden: build_dqn(hidden, dueling=False),
    "dueling_dqn": lambda hidden: build_dqn(hidden, dueling=True),
    "ppo_flat": lambda hidden: build_actor_critic(hidden, 36),
    "ppo_factored": lambda hidden: build_actor_critic(hidden, [9, 2, 2]),
}


def build_tiny_checkpoints() -> dict[str, dict]:
    """Seeded tiny networks, one per architecture, as state_dicts."""
    out = {}
    for i, (name, build) in enumerate(ARCHITECTURES.items()):
        torch.manual_seed(1000 + i)
        out[name] = build(TINY_HIDDEN).state_dict()
    return out


# --------------------------------------------------------------------------------------------
# Short seeded training runs
# --------------------------------------------------------------------------------------------

TRAINING_OVERRIDES = {
    "dueling_ddqn": dict(
        seed=0,
        hidden_size=32,
        buffer_size=1_000,
        learning_start=64,
        batch_size=16,
        train_freq=4,
        target_update_interval=50,
        checkpoint_interval=2,
    ),
    "ppo": dict(seed=0, hidden_size=32, rollout_steps=64, minibatch_size=16, update_epochs=2, checkpoint_interval=2),
}
TRAINING_EPISODES = 3
TRAINING_MAX_EPISODE_STEPS = 300


class _Patch:
    """Temporarily set attributes (restored on exit)."""

    def __init__(self, patches: list[tuple[object, str, object]]):
        self.patches = patches
        self.saved = []

    def __enter__(self):
        for obj, name, value in self.patches:
            self.saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)


def short_training_run(algo: str, runs_dir: Path) -> tuple[str, dict]:
    """Train `algo` ("dueling_ddqn" or "ppo") for a few seeded episodes with a shrunken config.

    Returns (metrics CSV text, final best-model state_dict).
    """
    from dataclasses import replace

    import src.learning_agents.dueling_ddqn_agent as dqn_agent
    import src.learning_agents.ppo_agent as ppo_agent
    import src.learning_agents.run_dirs as run_dirs
    from src.config.nn_config import DQNConfig, PPOConfig

    agent_module = dqn_agent if algo == "dueling_ddqn" else ppo_agent
    config = replace(DQNConfig() if algo == "dueling_ddqn" else PPOConfig(), **TRAINING_OVERRIDES[algo])
    original_make_env = agent_module.make_env

    def seeded_make_env(*args, **kwargs):
        env = original_make_env(*args, **kwargs)
        env.action_space.seed(0)
        env.unwrapped.sim.rng.seed(0)  # spawn sampling
        return env

    patches = [(run_dirs, "RUNS_DIR", Path(runs_dir)), (agent_module, "make_env", seeded_make_env)]
    with _Patch(patches):
        seed_everything(0)
        train = dqn_agent.train_dueling_ddqn if algo == "dueling_ddqn" else ppo_agent.train_ppo
        train(num_episodes=TRAINING_EPISODES, max_episode_steps=TRAINING_MAX_EPISODE_STEPS, config=config)

    (run_dir,) = [p for p in Path(runs_dir).iterdir() if p.is_dir()]
    metrics = (run_dir / f"metrics_{algo}.csv").read_text()
    state_dict = torch.load(run_dir / f"model_{algo}.pt", map_location="cpu", weights_only=True)
    return metrics, state_dict
