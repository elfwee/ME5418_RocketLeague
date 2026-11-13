"""Reward terms, the action-repeat wrapper and env seeding."""

import random

import numpy as np
import pytest
import scenarios as sc

from src.config.gym_config import GymConfig
from src.gym_env import rewards
from src.gym_env.wrappers import ActionRepeat

CFG = GymConfig()


@pytest.mark.parametrize(
    "goal, last_touch, expected",
    [
        ("ego", "ego", 20.0),
        ("ego", None, 20.0),
        ("ego", "opponent", 2.0),
        ("opponent", "ego", -20.0),
        ("opponent", "opponent", -20.0),
    ],
)
def test_goal_reward(goal, last_touch, expected):
    assert rewards.goal_reward(goal, last_touch, CFG) == expected


def test_relative_team():
    assert rewards.relative_team("blue", "blue") == "ego"
    assert rewards.relative_team("orange", "blue") == "opponent"
    assert rewards.relative_team(None, "orange") is None


def test_ball_progress_credits_only_the_toucher():
    forward, backward = 0.01, -0.01
    assert rewards.ball_progress_reward(forward, 0.0, 0.0, "ego", CFG) > 0.0
    assert rewards.ball_progress_reward(forward, 0.0, 0.0, None, CFG) > 0.0
    assert rewards.ball_progress_reward(forward, 0.0, 0.0, "opponent", CFG) == 0.0
    assert rewards.ball_progress_reward(backward, 0.0, 0.0, "ego", CFG) == 0.0
    assert rewards.ball_progress_reward(backward, 0.0, 0.0, "opponent", CFG) < 0.0
    assert rewards.ball_progress_reward(CFG.ball_progress_deadband, 0.0, 0.0, None, CFG) == 0.0


def test_ball_progress_weighting():
    near_goal = rewards.ball_progress_reward(0.01, 0.9, 0.0, None, CFG)
    midfield = rewards.ball_progress_reward(0.01, 0.0, 0.0, None, CFG)
    rolling = rewards.ball_progress_reward(0.01, 0.0, -0.9, None, CFG)
    assert near_goal > midfield > rolling > 0.0
    assert rolling == pytest.approx(0.01 * np.exp(1 / 1.2) * CFG.ground_ball_y_factor)


def test_shaping_reward():
    # Closing in on the ball pays; standing still costs the distance and time penalties
    assert rewards.shaping_reward(0.0, 0.2, 0.3, CFG) > 0.0
    still = rewards.shaping_reward(0.0, 0.2, 0.2, CFG)
    assert still == pytest.approx(-(0.2 * CFG.distance_penalty_scale) - CFG.time_penalty)


def test_action_repeat_of_one_is_a_plain_step():
    a = sc.make_env(bot_type="heuristic_bot", random_spawn=True)
    b = sc.make_env(bot_type="heuristic_bot", random_spawn=True)
    a.reset(seed=4)
    b.reset(seed=4)
    wrapped = ActionRepeat(b, 1)
    for action in sc.scripted_actions(9, 120, hold=5):
        obs_a, r_a, term_a, *_ = a.step(action)
        obs_b, r_b, term_b, *_ = wrapped.step(action)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert (r_a, term_a) == (r_b, term_b)
        if term_a:
            break


def test_action_repeat_releases_jump_after_the_first_frame():
    env = sc.make_env(bot_type="none", random_spawn=False)
    env.reset(seed=0)
    seen = []
    original_step = env.sim.step

    def spy(action, dt, action_orange=None):
        seen.append(action.jump)
        return original_step(action, dt, action_orange=action_orange)

    env.sim.step = spy
    ActionRepeat(env, 4).step(2)  # direction 0, jump, no boost
    assert seen == [True, False, False, False]


def test_reset_seed_does_not_touch_global_rngs():
    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), np.random.random())
    random.seed(123)
    np.random.seed(123)
    env = sc.make_env(bot_type="heuristic_bot", random_spawn=True)
    env.reset(seed=5)
    for action in sc.scripted_actions(1, 50, hold=4):
        env.step(action)
    assert (random.random(), np.random.random()) == expected
