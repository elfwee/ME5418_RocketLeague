"""Exact regression against trajectories recorded from the pre-refactor code (`tests/golden/`).

Observations must be bit-identical: trained checkpoints depend on every one of the 52 floats,
for Blue and for the mirrored Orange view.
"""

import numpy as np
import pytest
import scenarios as sc


def _load(name):
    return dict(np.load(sc.GOLDEN_DIR / f"{name}.npz"))


def _assert_same(actual: dict, expected: dict):
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"'{key}' differs from golden")


@pytest.mark.parametrize("name", sorted(sc.ENV_SCENARIOS))
def test_env_trajectory_matches_golden(name):
    _assert_same(sc.env_trajectory(**sc.ENV_SCENARIOS[name]), _load(name))


def test_goal_events_match_golden():
    expected = _load("goal_events")
    actual = sc.goal_events_trajectory()
    _assert_same(actual, expected)
    # Sanity: the scripted cases exercise every terminal reward
    terminal = {
        round(float(r[t]), 6) for r, t in zip(expected["reward"], expected["terminated_at"], strict=True) if t >= 0
    }
    assert terminal == {20.0, 2.0, -20.0}


def test_action_repeat_discrete_matches_golden():
    _assert_same(sc.action_repeat_trajectory(discrete=True, seed=21, decisions=250), _load("action_repeat_discrete"))


def test_action_repeat_factored_matches_golden():
    _assert_same(sc.action_repeat_trajectory(discrete=False, seed=22, decisions=250), _load("action_repeat_factored"))


def test_action_repeat_both_cars_matches_golden():
    _assert_same(sc.action_repeat_both_trajectory(seed=23, decisions=250), _load("action_repeat_both"))


def test_sim_bot_match_fixed_spawn_matches_golden():
    _assert_same(sc.sim_bots_trajectory(frames=1800, random_spawn=False), _load("sim_bots_fixed"))


def test_sim_bot_match_random_spawn_matches_golden():
    _assert_same(sc.sim_bots_trajectory(frames=1800, random_spawn=True, seed=5), _load("sim_bots_random"))
