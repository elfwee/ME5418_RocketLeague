"""Gym environment contract, action encoding and observation symmetry."""

import itertools

import numpy as np
import pytest
import scenarios as sc
from gymnasium.utils.env_checker import check_env


def test_check_env_passes():
    env = sc.make_env(bot_type="none")
    check_env(env)
    env.close()


@pytest.mark.parametrize("discrete", [True, False])
def test_spaces(discrete):
    env = sc.make_env(bot_type="heuristic_bot", discrete_action=discrete)
    assert env.observation_space.shape == (52,)
    if discrete:
        assert env.action_space.n == 36
    else:
        assert list(env.action_space.nvec) == [9, 2, 2]
    obs, _ = env.reset(seed=0)
    assert obs.shape == (52,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    env.close()


DIRECTIONS = [(0, 0), (1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 1), (-1, -1), (-1, 1)]


def test_action_encoding_covers_all_36_indices():
    """Flat index a <-> (direction, jump, boost) = (a // 4, (a % 4) // 2, a % 2), both decode paths agree."""
    env = sc.make_env(bot_type="none")
    seen = set()
    for a in range(36):
        direction, jump, boost = a // 4, (a % 4) // 2, a % 2
        assert direction * 4 + jump * 2 + boost == a
        flat = env._decode_action(a)
        factored = env._decode_action([direction, jump, boost])
        assert flat == factored
        dx, dy = DIRECTIONS[direction]
        norm = max(1.0, float(np.hypot(dx, dy)))
        assert (flat.dir_x, flat.dir_y) == pytest.approx((dx / norm, dy / norm))
        assert flat.jump == bool(jump) and flat.boost == bool(boost)
        seen.add((direction, jump, boost))
    assert len(seen) == 36
    env.close()


def test_mirrored_action_negates_x_only():
    env = sc.make_env(bot_type="none")
    for a in range(36):
        plain = env._decode_action(a)
        mirrored = env._decode_action(a, mirror_x=True)
        assert mirrored.dir_x == -plain.dir_x
        assert (mirrored.dir_y, mirrored.jump, mirrored.boost) == (plain.dir_y, plain.jump, plain.boost)
    env.close()


def test_clear_jump():
    from src.learning_agents.agent_common import clear_jump

    for direction, jump, boost in itertools.product(range(9), range(2), range(2)):
        a = direction * 4 + jump * 2 + boost
        assert clear_jump(a) == direction * 4 + boost
        assert clear_jump([direction, jump, boost]) == [direction, 0, boost]


def test_kickoff_observation_is_mirror_symmetric():
    """At a symmetric kickoff, Orange's mirrored view is Blue's view."""
    sim = sc.make_sim(enable_orange=True)
    blue, orange = sc.sim_obs(sim)
    np.testing.assert_allclose(blue, orange, atol=1e-9)


def test_mirror_flips_ball_x_features():
    """In Orange's mirrored view the ball's x position, x velocity and spin change sign."""
    data = np.load(sc.GOLDEN_DIR / "sim_bots_fixed.npz")
    obs, mirror = data["obs"], data["obs_mirror"]
    ball = slice(21, 26)
    np.testing.assert_array_equal(mirror[:, ball][:, [0, 2, 4]], -obs[:, ball][:, [0, 2, 4]])
    np.testing.assert_array_equal(mirror[:, ball][:, [1, 3]], obs[:, ball][:, [1, 3]])
