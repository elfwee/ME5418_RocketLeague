"""Observation layout and the left-right mirror."""

import numpy as np
import pytest
import scenarios as sc

from src.sim.core import observation as ob


def test_layout_is_contiguous_and_52_long():
    blocks = [
        ob.EGO_CAR,
        ob.EGO_RAYS,
        ob.BALL,
        ob.EGO_TO_BALL,
        ob.BALL_TO_OPPONENT_GOAL,
        ob.BALL_TO_OWN_GOAL,
        slice(ob.OPPONENT_PRESENT, ob.OPPONENT_PRESENT + 1),
        ob.OPPONENT_CAR,
        ob.OPPONENT_TO_BALL,
    ]
    assert blocks[0].start == 0
    for prev, nxt in zip(blocks, blocks[1:], strict=False):
        assert prev.stop == nxt.start
    assert blocks[-1].stop == ob.OBS_DIM == 52


def test_mirror_is_an_involution():
    obs = np.load(sc.GOLDEN_DIR / "sim_bots_random.npz")["obs"]
    for row in obs[::50]:
        row = row.tolist()
        assert ob.mirror_observation(ob.mirror_observation(row)) == row


def test_orange_view_is_the_mirror_of_orange_as_ego():
    """Orange's view = mirror(observation built with Orange as the ego car, unmirrored)."""
    data = np.load(sc.GOLDEN_DIR / "sim_bots_random.npz")
    for blue, orange in zip(data["obs"][::50], data["obs_mirror"][::50], strict=True):
        # Ball features are shared: the mirror negates x position, x velocity and spin
        np.testing.assert_array_equal(orange[ob.BALL], ob.mirror_observation(blue.tolist())[ob.BALL])
        # Each car sees the other as its opponent (flags differ: boost recovery vs is-boosting)
        np.testing.assert_array_equal(
            ob.mirror_observation(orange.tolist())[ob.OPPONENT_CAR][:12], blue[ob.EGO_CAR][:12]
        )


def test_orange_view_without_orange_car_raises():
    sim = sc.make_sim(enable_orange=False)
    assert len(ob.build_observation(sim, "blue")) == 52
    assert ob.build_observation(sim, "blue")[ob.OPPONENT_PRESENT] == 0.0
    with pytest.raises(ValueError):
        ob.build_observation(sim, "orange")


def test_sim_delegates_match_builder():
    sim = sc.make_sim(enable_orange=True, random_spawn=True, seed=3)
    assert sim.get_state_norm(as_flat_array=True) == ob.build_observation(sim, "blue")
    assert sim.get_state_norm_mirror(as_flat_array=True) == ob.build_observation(sim, "orange")


def test_owned_rng_makes_spawns_repeatable():
    a = sc.make_sim(enable_orange=True, random_spawn=True, seed=42)
    b = sc.make_sim(enable_orange=True, random_spawn=True, seed=42)
    assert a.sample_random_spawns() == b.sample_random_spawns()
