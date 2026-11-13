"""Car controllers, and the one 'policy plays Orange' semantics shared by --eval --mix and --head_to_head."""

import subprocess
import sys

import pytest
import scenarios as sc
import torch

from src.gym_env.wrappers import ActionRepeat
from src.neural_nets.checkpoint import load_policy
from src.neural_nets.policy import greedy_action
from src.sim.control import (
    HeuristicController,
    HumanController,
    IdleController,
    RandomController,
    make_controller,
)
from src.sim.control.policy import PolicyController
from src.sim.core.actions import CarAction

CKPT = sc.GOLDEN_DIR / "checkpoints"


def _record_actions(sim, log):
    original = sim.step

    def spy(action, dt=1 / 60, action_orange=None):
        log.append((action, action_orange))
        original(action, dt, action_orange=action_orange)

    sim.step = spy


@pytest.mark.parametrize("blue, orange", [("ppo_factored", "dqn"), ("dueling_dqn", "ppo_flat")])
def test_eval_mix_and_head_to_head_drive_both_cars_identically(blue, orange):
    """Decision D3: the env `control_car="both"` + ActionRepeat path and sim-level PolicyControllers
    must give both cars the same input every frame."""
    frames = 600
    blue_net, _ = load_policy(CKPT / f"{blue}.pt")
    orange_net, _ = load_policy(CKPT / f"{orange}.pt")

    env = sc.make_env(bot_type="none", control_car="both", random_spawn=False)
    obs, info = env.reset(seed=0)
    env_log = []
    _record_actions(env.sim, env_log)
    wrapped = ActionRepeat(env, 4)
    while len(env_log) < frames:
        a_blue = greedy_action(blue_net, torch.tensor(obs))
        a_orange = greedy_action(orange_net, torch.tensor(info["obs_orange"]))
        obs, _, terminated, truncated, info = wrapped.step(a_blue, a_orange)
        assert not (terminated or truncated), "pick checkpoints that do not score within the window"

    sim = sc.make_sim(enable_orange=True, random_spawn=False)
    sim_log = []
    _record_actions(sim, sim_log)
    blue_ctrl = PolicyController("blue", sim, CKPT / f"{blue}.pt")
    orange_ctrl = PolicyController("orange", sim, CKPT / f"{orange}.pt")
    for _ in range(len(env_log)):
        sim.step(blue_ctrl.get_action(), 1 / 60, action_orange=orange_ctrl.get_action())

    assert sim_log == env_log


def test_policy_controller_idles_when_orange_is_removed():
    sim = sc.make_sim(enable_orange=True)
    ctrl = PolicyController("orange", sim, CKPT / "dqn.pt")
    sim.set_orange_enabled(False)
    assert ctrl.get_action() == CarAction()


def test_make_controller():
    sim = sc.make_sim(enable_orange=True)
    assert isinstance(make_controller("human"), HumanController)
    assert isinstance(make_controller("heuristic_bot", "orange", sim), HeuristicController)
    assert isinstance(make_controller("random", sim=sim), RandomController)
    assert isinstance(make_controller("none"), IdleController)
    assert isinstance(make_controller("passive"), IdleController)
    with pytest.raises(ValueError):
        make_controller("keyboard")
    assert make_controller("none").get_action() == CarAction()
    # Without a pygame window the keyboard controller idles
    assert make_controller("human").get_action() == CarAction()


def test_sim_core_does_not_import_ai_or_controllers():
    code = (
        "import sys; import src.sim.core.simulation; "
        "bad = [m for m in sys.modules if m.startswith(('src.sim.ai', 'src.sim.control', 'torch', 'pygame'))]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=sc.GOLDEN_DIR.parents[1])
