"""Smoke tests for every documented `python -m src ...` command.

Each test runs the CLI in a subprocess against a throwaway copy of `src/`, so training runs,
checkpoints and plots land in a temporary `runs/` folder instead of the real one.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 300

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("cli")
    shutil.copytree(PROJECT_ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    return root


def run_cli(workdir: Path, *args: str, timeout: int = TIMEOUT) -> str:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, "-m", "src", *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"`python -m src {' '.join(args)}` failed:\n{output[-4000:]}"
    assert "Traceback" not in output, output[-4000:]
    return output


def run_dirs(workdir: Path, model: str) -> list[Path]:
    runs = workdir / "runs"
    return sorted(p for p in runs.glob(f"*_{model}") if p.is_dir()) if runs.is_dir() else []


def test_usage(workdir):
    assert "Usage" in run_cli(workdir)


def test_play_headless(workdir):
    out = run_cli(workdir, "--play", "--headless")
    assert "Headless simulation benchmark completed successfully!" in out


@pytest.mark.parametrize(
    "flag, label",
    [
        ("--dqn", "VANILLA DQN"),
        ("--double_dqn", "DOUBLE DQN"),
        ("--dueling_ddqn", "DUELING DOUBLE DQN"),
        ("--ddqn", "DUELING DOUBLE DQN"),
        ("--ppo", "PPO ACTOR-CRITIC"),
    ],
)
def test_network_demos(workdir, flag, label):
    out = run_cli(workdir, "--demo", flag)
    assert label in out
    assert "BACKWARD PASS" in out


@pytest.mark.parametrize("flag, model", [("--ppo", "ppo"), ("--dueling_ddqn", "dueling_ddqn")])
def test_train_two_episodes(workdir, flag, model):
    out = run_cli(workdir, "--train", flag, "--episodes", "2")
    assert "Saving outputs to" in out
    (run_dir,) = run_dirs(workdir, model)
    assert (run_dir / f"model_{model}.pt").is_file()
    assert (run_dir / f"graph_{model}.png").is_file()
    header = (run_dir / f"metrics_{model}.csv").read_text().splitlines()[0]
    extras = "entropy,value_loss" if model == "ppo" else "epsilon,td_loss"
    assert header == f"episode,step,reward,mean_reward_100,goal,goal_rate_100,{extras}"


@pytest.mark.parametrize(
    "flags, banner",
    [
        (["--ppo"], "[PPO Evaluation]"),
        (["--dueling_ddqn"], "[Dueling Double DQN Evaluation]"),
        (["--ppo", "--orange"], "[PPO Evaluation]"),
        (["--dueling_ddqn", "--bot-vs-bot", "--no_random_spawn"], "[Dueling Double DQN Evaluation]"),
        (["--double_dqn"], "[Double DQN Evaluation]"),
        (["--dqn"], "[Vanilla DQN Evaluation]"),
    ],
)
def test_eval_one_episode(workdir, flags, banner):
    out = run_cli(workdir, "--eval", *flags, "--episodes", "1")
    assert banner in out
    if flags[0] in ("--ppo", "--dueling_ddqn") and run_dirs(workdir, flags[0][2:]):
        assert "Loaded trained model weights" in out


def test_eval_with_explicit_model_path(workdir):
    ckpt = PROJECT_ROOT / "tests" / "golden" / "checkpoints" / "ppo_factored.pt"
    out = run_cli(workdir, "--eval", "--ppo", "--model_path", str(ckpt), "--episodes", "1")
    assert f"Loaded trained model weights from: {ckpt}" in out


def test_eval_gif(workdir, tmp_path):
    gif = tmp_path / "eval.gif"
    out = run_cli(workdir, "--eval", "--ppo", "--gif", "--gif_path", str(gif), "--episodes", "1")
    assert "Saved evaluation GIF" in out
    assert gif.is_file() and gif.stat().st_size > 0


def test_eval_mix(workdir):
    out = run_cli(workdir, "--eval", "--mix", "--dueling_ddqn", "--ppo", "--episodes", "1")
    assert "[Mixed Policy Evaluation (dueling_ddqn vs ppo)]" in out


def test_eval_mix_explicit_models(workdir):
    ckpts = PROJECT_ROOT / "tests" / "golden" / "checkpoints"
    out = run_cli(
        workdir,
        "--eval",
        "--mix",
        "--ppo",
        "--dqn",
        "--blue_model",
        str(ckpts / "ppo_flat.pt"),
        "--orange_model",
        str(ckpts / "dqn.pt"),
        "--episodes",
        "1",
    )
    assert "[Mixed Policy Evaluation (ppo vs dqn)]" in out
    assert f"Loaded trained weights from: {ckpts / 'ppo_flat.pt'}" in out
    assert f"Loaded trained weights from: {ckpts / 'dqn.pt'}" in out


def test_head_to_head_headless(workdir):
    out = run_cli(workdir, "--head_to_head", "--headless", "--max_seconds", "5")
    assert "[HeadToHead] Final:" in out


def test_head_to_head_explicit_models(workdir):
    ckpts = PROJECT_ROOT / "tests" / "golden" / "checkpoints"
    out = run_cli(
        workdir,
        "--head_to_head",
        "--headless",
        "--random_spawn",
        "--seed",
        "3",
        "--max_seconds",
        "5",
        "--blue",
        str(ckpts / "ppo_factored.pt"),
        "--orange",
        "none",
    )
    assert "Blue:   ppo_factored.pt" in out
    assert "Orange: heuristic bot" in out


def test_plot(workdir):
    out = run_cli(workdir, "--plot")
    assert "[Plotting] Saved" in out


def test_plot_dashboard(workdir):
    if not run_dirs(workdir, "ppo"):
        pytest.skip("needs the PPO run from test_train_two_episodes")
    (run_dir,) = run_dirs(workdir, "ppo")
    out = run_cli(workdir, "--plot", "--dashboard", str(run_dir / "metrics_ppo.csv"))
    assert "Saved dashboard" in out
    assert (run_dir / "dashboard_metrics_ppo.png").is_file()
