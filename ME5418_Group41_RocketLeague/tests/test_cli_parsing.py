"""Argument parsing of `python -m src` (fast: nothing is trained or simulated)."""

import pytest

from src import cli


def parse(parser_factory, *argv):
    return parser_factory().parse_args(list(argv))


def test_mode_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli._mode_parser().parse_known_args(["--train", "--eval"])


@pytest.mark.parametrize(
    "argv, side",
    [([], "blue"), (["--blue"], "blue"), (["--orange"], "orange"), (["--bot-vs-bot"], "both")],
)
def test_eval_side(argv, side):
    assert parse(cli.eval_parser, "--ppo", *argv).side == side


def test_play_flags():
    args = parse(cli.play_parser, "--orange", "--no-orange", "--headless")
    assert (args.side, args.orange_car, args.headless) == ("orange", False, True)
    assert parse(cli.play_parser).orange_car is True


def test_algorithm_flags_keep_command_line_order():
    assert parse(cli.eval_parser, "--mix", "--ppo", "--ddqn").algorithms == ["ppo", "dueling_ddqn"]
    assert parse(cli.eval_parser, "--mix", "--dueling_ddqn", "--ppo").algorithms == ["dueling_ddqn", "ppo"]


@pytest.mark.parametrize(
    "algorithms, orange_side, expected",
    [
        (["ppo", "dqn"], False, ("ppo", "dqn")),
        (["dqn"], False, ("dqn", "ppo")),
        (["dqn"], True, ("ppo", "dqn")),
        (["ppo"], False, ("ppo", "dueling_ddqn")),
        ([], False, ("dueling_ddqn", "ppo")),
        ([], True, ("ppo", "dueling_ddqn")),
    ],
)
def test_mix_policies(algorithms, orange_side, expected):
    assert cli.mix_policies(algorithms, orange_side) == expected


def test_mix_model_paths(tmp_path):
    from tests_helpers import golden_checkpoint

    ppo, dqn = golden_checkpoint("ppo_flat"), golden_checkpoint("dqn")
    assert cli.mix_model_paths(["a", "b"], None, None, "ppo", "dqn") == ("a", "b")
    assert cli.mix_model_paths(["a", "b"], "x", None, "ppo", "dqn") == ("x", "b")
    assert cli.mix_model_paths([], "x", "y", "ppo", "dqn") == ("x", "y")
    # A single --model_path goes to the side whose policy type matches the checkpoint
    assert cli.mix_model_paths([ppo], None, None, "dqn", "ppo") == (None, ppo)
    assert cli.mix_model_paths([dqn], None, None, "ppo", "dqn") == (None, dqn)
    assert cli.mix_model_paths([ppo], None, None, "ppo", "dqn") == (ppo, None)
    assert cli.mix_model_paths([str(tmp_path / "missing.pt")], None, None, "dqn", "ppo")[0].endswith("missing.pt")


def test_head_to_head_model_arguments():
    args = parse(cli.head_to_head_parser, "--blue", "a.pt", "--orange", "none", "--max_goals", "3")
    assert cli._model_or_bot(args.blue, "default.pt") == "a.pt"
    assert cli._model_or_bot(args.orange, "default.pt") is None
    assert cli._model_or_bot(None, "default.pt") == "default.pt"
    assert cli._model_or_bot(None, None) is None
    assert (args.max_goals, args.max_seconds) == (3, 600.0)


@pytest.mark.parametrize(
    "parser_factory, argv",
    [
        (cli.eval_parser, ["-ppo"]),
        (cli.eval_parser, ["--ppo", "--bots"]),
        (cli.eval_parser, ["--ppo", "--both"]),
        (cli.eval_parser, ["--ppo", "--save_gif"]),
        (cli.eval_parser, ["--ppo", "--fixed_spawn"]),
        (cli.eval_parser, ["--mix", "--blue_model_path", "x.pt"]),
        (cli.eval_parser, ["--mix", "--model_path_orange", "x.pt"]),
        (cli.play_parser, ["--bots"]),
        (cli.train_parser, ["--pp"]),  # no abbreviations
    ],
)
def test_dropped_aliases_are_rejected(parser_factory, argv):
    with pytest.raises(SystemExit):
        parser_factory().parse_args(argv)


@pytest.mark.parametrize("mode", [None, "play", "head_to_head", "demo", "train", "eval", "plot"])
def test_help(mode, capsys):
    argv = ["--help"] if mode is None else [f"--{mode}", "--help"]
    try:
        cli.main(argv)
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
