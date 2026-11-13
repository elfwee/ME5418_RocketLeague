"""Command-line interface: `python -m src --<mode> [options]`.

Exactly one mode flag picks what to run; every mode has its own options (`python -m src --<mode> --help`).
The mode is parsed first so that `--blue` / `--orange` can mean a side in `--play` / `--eval` and a
model path in `--head_to_head`.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

PROG = "python -m src"

USAGE = f"""Usage:
  {PROG} --play [--blue|--orange|--bot-vs-bot] [--no-orange]  # Play interactively (default: control Blue car)
  {PROG} --play --headless                     # Run headless physics benchmark
  {PROG} --head_to_head [--headless] [--random_spawn] [--seed N] [--max_goals N] [--max_seconds S] [--blue PATH|none] [--orange PATH|none]  # Continuous agent vs agent
  {PROG} --demo --gym                          # Demo Gymnasium environment
  {PROG} --demo --dqn|--double_dqn|--ddqn      # Test DQN neural network variants
  {PROG} --demo --ppo                          # Test PPO Actor-Critic network
  {PROG} --train --dueling_ddqn|--double_dqn|--dqn|--ppo [--episodes N] [--model_path PATH]  # Train an agent
  {PROG} --curriculum --dueling_ddqn|--double_dqn|--dqn|--ppo [--patience N] [--self_play_rounds N] [--max_stage_episodes N]  # Curriculum: none -> heuristic bot -> self-play
  {PROG} --eval --dueling_ddqn|--double_dqn|--dqn|--ppo [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] [--gif] [--gif_path PATH]  # Evaluate agent
  {PROG} --eval --mix [--dueling_ddqn --ppo] [--blue_model ... --orange_model ...] [--no_random_spawn]  # Mixed policy match
  {PROG} --plot [--runs ... | --dashboard ...] # Plot multi-run comparison or training dashboard

Run `{PROG} --<mode> --help` for every option of a mode."""

MODES = {
    "play": "play interactively, or run the headless physics benchmark",
    "head_to_head": "continuous match between two checkpoints or heuristic bots",
    "demo": "Gymnasium environment or network forward/backward pass demo",
    "train": "train a learning agent",
    "curriculum": "train a learning agent through a curriculum: no opponent -> heuristic bot -> self-play",
    "eval": "evaluate a trained agent, or two policies against each other (--mix)",
    "plot": "plot learning curves / training dashboards from runs/",
}

HEURISTIC_BOT_NAMES = ("none", "bot", "heuristic", "heuristic_bot")


@dataclass(frozen=True)
class Algorithm:
    """One trainable agent: its display name and how it maps onto the agent implementations."""

    label: str
    family: str  # "dqn" or "ppo"
    dueling: bool = False
    double: bool = False


ALGORITHMS = {
    "dueling_ddqn": Algorithm("Dueling Double DQN", "dqn", dueling=True, double=True),
    "double_dqn": Algorithm("Double DQN", "dqn", dueling=False, double=True),
    "dqn": Algorithm("Vanilla DQN", "dqn", dueling=False, double=False),
    "ppo": Algorithm("PPO", "ppo"),
}


# --------------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------------


def _mode_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="2D Rocket League simulation, Gymnasium environment and DQN / PPO learning agents.",
        epilog=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        allow_abbrev=False,
    )
    group = parser.add_mutually_exclusive_group()
    for mode, help_text in MODES.items():
        group.add_argument(f"--{mode}", dest="mode", action="store_const", const=mode, help=help_text)
    parser.add_argument("-h", "--help", action="store_true", help="show this help message (or a mode's help)")
    return parser


def _sub_parser(mode: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"{PROG} --{mode}", description=description, allow_abbrev=False)


def _add_algorithm_flags(parser: argparse.ArgumentParser, help_suffix: str = ""):
    """--dueling_ddqn (--ddqn) / --double_dqn / --dqn / --ppo, collected in command-line order."""
    group = parser.add_argument_group("algorithm")
    for name, algo in ALGORITHMS.items():
        flags = [f"--{name}", "--ddqn"] if name == "dueling_ddqn" else [f"--{name}"]
        group.add_argument(
            *flags, dest="algorithms", action="append_const", const=name, help=f"{algo.label}{help_suffix}"
        )
    parser.set_defaults(algorithms=[])


def _add_side_flags(parser: argparse.ArgumentParser, blue: str, orange: str, both: str):
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--blue", dest="side", action="store_const", const="blue", help=blue)
    side.add_argument("--orange", dest="side", action="store_const", const="orange", help=orange)
    side.add_argument("--bot-vs-bot", dest="side", action="store_const", const="both", help=both)
    parser.set_defaults(side="blue")


def play_parser() -> argparse.ArgumentParser:
    parser = _sub_parser("play", MODES["play"])
    _add_side_flags(
        parser,
        blue="you drive Blue, Orange is the heuristic bot (default)",
        orange="you drive Orange, Blue is the heuristic bot",
        both="heuristic bot vs heuristic bot",
    )
    parser.add_argument("--no-orange", dest="orange_car", action="store_false", help="play without the Orange car")
    parser.add_argument("--headless", action="store_true", help="run the 300-step headless physics benchmark")
    return parser


def head_to_head_parser() -> argparse.ArgumentParser:
    parser = _sub_parser("head_to_head", MODES["head_to_head"])
    parser.add_argument(
        "--blue",
        metavar="PATH|none",
        default=None,
        help="Blue checkpoint, or 'none' for the heuristic bot (default: latest PPO, else Dueling DDQN run)",
    )
    parser.add_argument("--orange", metavar="PATH|none", default=None, help="Orange checkpoint (same default)")
    parser.add_argument("--headless", action="store_true", help="no window; run as fast as possible")
    parser.add_argument("--random_spawn", action="store_true", help="randomise every kickoff")
    parser.add_argument("--seed", type=int, default=None, help="seed for repeatable random spawns")
    parser.add_argument("--max_goals", type=int, default=None, help="stop after this many goals in total")
    parser.add_argument("--max_seconds", type=float, default=600.0, help="stop after this much sim time (default 600)")
    return parser


def demo_parser() -> argparse.ArgumentParser:
    parser = _sub_parser("demo", MODES["demo"])
    parser.add_argument("--gym", dest="algorithms", action="append_const", const="gym", help="Gymnasium environment")
    _add_algorithm_flags(parser, " network forward/backward pass")
    return parser


def train_parser() -> argparse.ArgumentParser:
    parser = _sub_parser("train", MODES["train"])
    _add_algorithm_flags(parser)
    parser.add_argument("--episodes", type=int, default=None, help="number of episodes (default: until stopped)")
    parser.add_argument("--model_path", default=None, help="checkpoint to fine-tune from")
    return parser


def curriculum_parser() -> argparse.ArgumentParser:
    parser = _sub_parser("curriculum", MODES["curriculum"])
    _add_algorithm_flags(parser)
    parser.add_argument(
        "--patience", type=int, default=1000, help="episodes without a new best 100-ep mean before a stage ends"
    )
    parser.add_argument(
        "--self_play_rounds", type=int, default=5, help="self-play rounds; the opponent is refreshed after each"
    )
    parser.add_argument(
        "--max_stage_episodes", type=int, default=None, help="hard cap on episodes per stage / round (default: none)"
    )
    parser.add_argument("--model_name", default=None, help="run / model name (default: the algorithm name)")
    return parser


def eval_parser() -> argparse.ArgumentParser:
    parser = _sub_parser("eval", MODES["eval"])
    _add_algorithm_flags(parser, " (with --mix: first flag plays Blue, second plays Orange)")
    _add_side_flags(
        parser,
        blue="policy drives Blue vs the heuristic bot (default)",
        orange="policy drives Orange (mirrored) vs keyboard Blue",
        both="policy drives both cars",
    )
    parser.add_argument("--episodes", type=int, default=None, help="episodes to evaluate (default: from config)")
    parser.add_argument(
        "--model_path",
        action="append",
        default=[],
        help="checkpoint (default: latest run of that model); with --mix, give it twice for Blue then Orange",
    )
    parser.add_argument("--render", action="store_true", help="watch in a window")
    parser.add_argument("--no_random_spawn", action="store_true", help="fixed kickoff spawns for reproducible runs")
    parser.add_argument("--gif", action="store_true", help="record the first episode as a GIF")
    parser.add_argument("--gif_path", default=None, help="where to save the GIF (default: next to the checkpoint)")
    mix = parser.add_argument_group("mixed policies")
    mix.add_argument("--mix", action="store_true", help="pit two (possibly different) policies against each other")
    mix.add_argument("--blue_policy", default=None, help="Blue policy type (overrides the algorithm flags)")
    mix.add_argument("--orange_policy", default=None, help="Orange policy type (overrides the algorithm flags)")
    mix.add_argument("--blue_model", default=None, help="Blue checkpoint")
    mix.add_argument("--orange_model", default=None, help="Orange checkpoint")
    return parser


# --------------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------------


def _single_algorithm(parser: argparse.ArgumentParser, algorithms: list[str]) -> str | None:
    if len(algorithms) > 1:
        parser.error(f"choose one of {', '.join('--' + a for a in algorithms)}")
    return algorithms[0] if algorithms else None


def run_play(args: argparse.Namespace):
    from src.sim.game import play_headless, play_interactive

    if args.headless:
        play_headless(steps=300, enable_orange=args.orange_car)
    else:
        play_interactive(enable_orange=args.orange_car, control_car="none" if args.side == "both" else args.side)


def default_head_to_head_model() -> str | None:
    """Latest PPO checkpoint, else the latest Dueling DDQN one, else None (heuristic bot)."""
    from src.learning_agents.run_dirs import latest_model_path

    for model_name in ("ppo", "dueling_ddqn"):
        path = latest_model_path(model_name)
        if path.is_file():
            return str(path)
    return None


def _model_or_bot(value: str | None, default: str | None) -> str | None:
    value = default if value is None else value
    return None if value is None or value.lower() in HEURISTIC_BOT_NAMES else value


def run_head_to_head(args: argparse.Namespace):
    from src.eval.head_to_head import play_head_to_head

    default_model = default_head_to_head_model() if args.blue is None or args.orange is None else None
    play_head_to_head(
        blue_model=_model_or_bot(args.blue, default_model),
        orange_model=_model_or_bot(args.orange, default_model),
        render=not args.headless,
        max_goals=args.max_goals,
        max_seconds=args.max_seconds,
        random_spawn=args.random_spawn,
        seed=args.seed,
    )


DEMO_HELP = f"""Please specify a demo target:
  {PROG} --demo --gym           # Gymnasium environment demo
  {PROG} --demo --dqn           # Vanilla DQN network test (PyTorch)
  {PROG} --demo --double_dqn    # Double DQN network test (PyTorch)
  {PROG} --demo --dueling_ddqn  # Dueling Double DQN network test (PyTorch) [alias: --ddqn]
  {PROG} --demo --ppo           # Proximal Policy Optimization network test (PyTorch)"""


def run_demo(args: argparse.Namespace, parser: argparse.ArgumentParser):
    target = _single_algorithm(parser, args.algorithms)
    if target is None:
        print(DEMO_HELP)
    elif target == "gym":
        from src.gym_env.demo import demo_gym

        demo_gym()
    elif target == "ppo":
        from src.neural_nets.demos import run_ppo_network

        run_ppo_network()
    else:
        from src.neural_nets.demos import run_dqn_network

        algo = ALGORITHMS[target]
        run_dqn_network(enable_dueling_dqn=algo.dueling, enable_double_dqn=algo.double)


TRAIN_HELP = f"""Please specify a learning agent to train:
  {PROG} --train --dueling_ddqn [--episodes N] # Train Dueling Double DQN agent [alias: --ddqn]
  {PROG} --train --double_dqn   [--episodes N] # Train Double DQN agent
  {PROG} --train --dqn          [--episodes N] # Train Vanilla DQN agent
  {PROG} --train --ppo          [--episodes N] # Train PPO agent"""


def run_train(args: argparse.Namespace, parser: argparse.ArgumentParser):
    name = _single_algorithm(parser, args.algorithms)
    if name is None:
        print(TRAIN_HELP)
        return
    algo = ALGORITHMS[name]
    if algo.family == "ppo":
        from src.learning_agents.ppo_agent import train_ppo

        train_ppo(num_episodes=args.episodes, model_path=args.model_path, hyperparameter_set=name)
    else:
        from src.learning_agents.dueling_ddqn_agent import train_dueling_ddqn

        train_dueling_ddqn(
            num_episodes=args.episodes,
            model_path=args.model_path,
            enable_dueling_dqn=algo.dueling,
            enable_double_dqn=algo.double,
            hyperparameter_set=name,
        )


CURRICULUM_HELP = f"""Please specify a learning agent for the curriculum:
  {PROG} --curriculum --dueling_ddqn # Dueling Double DQN [alias: --ddqn]
  {PROG} --curriculum --double_dqn   # Double DQN
  {PROG} --curriculum --dqn          # Vanilla DQN
  {PROG} --curriculum --ppo          # PPO"""


def run_curriculum(args: argparse.Namespace, parser: argparse.ArgumentParser):
    name = _single_algorithm(parser, args.algorithms)
    if name is None:
        print(CURRICULUM_HELP)
        return
    from dataclasses import replace

    from src.config.nn_config import DQNConfig, PPOConfig
    from src.curriculum_learning.curriculum_learning import Curriculum
    from src.learning_agents.dueling_ddqn_agent import DuelingDDQNAgent
    from src.learning_agents.ppo_agent import PPOAgent

    algo = ALGORITHMS[name]
    if algo.family == "ppo":
        agent, config = PPOAgent, replace(PPOConfig(), hyperparameter_set=name)
    else:
        agent = DuelingDDQNAgent
        config = replace(
            DQNConfig(), hyperparameter_set=name, enable_dueling_dqn=algo.dueling, enable_double_dqn=algo.double
        )
    Curriculum(
        agent,
        patience=args.patience,
        self_play_rounds=args.self_play_rounds,
        max_stage_episodes=args.max_stage_episodes,
        model_name=args.model_name,
        config=config,
    ).learn()


EVAL_HELP = f"""Please specify a learning agent to evaluate:
  {PROG} --eval --dueling_ddqn [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] # Evaluate Dueling Double DQN [alias: --ddqn]
  {PROG} --eval --double_dqn   [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] # Evaluate Double DQN
  {PROG} --eval --dqn          [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] # Evaluate Vanilla DQN
  {PROG} --eval --ppo          [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] # Evaluate PPO
  {PROG} --eval --mix [--dueling_ddqn --ppo | --ppo --dueling_ddqn] [--render] [--no_random_spawn] # Mixed policy vs policy"""


def run_eval(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.mix:
        run_eval_mix(args)
        return
    name = _single_algorithm(parser, args.algorithms)
    if name is None:
        print(EVAL_HELP)
        return
    algo = ALGORITHMS[name]
    common = dict(
        num_episodes=args.episodes,
        render=args.render,
        hyperparameter_set=name,
        model_path=args.model_path[0] if args.model_path else None,
        save_gif=args.gif,
        gif_path=args.gif_path,
        control_car=args.side,
        random_spawn=not args.no_random_spawn,
    )
    if algo.family == "ppo":
        from src.learning_agents.ppo_agent import evaluate_ppo

        metrics = evaluate_ppo(**common)
    else:
        from src.learning_agents.dueling_ddqn_agent import evaluate_dueling_ddqn

        metrics = evaluate_dueling_ddqn(enable_dueling_dqn=algo.dueling, enable_double_dqn=algo.double, **common)
    print(f"[{algo.label} Evaluation] {metrics}")


def mix_policies(algorithms: list[str], orange_side: bool) -> tuple[str, str]:
    """(Blue, Orange) policy types for --eval --mix from the algorithm flags, in command-line order.

    With one flag, that policy plays Blue (Orange with --orange) against PPO, or against Dueling DDQN
    when the flag is PPO. With none, Dueling DDQN plays PPO.
    """
    if len(algorithms) >= 2:
        return algorithms[0], algorithms[1]
    if len(algorithms) == 1:
        chosen = algorithms[0]
        other = "ppo" if chosen != "ppo" else "dueling_ddqn"
        return (other, chosen) if orange_side else (chosen, other)
    return ("ppo", "dueling_ddqn") if orange_side else ("dueling_ddqn", "ppo")


def mix_model_paths(
    model_paths: list[str], blue_model: str | None, orange_model: str | None, blue_policy: str, orange_policy: str
) -> tuple[str | None, str | None]:
    """(Blue, Orange) checkpoints for --eval --mix.

    --blue_model / --orange_model win. Two --model_path values go to Blue then Orange. A single
    --model_path goes to the side whose policy type matches the checkpoint (Blue when ambiguous).
    """
    if len(model_paths) >= 2:
        return blue_model or model_paths[0], orange_model or model_paths[1]
    if len(model_paths) == 1 and blue_model is None and orange_model is None:
        from src.neural_nets.checkpoint import is_ppo_state_dict, load_state_dict

        single_path = model_paths[0]
        state_dict = load_state_dict(single_path)
        if state_dict is not None:
            is_ppo_ckpt = is_ppo_state_dict(state_dict)
            if is_ppo_ckpt and orange_policy == "ppo" and blue_policy != "ppo":
                return None, single_path
            if not is_ppo_ckpt and orange_policy != "ppo" and blue_policy == "ppo":
                return None, single_path
        return single_path, None
    return blue_model, orange_model


def run_eval_mix(args: argparse.Namespace):
    from src.eval.mixed_eval import evaluate_mixed_policies

    blue_policy, orange_policy = mix_policies(args.algorithms, orange_side=args.side == "orange")
    blue_policy = args.blue_policy or blue_policy
    orange_policy = args.orange_policy or orange_policy
    blue_model, orange_model = mix_model_paths(
        args.model_path, args.blue_model, args.orange_model, blue_policy, orange_policy
    )
    metrics = evaluate_mixed_policies(
        blue_policy=blue_policy,
        orange_policy=orange_policy,
        blue_model_path=blue_model,
        orange_model_path=orange_model,
        num_episodes=args.episodes,
        render=args.render,
        save_gif=args.gif,
        gif_path=args.gif_path,
        random_spawn=not args.no_random_spawn,
    )
    print(f"[Mixed Policy Evaluation ({blue_policy} vs {orange_policy})] {metrics}")


def run_plot(rest: list[str]):
    from src.learning_agents.plot_learning_curves import main as plot_main

    plot_main(rest)


SUB_PARSERS = {
    "play": play_parser,
    "head_to_head": head_to_head_parser,
    "demo": demo_parser,
    "train": train_parser,
    "curriculum": curriculum_parser,
    "eval": eval_parser,
}


def main(argv: Sequence[str] | None = None):
    mode_parser = _mode_parser()
    known, rest = mode_parser.parse_known_args(argv)
    if known.mode is None:
        if known.help:
            mode_parser.print_help()
        else:
            print(USAGE)
        return
    if known.mode == "plot":
        run_plot(rest + (["--help"] if known.help else []))
        return

    parser = SUB_PARSERS[known.mode]()
    args = parser.parse_args(rest + (["--help"] if known.help else []))
    if known.mode == "play":
        run_play(args)
    elif known.mode == "head_to_head":
        run_head_to_head(args)
    elif known.mode == "demo":
        run_demo(args, parser)
    elif known.mode == "train":
        run_train(args, parser)
    elif known.mode == "curriculum":
        run_curriculum(args, parser)
    elif known.mode == "eval":
        run_eval(args, parser)
