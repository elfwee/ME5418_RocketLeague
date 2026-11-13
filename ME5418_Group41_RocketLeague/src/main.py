"""Main entry point for interactive 2D Rocket League simulation, Gym demo, Neural Network tests, and Learning Agents."""

import os
import sys

from src.eval.head_to_head import play_head_to_head
from src.eval.mixed_eval import evaluate_mixed_policies
from src.gym_env.rocket_league_env import demo_gym
from src.learning_agents.agent_common import latest_model_path
from src.learning_agents.dueling_ddqn_agent import (
    evaluate_dueling_ddqn,
    train_dueling_ddqn,
)
from src.learning_agents.ppo_agent import evaluate_ppo, train_ppo
from src.neural_nets.demos import run_dqn_network, run_ppo_network
from src.paths import RUNS_DIR
from src.sim.game import play_headless, play_interactive


def _resolve_default_h2h_model():
    """Return the first available trained checkpoint for --head_to_head, or None for HeuristicBot."""
    candidates = (
        str(latest_model_path("ppo")),
        str(latest_model_path("dueling_ddqn")),
        f"{RUNS_DIR}/20261106_0911/model_ppo.pt",
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _arg_value(flag: str, default=None):
    """Return the value following `flag` in sys.argv, or `default` if absent."""
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        raise SystemExit(f"{flag} requires a value")
    return default


def _h2h_model_arg(flag: str, default):
    """Resolve --blue/--orange model path for --head_to_head ('none'/'bot'/'heuristic' -> HeuristicBot)."""
    val = _arg_value(flag, default)
    if isinstance(val, str) and val.lower() in ("none", "bot", "heuristic", "heuristic_bot"):
        return None
    return val


def main():
    if "--play" in sys.argv:
        enable_orange = "--no-orange" not in sys.argv
        if "--orange" in sys.argv:
            control_car = "orange"
        elif "--bot-vs-bot" in sys.argv or "--bots" in sys.argv:
            control_car = "none"
        else:
            # Default or explicit `--blue`
            control_car = "blue"

        if "--headless" in sys.argv:
            play_headless(steps=300, enable_orange=enable_orange)
        else:
            play_interactive(enable_orange=enable_orange, control_car=control_car)
    elif "--head_to_head" in sys.argv:
        default_model = _resolve_default_h2h_model()
        max_goals_val = _arg_value("--max_goals")
        max_seconds_val = _arg_value("--max_seconds")
        play_head_to_head(
            blue_model=_h2h_model_arg("--blue", default_model),
            orange_model=_h2h_model_arg("--orange", default_model),
            render="--headless" not in sys.argv,
            max_goals=int(max_goals_val) if max_goals_val is not None else None,
            max_seconds=float(max_seconds_val) if max_seconds_val is not None else 600.0,
            random_spawn="--random_spawn" in sys.argv,
            seed=int(seed) if (seed := _arg_value("--seed")) is not None else None,
        )
    elif "--demo" in sys.argv:
        if "--gym" in sys.argv:
            demo_gym()
        elif "--dqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=False, enable_double_dqn=False)
        elif "--double_dqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=False, enable_double_dqn=True)
        elif "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            run_dqn_network(enable_dueling_dqn=True, enable_double_dqn=True)
        elif "--ppo" in sys.argv or "-ppo" in sys.argv:
            run_ppo_network()
        else:
            print(
                "Please specify a demo target:\n"
                "  python -m src --demo --gym           # Gymnasium environment demo\n"
                "  python -m src --demo --dqn           # Vanilla DQN network test (PyTorch)\n"
                "  python -m src --demo --double_dqn    # Double DQN network test (PyTorch)\n"
                "  python -m src --demo --dueling_ddqn  # Dueling Double DQN network test (PyTorch) [alias: --ddqn]\n"
                "  python -m src --demo --ppo           # Proximal Policy Optimization network test (PyTorch) [alias: -ppo]"
            )
    elif "--train" in sys.argv:
        num_episodes = None
        if "--episodes" in sys.argv:
            idx = sys.argv.index("--episodes")
            if idx + 1 < len(sys.argv):
                num_episodes = int(sys.argv[idx + 1])

        model_path = None
        if "--model_path" in sys.argv:
            idx = sys.argv.index("--model_path")
            if idx + 1 < len(sys.argv):
                model_path = sys.argv[idx + 1]

        if "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            train_dueling_ddqn(
                num_episodes=num_episodes,
                model_path=model_path,
                enable_dueling_dqn=True,
                enable_double_dqn=True,
                hyperparameter_set="dueling_ddqn",
            )
        elif "--double_dqn" in sys.argv:
            train_dueling_ddqn(
                num_episodes=num_episodes,
                model_path=model_path,
                enable_dueling_dqn=False,
                enable_double_dqn=True,
                hyperparameter_set="double_dqn",
            )
        elif "--dqn" in sys.argv:
            train_dueling_ddqn(
                num_episodes=num_episodes,
                model_path=model_path,
                enable_dueling_dqn=False,
                enable_double_dqn=False,
                hyperparameter_set="dqn",
            )
        elif "--ppo" in sys.argv or "-ppo" in sys.argv:
            train_ppo(num_episodes=num_episodes, model_path=model_path, hyperparameter_set="ppo")
        else:
            print(
                "Please specify a learning agent to train:\n"
                "  python -m src --train --dueling_ddqn [--episodes N] # Train Dueling Double DQN agent [alias: --ddqn]\n"
                "  python -m src --train --double_dqn   [--episodes N] # Train Double DQN agent\n"
                "  python -m src --train --dqn          [--episodes N] # Train Vanilla DQN agent\n"
                "  python -m src --train --ppo          [--episodes N] # Train PPO agent [alias: -ppo]"
            )
    elif "--eval" in sys.argv:
        render = "--render" in sys.argv
        save_gif = "--gif" in sys.argv or "--save_gif" in sys.argv
        gif_path = None
        if "--gif_path" in sys.argv:
            idx = sys.argv.index("--gif_path")
            if idx + 1 < len(sys.argv):
                gif_path = sys.argv[idx + 1]

        model_path = None
        if "--model_path" in sys.argv:
            idx = sys.argv.index("--model_path")
            if idx + 1 < len(sys.argv):
                model_path = sys.argv[idx + 1]

        num_episodes = None
        if "--episodes" in sys.argv:
            idx = sys.argv.index("--episodes")
            if idx + 1 < len(sys.argv):
                num_episodes = int(sys.argv[idx + 1])

        if "--orange" in sys.argv:
            control_car = "orange"
        elif "--bot-vs-bot" in sys.argv or "--bots" in sys.argv or "--both" in sys.argv:
            control_car = "both"
        else:
            # Default or explicit `--blue`
            control_car = "blue"
        eval_random_spawn = not ("--no_random_spawn" in sys.argv or "--fixed_spawn" in sys.argv)

        if "--mix" in sys.argv:
            # Parse explicit or positional policy flags for Blue and Orange
            policy_flags = {
                "--dueling_ddqn": "dueling_ddqn",
                "--ddqn": "dueling_ddqn",
                "--double_dqn": "double_dqn",
                "--dqn": "dqn",
                "--ppo": "ppo",
                "-ppo": "ppo",
            }
            ordered_policies = [policy_flags[arg] for arg in sys.argv if arg in policy_flags]

            if len(ordered_policies) >= 2:
                blue_policy, orange_policy = ordered_policies[0], ordered_policies[1]
            elif len(ordered_policies) == 1:
                chosen = ordered_policies[0]
                other = "ppo" if chosen != "ppo" else "dueling_ddqn"
                if "--orange" in sys.argv and "--blue" not in sys.argv:
                    blue_policy, orange_policy = other, chosen
                else:
                    blue_policy, orange_policy = chosen, other
            else:
                if "--orange" in sys.argv and "--blue" not in sys.argv:
                    blue_policy, orange_policy = "ppo", "dueling_ddqn"
                else:
                    blue_policy, orange_policy = "dueling_ddqn", "ppo"

            if "--blue_policy" in sys.argv:
                idx = sys.argv.index("--blue_policy")
                if idx + 1 < len(sys.argv):
                    blue_policy = sys.argv[idx + 1]
            if "--orange_policy" in sys.argv:
                idx = sys.argv.index("--orange_policy")
                if idx + 1 < len(sys.argv):
                    orange_policy = sys.argv[idx + 1]

            blue_model_path = None
            orange_model_path = None
            for flag in ("--blue_model", "--blue_model_path", "--model_path_blue"):
                if flag in sys.argv:
                    idx = sys.argv.index(flag)
                    if idx + 1 < len(sys.argv):
                        blue_model_path = sys.argv[idx + 1]
            for flag in ("--orange_model", "--orange_model_path", "--model_path_orange"):
                if flag in sys.argv:
                    idx = sys.argv.index(flag)
                    if idx + 1 < len(sys.argv):
                        orange_model_path = sys.argv[idx + 1]

            # If `--model_path` was passed without `--blue_model`/`--orange_model`, assign intelligently
            model_paths = [sys.argv[i + 1] for i, arg in enumerate(sys.argv[:-1]) if arg == "--model_path"]
            if len(model_paths) >= 2:
                if blue_model_path is None:
                    blue_model_path = model_paths[0]
                if orange_model_path is None:
                    orange_model_path = model_paths[1]
            elif len(model_paths) == 1 and blue_model_path is None and orange_model_path is None:
                single_path = model_paths[0]
                if os.path.exists(single_path):
                    import torch

                    sd = torch.load(single_path, map_location="cpu", weights_only=True)
                    is_ppo_ckpt = any(k.startswith(("actor", "critic", "fc_actor", "fc_critic")) for k in sd.keys())
                    if is_ppo_ckpt and orange_policy == "ppo" and blue_policy != "ppo":
                        orange_model_path = single_path
                    elif (not is_ppo_ckpt) and orange_policy != "ppo" and blue_policy == "ppo":
                        orange_model_path = single_path
                    else:
                        blue_model_path = single_path
                else:
                    blue_model_path = single_path

            metrics = evaluate_mixed_policies(
                blue_policy=blue_policy,
                orange_policy=orange_policy,
                blue_model_path=blue_model_path,
                orange_model_path=orange_model_path,
                num_episodes=num_episodes,
                render=render,
                save_gif=save_gif,
                gif_path=gif_path,
                random_spawn=eval_random_spawn,
            )
            print(f"[Mixed Policy Evaluation ({blue_policy} vs {orange_policy})] {metrics}")
        elif "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                num_episodes=num_episodes,
                render=render,
                enable_dueling_dqn=True,
                enable_double_dqn=True,
                hyperparameter_set="dueling_ddqn",
                model_path=model_path,
                save_gif=save_gif,
                gif_path=gif_path,
                control_car=control_car,
                random_spawn=eval_random_spawn,
            )
            print(f"[Dueling Double DQN Evaluation] {metrics}")
        elif "--double_dqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                num_episodes=num_episodes,
                render=render,
                enable_dueling_dqn=False,
                enable_double_dqn=True,
                hyperparameter_set="double_dqn",
                model_path=model_path,
                save_gif=save_gif,
                gif_path=gif_path,
                control_car=control_car,
                random_spawn=eval_random_spawn,
            )
            print(f"[Double DQN Evaluation] {metrics}")
        elif "--dqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                num_episodes=num_episodes,
                render=render,
                enable_dueling_dqn=False,
                enable_double_dqn=False,
                hyperparameter_set="dqn",
                model_path=model_path,
                save_gif=save_gif,
                gif_path=gif_path,
                control_car=control_car,
                random_spawn=eval_random_spawn,
            )
            print(f"[Vanilla DQN Evaluation] {metrics}")
        elif "--ppo" in sys.argv or "-ppo" in sys.argv:
            metrics = evaluate_ppo(
                num_episodes=num_episodes,
                render=render,
                hyperparameter_set="ppo",
                model_path=model_path,
                save_gif=save_gif,
                gif_path=gif_path,
                control_car=control_car,
                random_spawn=eval_random_spawn,
            )
            print(f"[PPO Evaluation] {metrics}")
        else:
            print(
                "Please specify a learning agent to evaluate:\n"
                "  python -m src --eval --dueling_ddqn [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] # Evaluate Dueling Double DQN [alias: --ddqn]\n"
                "  python -m src --eval --double_dqn   [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] # Evaluate Double DQN\n"
                "  python -m src --eval --dqn          [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] # Evaluate Vanilla DQN\n"
                "  python -m src --eval --ppo          [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] # Evaluate PPO [alias: -ppo]\n"
                "  python -m src --eval --mix [--dueling_ddqn --ppo | --ppo --dueling_ddqn] [--render] [--no_random_spawn] # Mixed policy vs policy"
            )
    elif "--plot" in sys.argv:
        from src.learning_agents.plot_learning_curves import main as plot_main

        plot_args = [arg for arg in sys.argv[1:] if arg != "--plot"]
        plot_main(plot_args)
    else:
        print(
            "Usage:\n"
            "  python -m src --play [--blue|--orange|--bot-vs-bot] # Play interactively (default: control Blue car)\n"
            "  python -m src --play --headless                     # Run headless physics benchmark\n"
            "  python -m src --head_to_head [--headless] [--random_spawn] [--seed N] [--max_goals N] [--max_seconds S] [--blue PATH|none] [--orange PATH|none]  # Continuous agent vs agent\n"
            "  python -m src --demo --gym                          # Demo Gymnasium environment\n"
            "  python -m src --demo --dqn|--double_dqn|--ddqn      # Test DQN neural network variants\n"
            "  python -m src --demo --ppo                          # Test PPO Actor-Critic network\n"
            "  python -m src --train --dueling_ddqn|--ppo          # Train Dueling Double DQN or PPO agent\n"
            "  python -m src --eval --dueling_ddqn|--ppo [--blue|--orange|--bot-vs-bot] [--render] [--no_random_spawn] [--gif] # Evaluate agent\n"
            "  python -m src --eval --mix [--dueling_ddqn --ppo] [--blue_model ... --orange_model ...] [--no_random_spawn]     # Mixed policy match\n"
            "  python -m src --plot [--runs ... | --dashboard ...] # Plot multi-run comparison or training dashboard"
        )
