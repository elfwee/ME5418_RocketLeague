"""Main entry point for interactive 2D Rocket League simulation, Gym demo, Neural Network tests, and Learning Agents."""

import os
import sys

# Ensure repository root is on sys.path when invoked directly (e.g., `python src/main.py`)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.sim.game import play_headless, play_interactive
from src.gym_env.rocket_league_env import demo_gym
from src.neural_nets.dueling_ddqn_torch import run_dqn_network
from src.neural_nets.ppo_torch import run_ppo_network
from src.learning_agents.dueling_ddqn_agent import evaluate_dueling_ddqn, train_dueling_ddqn
from src.learning_agents.ppo_agent import evaluate_ppo, train_ppo


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
            if idx +1 < len(sys.argv):
                num_episodes = int(sys.argv[idx + 1])

        model_path = None
        if "--model_path" in sys.argv:
            idx = sys.argv.index("--model_path")
            if idx + 1 < len(sys.argv):
                model_path = sys.argv[idx + 1]

        if "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            train_dueling_ddqn(num_episodes=num_episodes, model_path=model_path, enable_dueling_dqn=True, enable_double_dqn=True, hyperparameter_set="dueling_ddqn")
        elif "--double_dqn" in sys.argv:
            train_dueling_ddqn(num_episodes=num_episodes, model_path=model_path, enable_dueling_dqn=False, enable_double_dqn=True, hyperparameter_set="double_dqn")
        elif "--dqn" in sys.argv:
            train_dueling_ddqn(num_episodes=num_episodes, model_path=model_path, enable_dueling_dqn=False, enable_double_dqn=False, hyperparameter_set="dqn")
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
        model_path = None
        if "--model_path" in sys.argv:
            idx = sys.argv.index("--model_path")
            if idx + 1 < len(sys.argv):
                model_path = sys.argv[idx + 1]

        if "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                render=render,
                enable_dueling_dqn=True,
                enable_double_dqn=True,
                hyperparameter_set="dueling_ddqn",
                model_path=model_path,
            )
            print(f"[Dueling Double DQN Evaluation] {metrics}")
        elif "--double_dqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                render=render,
                enable_dueling_dqn=False,
                enable_double_dqn=True,
                hyperparameter_set="double_dqn",
                model_path=model_path,
            )
            print(f"[Double DQN Evaluation] {metrics}")
        elif "--dqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                render=render,
                enable_dueling_dqn=False,
                enable_double_dqn=False,
                hyperparameter_set="dqn",
                model_path=model_path,
            )
            print(f"[Vanilla DQN Evaluation] {metrics}")
        elif "--ppo" in sys.argv or "-ppo" in sys.argv:
            metrics = evaluate_ppo(render=render, hyperparameter_set="ppo", model_path=model_path)
            print(f"[PPO Evaluation] {metrics}")
        else:
            print(
                "Please specify a learning agent to evaluate:\n"
                "  python -m src --eval --dueling_ddqn [--render] # Evaluate Dueling Double DQN [alias: --ddqn]\n"
                "  python -m src --eval --double_dqn   [--render] # Evaluate Double DQN\n"
                "  python -m src --eval --dqn          [--render] # Evaluate Vanilla DQN\n"
                "  python -m src --eval --ppo          [--render] # Evaluate PPO [alias: -ppo]"
            )
    else:
        print(
            "Usage:\n"
            "  python -m src --play [--blue|--orange|--bot-vs-bot] # Play interactively (default: control Blue car)\n"
            "  python -m src --play --headless                     # Run headless physics benchmark\n"
            "  python -m src --demo --gym                          # Demo Gymnasium environment\n"
            "  python -m src --demo --dqn|--double_dqn|--ddqn      # Test DQN neural network variants\n"
            "  python -m src --demo --ppo                          # Test PPO Actor-Critic network\n"
            "  python -m src --train --dueling_ddqn|--ppo          # Train Dueling Double DQN or PPO agent\n"
            "  python -m src --eval --dueling_ddqn|--ppo [--render] # Evaluate Dueling Double DQN or PPO agent"
        )


if __name__ == "__main__":
    main()
