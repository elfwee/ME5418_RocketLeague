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
from src.neural_nets.a2c_torch import test_a2c_torch
from src.neural_nets.ppo import run_ppo_network
from src.learning_agents.dueling_ddqn_agent import evaluate_dueling_ddqn, train_dueling_ddqn
from src.learning_agents.ppo_agent import evaluate_ppo, train_ppo
from src.learning_agents.head_to_head import play_head_to_head
from src.config.nn_config import PPOConfig
from src.agent.agent import train_agent, play_agent

RUNS_DIR = f"{ROOT_DIR}/runs"

DEFAULT_BLUE_MODEL = f"{RUNS_DIR}/20261106_0911/model_ppo.pt"
# DEFAULT_BLUE_MODEL = None # defaults to heuristics bot
DEFAULT_ORANGE_MODEL = f"{RUNS_DIR}/20261106_0911/model_ppo.pt"
# DEFAULT_ORANGE_MODEL = None # defaults to heuristics bot

def _arg_value(flag: str, default=None):
    """Return the value following `flag` in sys.argv, or `default` if absent."""
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        raise SystemExit(f"{flag} requires a value")
    return default


def main():
    if "--play" in sys.argv:
        if "--agent" in sys.argv:
            play_agent(_arg_value("--agent"))
        elif "--headless" in sys.argv:
            play_headless(steps=300, enable_orange=True)
        else:
            play_interactive(enable_orange=True)
    elif "--head_to_head" in sys.argv:
        play_head_to_head(
            blue_model=_arg_value("--blue", DEFAULT_BLUE_MODEL),
            orange_model=_arg_value("--orange", DEFAULT_ORANGE_MODEL),
            render="--headless" not in sys.argv,
            random_spawn="--random_spawn" in sys.argv,
            seed=int(seed) if (seed := _arg_value("--seed", None)) is not None else None,
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
        elif "--a2c" in sys.argv or "-a2c" in sys.argv:
            test_a2c_torch()
        elif "--ppo" in sys.argv or "-ppo" in sys.argv:
            run_ppo_network()
        else:
            print(
                "Please specify a demo target:\n"
                "  python -m src --demo --gym           # Gymnasium environment demo\n"
                "  python -m src --demo --dqn           # Vanilla DQN network test (PyTorch)\n"
                "  python -m src --demo --double_dqn    # Double DQN network test (PyTorch)\n"
                "  python -m src --demo --dueling_ddqn  # Dueling Double DQN network test (PyTorch) [alias: --ddqn]\n"
                "  python -m src --demo --a2c           # Advantage Actor-Critic network test (PyTorch) [alias: -a2c]\n"
                "  python -m src --demo --ppo           # Proximal Policy Optimization network test (PyTorch) [alias: -ppo]"
            )
    elif "--train" in sys.argv:
        if "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            train_dueling_ddqn(enable_dueling_dqn=True, enable_double_dqn=True, hyperparameter_set="dueling_ddqn")
        elif "--double_dqn" in sys.argv:
            train_dueling_ddqn(enable_dueling_dqn=False, enable_double_dqn=True, hyperparameter_set="double_dqn")
        elif "--dqn" in sys.argv:
            train_dueling_ddqn(enable_dueling_dqn=False, enable_double_dqn=False, hyperparameter_set="dqn")
        elif "--ppo" in sys.argv or "-ppo" in sys.argv:
            train_ppo(hyperparameter_set="ppo", num_episodes=15_000)
        elif "--a2c" in sys.argv or "-a2c" in sys.argv:
            train_agent(
                number_of_episodes=int(_arg_value("--episodes", 100_000)),
                pretrain_model_filepath=_arg_value("--pretrain"),
            )
        else:
            print(
                "Please specify a learning agent to train:\n"
                "  python -m src --train --dueling_ddqn  # Train Dueling Double DQN agent [alias: --ddqn]\n"
                "  python -m src --train --double_dqn    # Train Double DQN agent\n"
                "  python -m src --train --dqn           # Train Vanilla DQN agent\n"
                "  python -m src --train --ppo           # Train PPO agent [alias: -ppo]\n"
                "  python -m src --train --a2c [--episodes N] [--pretrain PATH]  # Train A2C agent"
            )
    elif "--eval" in sys.argv:
        render = "--render" in sys.argv
        if "--dueling_ddqn" in sys.argv or "--ddqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                render=render, enable_dueling_dqn=True, enable_double_dqn=True, hyperparameter_set="dueling_ddqn"
            )
            print(f"[Dueling Double DQN Evaluation] {metrics}")
        elif "--double_dqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                render=render, enable_dueling_dqn=False, enable_double_dqn=True, hyperparameter_set="double_dqn"
            )
            print(f"[Double DQN Evaluation] {metrics}")
        elif "--dqn" in sys.argv:
            metrics = evaluate_dueling_ddqn(
                render=render, enable_dueling_dqn=False, enable_double_dqn=False, hyperparameter_set="dqn"
            )
            print(f"[Vanilla DQN Evaluation] {metrics}")
        elif "--ppo" in sys.argv or "-ppo" in sys.argv:
            metrics = evaluate_ppo(render=render, hyperparameter_set="ppo")
            print(f"[PPO Evaluation] {metrics}")
        else:
            print(
                "Please specify a learning agent to evaluate:\n"
                "  python -m src --eval --dueling_ddqn [--render]  # Evaluate Dueling Double DQN [alias: --ddqn]\n"
                "  python -m src --eval --double_dqn [--render]    # Evaluate Double DQN\n"
                "  python -m src --eval --dqn [--render]           # Evaluate Vanilla DQN\n"
                "  python -m src --eval --ppo [--render]           # Evaluate PPO [alias: -ppo]"
            )
    else:
        print(
            "Usage:\n"
            "  python -m src --play                              # Play interactively\n"
            "  python -m src --play --headless                   # Run headless physics benchmark\n"
            "  python -m src --play --agent PATH                 # Watch a trained A2C agent play\n"
            "  python -m src --head_to_head [--headless] [--random_spawn] [--seed N] [--blue PATH] [--orange PATH]  # Agent vs agent\n"
            "  python -m src --demo --gym                        # Demo Gymnasium environment\n"
            "  python -m src --demo --dqn|--double_dqn|--ddqn    # Test DQN neural network variants\n"
            "  python -m src --demo --a2c|--ppo                  # Test A2C / PPO neural networks\n"
            "  python -m src --train --dueling_ddqn|--ppo|--a2c  # Train Dueling Double DQN, PPO or A2C agent\n"
            "  python -m src --eval --dueling_ddqn|--ppo         # Evaluate Dueling Double DQN or PPO (add --render for GUI)"
        )


if __name__ == "__main__":
    main()
