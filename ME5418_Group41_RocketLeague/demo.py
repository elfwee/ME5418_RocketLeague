"""
ME5418 Project Milestone 2 - Rocket League Neural Network (Dueling DQN) Demo
Entry point demonstrating Dueling DQN instantiation and a complete headless
forward/backward pass with input/output tensors and gradient verification.
"""
import os
import sys

# Ensure src can be imported directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.dqn.model import run_network


if __name__ == "__main__":
    run_network()
