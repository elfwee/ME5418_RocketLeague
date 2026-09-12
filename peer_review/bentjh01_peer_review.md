# Reinforcement Learning Robust Stair-Climbing Skills for Open Duck Mini through Reward Shaping and Curriculum Learning

## Motivation
- Good to highlight that IK struggles with 
- should state what ZMP is and add a reference. 
## Problem Statement
## Difficulty

# Thermal and Payload Aware Control of a Two Joint Robot Arm

## Motivation
- Starting from a simple 2-link reacher, added with uncertain payload sensor and motor overheating creates an interesting resource management problem between motor control and payload/heat constraint. This provides reasonable motivation for RL due to delayed consequences from motor use.
- The proposed experiments are strong, with comparisons against manual controller and tests on payload generalization and sensing errors.
## Problem Statement & RL Cast
- Problem statement is clear and precise, with well casted state, action and reward criteria. 
- Using accumulated motor torque as temperature models is reasonable, without the need to model detailed thermodynamics.
## Difficulty
- The 2-link arm is relatively simple, but the added payload uncertainty, thermal constraints, sensor errors, and generalization experiments make the overall project challenging.
## Suggestion
- The main concern is the heavy discretization of the state and action space. Discretizing them may lose useful information and may cause imprecise and jerky arm motion.
- The project seems to designed to make DQN feasible. It maybe worth clarifying why DQN was preferred over policy gradient or action-critic approach, which can handle continuous observations and actions.
- The reward contains multiple shaping terms. Balancing many reward terms will require immense amount of trial and error and may introduce unintended agent behaviours.