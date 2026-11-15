# Architecture
## Proximal Policy Optimization (PPO)
- relies of the concept of Actor and Critic

Actor:
- Has a network to churn out values

Critic:
- Has a network to calculate Generalized Advantage Estimation (GAE)

How to update?
Clipped surrogate objective using min function with epsilon to tune and probability ratio.

# Agent training
## Unable to learn
Observation: if directly trained from scratch against the heuristic bot, both PPO and Dueling DDQN are unable to learn. Meaning, it cannot go beyond to score.

## Result convergence
PPO convergence was a problem.

# Early problems
When switched sides, the models were unable to perform due to Pymunk solver prioritizes blue car first. Fixes are needed.

# Comparison among the models
Observation: Dueling DDQN outperforms PPO.