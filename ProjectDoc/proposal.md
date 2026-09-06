# Autonomous Decision-making Under Dynamic, Adversarial Constraints for 2D Soccer
_Project Proposal for Group 41_
*Benjamin Teh, Jensen Lu, Wee Fook Choon*

# Motivation
A soccer-like environment inspired by Rocket League presents continuous control challenges which requires the agent to perform dynamic ball interception, momemtum catching and strategic positioning to score goals (or point). The agent also needs to navigate the field with an opponent chasing after the ball to score goals, which provides additional layer of challenge for the agent to consider the next best of actions.
For this project, the environment has been reduced to 2D plane to reduce computational overhead for the training. However, this is still a non-trivial abstraction for the problem. Formulation of this problem in 2D is to include continuous state representations on ball kinetics, vehicle pose, and dynamic opponent vectors. Additionally, the agent is trained against a rule-based opponent, rather than a operating in a static environment.
This project evaluates the effectiveness of continuous reinforcement learning against established heuristic rules to discover whether superior tactics can surface which surpass human-engineered logics.

# Conventional Algorithms
- General ways to solve the problem is to provide a rules for the bots to perform certain predefined actions to score points.
- Should RL be used to solve the proble?
    - No obvious optimal policy,
    - Delayed rewards as planning over many episodes to score in needed.
    - clear desired outcome. 
- What is to be learnt:
    - Contact dynamics
    - sparse rewards
    - planning
    - multi objective optimization
    - multi agent extension
    - Continuous physics
    - momentum
    - bouncing
    - gravity
    - timing
    - strategic positioning
- Robot must learn that hitting ball in the past led to scoring. 
- search space is large given possible states, possible actions, and predicting \
where the ball will end up. 
- problem consists of:
    - physics
    - strategy
    - prediction
- Exsisting methods
    - rule-based ai
    - finite state machine
    - utility ai
    - Physics Prediction
    - montecarlo tree search algorithm. 
    - model predictive control
- possible RL methods:
    - Proximal poilicy optimization
    - Q learning
    - deep q learning 
    - soft actor critic
    - TD3
    - multiagent reinforcement learning 
    - self play
    - curriculum learning

# Problem Statements
- Continuous State
- Discrete Actions
- Agent has full observation on the map.

# RL Cast
- State space:
    - Distance of the ball from the agent.
- Action space:
    - X actions: Up, Down, Left, Right, Boost, Jump
    - Permutations allowed: (e.g., Left + Boost)
- Reward structure:
    - Staying in place is a large penalty.
    - Heading towards the ball has small reward.
    - Bringing the ball to opponent's goal gives big positive reward.
