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
- to reference later:
    - RLGym/rlgym
    - arXiv:1712.01815
    - arXiv:1912.06680

# Problem Statements
The environment consists of a fully observable, gravity-bound, continuous 2D enclosed arena containing an RL-controlled car, a ball, and a scripted adversarial car. Two elevated goal regions are located on opposing sides of the arena. The agent must control its vehicle to direct the ball into the opponent's goal while simultaneously preventing the opponent from scoring in its own goal. The elevated goals and non-uniform arena boundaries require the agent to learn complex maneuvers such as dynamic positioning, aerial ball handling, and ball juggling, rather than simply pushing the ball toward the opponent's goal along the ground.

The opponent follows a fixed heuristic policy, creating a dynamic adversarial environment in which the agent must respond to both ball dynamics and opponent behaviour. The learning objective is to discover a control policy capable of combining low-level vehicle mechanics with higher-level attacking and defensive behaviours.


# RL Cast
State space: At each timestep, the agent observes the normalized kinematic states of itself, the opponent, and the ball. The agent and opponent states contain 2D position, linear velocity, orientation, and angular velocity, while the ball state contains its position, linear velocity, and angular velocity. Relative direction and distance information is also provided for the agent-to-ball, opponent-to-ball, ball-to-opponent-goal, and ball-to-own-goal relationships. The observation additionally contains boost availability, grounded/aerial state, and second-jump availability, together with eight normalized boundary raycasts separated by 45-degree intervals. These raycasts provide local awareness of the non-uniform arena boundaries without explicitly providing the complete arena geometry.

Action space: A factored multi-discrete action space consisting of direction, jump, and boost provides 9 x 2 x 2 = 36 possible simultaneous action combinations. The directional component consists of eight directions at 45-degree intervals and one neutral input, while jump and boost are binary on/off commands. A directional input combined with a second jump while airborne triggers a directional dodge.

Reward structure: The primary reward is based on the outcome of the game. The agent receives a large positive reward (e.g., +20) when it scores in the opponent's goal and a large negative reward (e.g., −20) when the opponent scores in its own goal. Since goal-scoring events are relatively sparse, a smaller intermediate reward is provided based on the ball's progress toward the opponent's goal. If the ball moves closer to the opponent's goal between consecutive timesteps, the agent receives a small positive reward proportional to that progress; if the ball moves farther away, it receives a small negative reward. The goal-scoring rewards are substantially larger than these intermediate rewards so that the agent remains primarily motivated to score goals while preventing the opponent from scoring.

# RL Algorithm
DQN, PPO, A2C etc...