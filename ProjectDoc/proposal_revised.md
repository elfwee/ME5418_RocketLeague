# Autonomous Decision-Making Under Dynamic and Adversarial Constraints for 2D Soccer

_Project Proposal for Group 41_
_Benjamin Teh, Jensen Lu, Wee Fook Choon_

# Motivation

Dynamic target interception and non-prehensile manipulation in multi-agent environments are fundamental challenges in
mobile robotics. These autonomous agents range from the low-velocity, high-inertia tugboats used for ship berthing [2]
to high-velocity, low-inertia unmanned aerial vehicles [4]. One less safety-critical application of these challenges is
robot soccer [1].

Inspired by the popular video game Rocket League, this project seeks to develop a policy for dynamic target interception
and non-prehensile manipulation in a multi-agent environment. Here, the agent must knock a ball into the opponent's goal
while simultaneously preventing the opponent from doing the same.

To simplify the problem, this project uses a 2D planar robot soccer framework. However, several key challenges are still
apparent from its 3D counterpart, namely, indirect manipulation of the ball, dynamic adversarial interaction, and
alternating dynamics between continuous motion and discontinuous impact. The simplified environment therefore serves as
a controlled testbed for studying several challenges shared with robotic systems: dynamic target interception,
non-prehensile object manipulation, contact-rich control, and sequential decision-making in the presence of another
autonomous agent.

# Conventional Algorithms

A traditional approach for robot soccer is using finite state machines, where the robot switches between states such as
defending the goal, chasing the ball, and attempting a shot, depending on user-defined conditions based on sensor
information. This is a highly interpretable approach but is limited by the programmer's knowledge of possible situations
and is susceptible to omitted edge cases.

Within each state, the robot's actions can be determined using computationally efficient rule-based controllers or more
sophisticated planning and control approaches such as model predictive control (MPC). MPC optimizes actions over a
finite prediction horizon using a model of the system dynamics and an objective function. While these approaches enable
online planning, their computational requirements and design complexity can increase considerably when accounting for
continuous vehicle-ball interactions and the actions of an adversarial opponent.

Reinforcement learning (RL) provides an alternative by allowing the agent to learn its control policy through repeated
interaction with the environment. During training, the agent explores the state-action space and can encounter
situations that may not have been explicitly considered when designing a rule-based controller, reducing the need to
manually specify behaviours for every possible scenario. RL optimizes for cumulative discounted rewards over time,
allowing actions to be learned based on their longer-term consequences rather than only their immediate outcomes. Once
trained, these behaviours are represented by the learned policy, allowing actions to be selected directly during runtime
without repeatedly performing an online search over future trajectories.

# Problem Statements

The environment consists of a gravity-bound, side-view 2D enclosed arena containing an RL-controlled car, a
ball, and a scripted adversarial car. Two elevated goal regions are located on opposing sides of the arena. This poses 
as a continous state environment for the agent to explore. The agent
must control its vehicle to direct the ball into the opponent's goal while simultaneously preventing the opponent from
scoring in its own goal. The elevated goals and non-uniform arena boundaries require the agent to learn complex
maneuvers such as dynamic positioning, aerial ball handling, and ball juggling, rather than simply pushing the ball
toward the opponent's goal along the ground, which would cause it to rebound from the wall beneath the elevated goal.

The opponent follows a finite state machine, creating a dynamic adversarial environment in which the agent must respond
to both ball dynamics and opponent behaviour. The learning objective is to discover a control policy capable of
combining low-level vehicle mechanics with high-level attacking and defensive behaviours. Observation noise may be
introduced during evaluation to investigate robustness to imperfect state estimation.

# RL Cast

State space: At each timestep, the agent observes the following:

- Kinematic state of itself and opponent:
    - in terms of position, orientation, linear velocity, and angular velocity
- Kinematic state of ball:
    - in terms of position, linear velocity, and angular velocity
- Relative direction and distance for the following relationships:
    - agent-to-ball
    - opponent-to-ball
    - ball-to-opponent's goal
    - ball-to-own goal
- Other information:
    - Availability of boost control
    - Availability of second-jump
    - Grounded/aerial state
    - 8 normalized boundary raycasts separated by 45-degree intervals

These raycasts provide local awareness of the non-uniform arena boundaries without the need to directly encode the
static arena geometry.

Action space: A factored multi-discrete action space consisting of direction, jump, and boost provides 9 x 2 x 2 = 36
possible simultaneous action combinations. The directional component consists of eight directions at 45-degree intervals
and one neutral input, while jump and boost are binary on/off commands. A directional input combined with a second jump
while airborne triggers a directional dodge.

Reward structure: The primary reward is based on the outcome of the game. The agent receives a large positive reward
(e.g., +20) when it scores in the opponent's goal and a large negative reward (e.g., -20) when the opponent scores in
its own goal. Since goal-scoring events are relatively sparse, smaller intermediate rewards are provided based on the
ball's movement and position within the arena. Progress toward the opponent's goal is positively rewarded, while
movement toward the agent's own goal is penalized. The magnitude of these intermediate rewards is scaled according to
the ball's position, such that moving the ball toward its own goal becomes increasingly penalized as the ball approaches
it, while clearing the ball away from its own goal gets rewarded. The goal-scoring rewards are substantially larger than
the intermediate rewards so that scoring and preventing goals remain the agent's primary objectives.

# RL Algorithm

- Our problem has a continuous state space and a relatively small
  discrete action space, so both value-based and policy-based
  algorithms, such as DQN and PPO, are applicable.
- DQN estimates the expected return of each discrete action and
  selects the action with the highest estimated value. With 36 closely
  related action combinations, approximation errors between similarly
  valued actions may lead to abrupt changes in the selected action,
  particularly during early training.
- PPO instead learns a stochastic policy over the available actions.
  For our factored action space, the actor can output separate
  probability distributions for direction, jump, and boost, while the
  critic estimates the expected return of the current state.
- We will use PPO as the primary algorithm because it supports
  stochastic exploration and is generally stable to train with an
  actor-critic architecture. Although on-policy PPO is less
  sample-efficient than off-policy methods, our simulation can be
  initialized cheaply and run in parallel to collect experience.
- DQN will be used as a comparison to investigate whether the
  policy-based formulation is more suitable for this task.
- Neural network: an MLP actor-critic architecture will process the
  observation vector. The actor outputs the action distributions,
  while the critic outputs a scalar state-value estimate.

# Experiment & Evaluation

- Training will use a curriculum of increasing opponent difficulty.
  The initial stage contains no adversarial opponent, allowing the
  agent to first learn basic movement, ball interaction, and scoring.
  An opponent is then introduced with progressively stronger scripted
  behaviours.
- The lowest-level opponent will primarily follow the ball with
  limited use of jumping and boosting. More capable opponents will use
  scripted lookahead of the future ball and vehicle states. Opponent
  difficulty can be varied by changing the lookahead horizon and
  available behaviours.
- Policies trained at each curriculum level will be evaluated against
  opponents from multiple difficulty levels. The results will be
  summarized using a comparison matrix to determine whether policies
  learned at one level transfer to more difficult or different
  opponents.
- Performance will be evaluated using match win rate, goals scored,
  goals conceded, and goal differential over repeated matches.
- We will compare the proposed shaped reward against a sparse-reward
  baseline containing only the goal-scoring and goal-conceding
  rewards, to evaluate whether the intermediate rewards improve
  learning.
- Since the action space is discrete, we will also compare PPO and DQN
  under the same environment and evaluation conditions to investigate
  which formulation is more suitable for the task.
- Observation noise may additionally be introduced during evaluation
  to test robustness to imperfect state estimation.

# References:

[1] Taourirte, Aya, and Md Sohag Mia. "Multi-Agent Reinforcement Learning and Real-Time Decision-Making in Robotic
Soccer for Virtual Environments." arXiv, 2025. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.2512.03166.

[2] Oh, Jaejin, and Jongdae Jung. "Contact-Based Cooperative Tugboat-Assisted Ship Berthing Control via Physics-Informed
Reinforcement Learning." Ocean Engineering, vol. 363, Aug. 2026, p. 126675. DOI.org (Crossref),
https://doi.org/10.1016/j.oceaneng.2026.126675.

[3] T. M. Cao, H. A. Pham, M. Walter, V. Gies and T. Soriano, "Multi-Agent Robot Swarms: A Review of Sensing and
Perceptual Strategies for RoboCup Soccer," 2025 11th International Conference on Mechatronics and Robotics Engineering
(ICMRE), Lille, France, 2025, pp. 126-131, doi: 10.1109/ICMRE64970.2025.10976285.

[4] Brust, Matthias R., et al. "Swarm-Based Counter UAV Defense System." Discover Internet of Things, vol. 1, no. 1,
Dec. 2021, p. 2. DOI.org (Crossref), https://doi.org/10.1007/s43926-021-00002-x.
