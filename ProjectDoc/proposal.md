# Autonomous Decision-Making Under Dynamic and Adversarial Constraints for 2D Soccer
_Project Proposal for Group 41_
*Benjamin Teh, Jensen Lu, Wee Fook Choon*

# Motivation

Dynamic target interception and nonprehensible manipulation in multi-agent environments are fundamental challenges in \
mobile robotics. These autonomous agents range from the low-velocity-high-intertia tugboats berthing a ship [2] to the \
high-velocity-low-inertia unmanned aerial vehicles [4]. One less serious application of this topic is on robot soccer [1]. 

Inspired by the popular video game Rocket League, this project seeks to create a policy for dynamic target \
interception and nonprehensible manipulation in multi-agent environments. Here, the agent would have to knock a ball \
into a goal against an opponent and vice versa.

To simplify the problem, this project uses a 2D planar robot soccer framework. However, the key challenges are still \
apparent from it's 3D counterpart. Namely, indirecti actuation of the ball, adversarial non-stationarity, and \
alternating dynamics between continous flight and discontinuous impact. 

# Conventional Algorithms

A traditional approach for robot soccer is finite state machines where the robot switches between states such as \
defend the goal, chase the ball, and attempting a shot where the robot switches between depending on user defined \
conditions based on sensor information. This in a highly interpretable approach but is limited to the capability of the \
programmer's knowledge space and is suceptiable to edge cases to be omitted. 

Within each state machine, the robots actions can be defined by rule-based approaches which are computationally efficient \
but limited in complexity, but also more complex approaches such as dynamic window approach (DWA) and model predictive \
control (MPC) approaches. In DWA, a selection of fixed dynamics (throttle, jump, boost) is applied to a lookahead time \
(the window), then the reward is calculated for that window based on a defined reward funtion, and finally the dynamics \
with the highest reward is selected for the next time-step. In MPC, the programmer would have to formulate the equations \
that describe the dynamics of that particular task such as scoring a goal, so that the MPC can optimise for it. Both \
approaches are computationally intensive and balloon in complexity when in an advisarial environment, thus struggles to \
be real-time. 

Reinforcement learning (RL) is an alternative approach that eliviates the issues presented by traditional approaches. \
During training, reinforcement learning explores the state-action space which which allows the agent to find niche states \
that may be omitted by a programmer thus bypassing the need for extensive tuning of parameters and states. During runtime, \
RL optimises for the maximum return of state-action pairs which includes the discounted rewards across a long time horizon. \
This means that the long term return is encoded in the state which reduces the computational intensity, and is able to \
optimise for a longer time horizon. 

# Problem Statements
The environment consists of a fully observable, gravity-bound, continuous 2D enclosed arena containing an RL-controlled\
car, a ball, and a scripted adversarial car. Two elevated goal regions are located on opposing sides of the arena. The \
agent must control its vehicle to direct the ball into the opponent's goal while simultaneously preventing the opponent\
from scoring in its own goal. The elevated goals and non-uniform arena boundaries require the agent to learn complex \
maneuvers such as dynamic positioning, aerial ball handling, and ball juggling, rather than simply pushing the ball \
toward the opponent's goal along the ground, which would cause it to rebound from the wall beneath the elevated goal.

The opponent follows a fixed heuristic policy, creating a dynamic adversarial environment in which the agent must \
respond to both ball dynamics and opponent behaviour. The learning objective is to discover a control policy capable of\
combining low-level vehicle mechanics with higher-level attacking and defensive behaviours.


# RL Cast
State space: At each timestep, the agent observes the normalized kinematic states of itself, the opponent, and the ball.\
 The agent and opponent states contain 2D position, linear velocity, orientation, and angular velocity, while the ball \
 state contains its position, linear velocity, and angular velocity. Relative direction and distance information is also \
 provided for the agent-to-ball, opponent-to-ball, ball-to-opponent-goal, and ball-to-own-goal relationships. The \
 observation additionally contains boost availability, grounded/aerial state, and second-jump availability, together \
 with eight normalized boundary raycasts separated by 45-degree intervals. These raycasts provide local awareness of the\
 non-uniform arena boundaries without the need to directly encode the static arena geometry.

Action space: A factored multi-discrete action space consisting of direction, jump, and boost provides 9 x 2 x 2 = 36 \
possible simultaneous action combinations. The directional component consists of eight directions at 45-degree intervals\
and one neutral input, while jump and boost are binary on/off commands. A directional input combined with a second jump \
while airborne triggers a directional dodge.

Reward structure: The primary reward is based on the outcome of the game. The agent receives a large positive reward\
(e.g., +20) when it scores in the opponent's goal and a large negative reward (e.g., −20) when the opponent scores in\
its own goal. Since goal-scoring events are relatively sparse, smaller intermediate rewards are provided based on the\
ball's movement and position within the arena. Progress toward the opponent's goal is positively rewarded, while movement\
toward the agent's own goal is penalized. The magnitude of these intermediate rewards is scaled according to the ball's\
position, such that moving the ball toward its own goal becomes increasingly penalized as the ball approaches it, while\
clearing the ball away from its own goal gets rewarded. The goal-scoring rewards are substantially larger than\
the intermediate rewards so that scoring and preventing goals remain the agent's primary objectives.

# RL Algorithm
DQN, PPO, A2C etc...

# References:
[1] Taourirte, Aya, and Md Sohag Mia. “Multi-Agent Reinforcement Learning and Real-Time Decision-Making in Robotic Soccer for Virtual Environments.” arXiv, 2025. DOI.org (Datacite), https://doi.org/10.48550/ARXIV.2512.03166.
[2] Oh, Jaejin, and Jongdae Jung. “Contact-Based Cooperative Tugboat-Assisted Ship Berthing Control via Physics-Informed Reinforcement Learning.” Ocean Engineering, vol. 363, Aug. 2026, p. 126675. DOI.org (Crossref), https://doi.org/10.1016/j.oceaneng.2026.126675.
[3] T. M. Cao, H. A. Pham, M. Walter, V. Gies and T. Soriano, "Multi-Agent Robot Swarms: A Review of Sensing and Perceptual Strategies for RoboCup Soccer," 2025 11th International Conference on Mechatronics and Robotics Engineering (ICMRE), Lille, France, 2025, pp. 126-131, doi: 10.1109/ICMRE64970.2025.10976285.
[4] Brust, Matthias R., et al. “Swarm-Based Counter UAV Defense System.” Discover Internet of Things, vol. 1, no. 1, Dec. 2021, p. 2. DOI.org (Crossref), https://doi.org/10.1007/s43926-021-00002-x.
 
