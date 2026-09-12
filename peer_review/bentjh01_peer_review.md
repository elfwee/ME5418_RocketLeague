# Thermal and Payload Aware Control of a Two Joint Robot Arm

Strengths:
- Starting from a simple 2-link reacher, added with uncertain payload sensor and motor overheating creates an interesting resource management problem between motor control and payload/heat constraint. This provides reasonable motivation for RL due to delayed consequences from motor use.
- The proposed experiments are strong, with comparisons against manual controller and tests on payload generalization and sensing errors.
- Problem statement is clear and precise, with well casted state, action and reward criteria. 
- Using accumulated motor torque as temperature models is reasonable, without the need to model detailed thermodynamics.
- For difficulty, the 2-link arm started relatively simple, but the added payload uncertainty, thermal constraints, sensor errors, and generalization experiments make the overall project challenging.

Suggestions:
- The main concern is the heavy discretization of the state and action space. Discretizing them may lose useful information and may cause imprecise and jerky arm motion.
- The project seems to designed to make DQN feasible. It maybe worth clarifying why DQN was preferred over policy gradient or action-critic approach, which can handle continuous observations and actions.
- The reward contains multiple shaping terms. Balancing many reward terms will require immense amount of trial and error and may introduce unintended agent behaviours.

# Proposal_Group35_CROSS A Mixture-of-Expert Reinforcement Learning Framework for Generalized Large-Scale Traffic Signal Control
Reviewer: Elfred

Strengths:
- The proposal effectively highlights RL's suitability for ASTC, noting that dynamic traffic patterns and heterogenous intersection layouts make it difficult for a single shared policy to succeed.
- The proposal points out that conventional algorithms rely on predefined rules and fail to learn specialized strategies from experience, which strongly justifies the use of RL.
- The problem statement is clear and well-structured, with an individual agent to each intersection while detailing the local state observations and action spaces.
- Comparing the CROSS framework against a mix of methods and conventional baselines across multiple metrics is an excellent way to validate the approach.

Suggestions:
- It would be helpful to spell out acronyms (e.g., PPO, GRU, MLP, SUMO) upon their first use to ensure clarity for readers.
- While the proposal mentions using a predictive clustering module to route MLP experts, providing a more detailed description of how these experts will be trained and specialized for different traffic patterns would help with the methodology section.
- The proposla could further elborate on why predefined control rules fail for this context. For example, which traffic scenario requires experience-based strategies to overcome and not Max-Pressure rules.
 this project sounds a little too difficult to be building from scratch. Seems like it would rely heavily on libraries
