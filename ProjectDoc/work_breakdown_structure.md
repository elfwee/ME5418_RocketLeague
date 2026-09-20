0.0 2D Soccer Reinforcement Learning Project
│
├── 1.0 Project Planning & System Design
│   ├── 1.1 Define project objectives and scope
│   ├── 1.2 Define system architecture
│   ├── 1.3 Define development milestones
│   ├── 1.4 Define experiment methodology
│   └── 1.5 Set up repository / development workflow
│
├── 2.0 Gym Environment Development
│   │
│   ├── 2.1 Arena Development
│   │   ├── 2.1.1 Create 2D enclosed arena
│   │   ├── 2.1.2 Implement non-uniform boundaries
│   │   ├── 2.1.3 Implement elevated goals
│   │   └── 2.1.4 Implement goal detection
│   │
│   ├── 2.2 Vehicle Development
│   │   ├── 2.2.1 Implement vehicle movement
│   │   ├── 2.2.2 Implement jumping
│   │   ├── 2.2.3 Implement second jump / dodge
│   │   ├── 2.2.4 Implement boost
│   │   └── 2.2.5 Implement vehicle collision physics
│   │
│   ├── 2.3 Ball Development
│   │   ├── 2.3.1 Implement ball physics
│   │   ├── 2.3.2 Implement ball-wall collisions
│   │   └── 2.3.3 Implement vehicle-ball interactions
│   │
│   ├── 2.4 Episode Management
│   │   ├── 2.4.1 Define initial states
│   │   ├── 2.4.2 Implement environment reset
│   │   ├── 2.4.3 Implement scoring events
│   │   └── 2.4.4 Implement episode termination
│   │
│   └── 2.5 Gym Environment Testing
│       ├── 2.5.1 Test physics
│       ├── 2.5.2 Test collisions
│       ├── 2.5.3 Test controls
│       └── 2.5.4 Test reset / termination logic
│
│       ◆ MILESTONE 1:
│         Gym Environment Complete — 16 October
│
├── 3.0 RL Interface Development
│   │
│   ├── 3.1 State / Observation Space
│   │   ├── 3.1.1 Agent kinematic state
│   │   ├── 3.1.2 Opponent kinematic state
│   │   ├── 3.1.3 Ball kinematic state
│   │   ├── 3.1.4 Relative direction / distance features
│   │   ├── 3.1.5 Boost availability
│   │   ├── 3.1.6 Second-jump availability
│   │   ├── 3.1.7 Grounded / aerial state
│   │   └── 3.1.8 Eight boundary raycasts
│   │
│   ├── 3.2 Action Space
│   │   ├── 3.2.1 Direction: 9 actions
│   │   ├── 3.2.2 Jump: 2 actions
│   │   ├── 3.2.3 Boost: 2 actions
│   │   └── 3.2.4 Validate 36 action combinations
│   │
│   └── 3.3 Reward Function
│       ├── 3.3.1 Goal reward
│       ├── 3.3.2 Goal-conceded penalty
│       ├── 3.3.3 Ball progress reward
│       ├── 3.3.4 Defensive clearing reward
│       └── 3.3.5 Reward scaling / balancing
│
├── 4.0 Neural Network Development
│   │
│   ├── 4.1 Design MLP architecture
│   ├── 4.2 Implement shared / feature layers
│   │
│   ├── 4.3 PPO Actor
│   │   ├── 4.3.1 Direction probability head
│   │   ├── 4.3.2 Jump probability head
│   │   └── 4.3.3 Boost probability head
│   │
│   ├── 4.4 PPO Critic
│   │   └── 4.4.1 State-value output
│   │
│   ├── 4.5 DQN Network
│   │   └── 4.5.1 Q-value outputs for comparison
│   │
│   └── 4.6 Network Testing
│       ├── 4.6.1 Verify input dimensions
│       ├── 4.6.2 Verify output dimensions
│       └── 4.6.3 Verify forward/backward passes
│
│       ◆ MILESTONE 2:
│         Neural Network Complete — 30 October
│
├── 5.0 Learning Agent Development
│   │
│   ├── 5.1 PPO Implementation
│   │   ├── 5.1.1 Action sampling
│   │   ├── 5.1.2 Trajectory collection
│   │   ├── 5.1.3 Return calculation
│   │   ├── 5.1.4 Advantage estimation
│   │   ├── 5.1.5 PPO clipped objective
│   │   ├── 5.1.6 Critic/value loss
│   │   └── 5.1.7 Policy update loop
│   │
│   ├── 5.2 DQN Implementation
│   │   ├── 5.2.1 Experience collection
│   │   ├── 5.2.2 Replay buffer
│   │   ├── 5.2.3 Q-learning update
│   │   └── 5.2.4 Exploration strategy
│   │
│   └── 5.3 Agent Integration
│       ├── 5.3.1 Connect observations to network
│       ├── 5.3.2 Connect network outputs to actions
│       ├── 5.3.3 Connect rewards to learning algorithm
│       └── 5.3.4 Run end-to-end learning test
│
│       ◆ MILESTONE 3:
│         Learning Agent Complete — 6 November
│
├── 6.0 Scripted Opponent Development
│   │
│   ├── 6.1 Implement opponent FSM
│   ├── 6.2 Ball chasing behaviour
│   ├── 6.3 Defensive behaviour
│   ├── 6.4 Attacking behaviour
│   ├── 6.5 Jump behaviour
│   ├── 6.6 Boost behaviour
│   ├── 6.7 Lookahead behaviour
│   └── 6.8 Define opponent difficulty levels
│
├── 7.0 Training
│   │
│   ├── 7.1 Curriculum Level 1
│   │   └── Train without opponent
│   │
│   ├── 7.2 Curriculum Level 2
│   │   └── Train against basic opponent
│   │
│   ├── 7.3 Curriculum Level 3
│   │   └── Train against stronger opponent
│   │
│   ├── 7.4 Curriculum Level 4
│   │   └── Train against full scripted opponent
│   │
│   └── 7.5 Save checkpoints and training metrics
│
├── 8.0 Experiment & Evaluation
│   │
│   ├── 8.1 Evaluate trained policies
│   │
│   ├── 8.2 Measure
│   │   ├── 8.2.1 Win rate
│   │   ├── 8.2.2 Goals scored
│   │   ├── 8.2.3 Goals conceded
│   │   └── 8.2.4 Goal differential
│   │
│   ├── 8.3 Curriculum Transfer Evaluation
│   │   └── Test policies against multiple opponent levels
│   │
│   ├── 8.4 Algorithm Comparison
│   │   └── PPO vs DQN
│   │
│   ├── 8.5 Reward Ablation
│   │   └── Shaped reward vs sparse reward
│   │
│   ├── 8.6 Robustness Evaluation
│   │   └── Observation noise testing
│   │
│   └── 8.7 Compile comparison matrix
│
└── 9.0 Final Report
    │
    ├── 9.1 Introduction / motivation
    ├── 9.2 Problem formulation
    ├── 9.3 Gym environment methodology
    ├── 9.4 RL state/action/reward formulation
    ├── 9.5 PPO/DQN methodology
    ├── 9.6 Experimental methodology
    ├── 9.7 Results and visualisations
    ├── 9.8 Discussion
    ├── 9.9 Limitations / future work
    ├── 9.10 Conclusion
    ├── 9.11 Editing and formatting
    └── 9.12 Final submission

        ◆ MILESTONE 4:
          Final Report — 22 November
