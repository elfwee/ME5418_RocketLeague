flowchart TB

    P["0.0 2D Soccer Reinforcement Learning Project"]

    P --> A["1.0 Project Planning & System Design"]
    P --> B["2.0 Gym Environment Development"]
    P --> C["3.0 RL Interface Development"]
    P --> D["4.0 Neural Network Development"]
    P --> E["5.0 Learning Agent Development"]
    P --> F["6.0 Scripted Opponent Development"]
    P --> G["7.0 Training"]
    P --> H["8.0 Experiment & Evaluation"]
    P --> I["9.0 Final Report"]

    %% =========================
    %% PROJECT PLANNING
    %% =========================

    A --> A1["1.1 Define objectives & scope"]
    A --> A2["1.2 Define system architecture"]
    A --> A3["1.3 Define development milestones"]
    A --> A4["1.4 Define experimental methodology"]
    A --> A5["1.5 Set up repository & workflow"]

    %% =========================
    %% GYM ENVIRONMENT
    %% =========================

    B --> B1["2.1 Arena Development"]
    B --> B2["2.2 Vehicle Development"]
    B --> B3["2.3 Ball Development"]
    B --> B4["2.4 Episode Management"]
    B --> B5["2.5 Environment Testing"]

    B1 --> B11["2.1.1 Create 2D enclosed arena"]
    B1 --> B12["2.1.2 Implement non-uniform boundaries"]
    B1 --> B13["2.1.3 Implement elevated goals"]
    B1 --> B14["2.1.4 Implement goal detection"]

    B2 --> B21["2.2.1 Vehicle movement"]
    B2 --> B22["2.2.2 Jump mechanics"]
    B2 --> B23["2.2.3 Second jump / dodge"]
    B2 --> B24["2.2.4 Boost mechanics"]
    B2 --> B25["2.2.5 Vehicle collision physics"]

    B3 --> B31["2.3.1 Ball physics"]
    B3 --> B32["2.3.2 Ball-wall collisions"]
    B3 --> B33["2.3.3 Vehicle-ball interactions"]

    B4 --> B41["2.4.1 Define initial states"]
    B4 --> B42["2.4.2 Environment reset"]
    B4 --> B43["2.4.3 Scoring events"]
    B4 --> B44["2.4.4 Episode termination"]

    B5 --> B51["2.5.1 Physics testing"]
    B5 --> B52["2.5.2 Collision testing"]
    B5 --> B53["2.5.3 Control testing"]
    B5 --> B54["2.5.4 Reset / termination testing"]

    B --> M1{{"M1: Gym Environment Complete<br/>16 October"}}

    %% =========================
    %% RL INTERFACE
    %% =========================

    C --> C1["3.1 Observation Space"]
    C --> C2["3.2 Action Space"]
    C --> C3["3.3 Reward Function"]

    C1 --> C11["3.1.1 Agent kinematics"]
    C1 --> C12["3.1.2 Opponent kinematics"]
    C1 --> C13["3.1.3 Ball kinematics"]
    C1 --> C14["3.1.4 Relative direction & distance"]
    C1 --> C15["3.1.5 Boost availability"]
    C1 --> C16["3.1.6 Second-jump availability"]
    C1 --> C17["3.1.7 Grounded / aerial state"]
    C1 --> C18["3.1.8 Eight boundary raycasts"]

    C2 --> C21["3.2.1 Direction: 9 actions"]
    C2 --> C22["3.2.2 Jump: 2 actions"]
    C2 --> C23["3.2.3 Boost: 2 actions"]
    C2 --> C24["3.2.4 Validate 36 action combinations"]

    C3 --> C31["3.3.1 Goal reward"]
    C3 --> C32["3.3.2 Goal-conceded penalty"]
    C3 --> C33["3.3.3 Ball progress reward"]
    C3 --> C34["3.3.4 Defensive clearing reward"]
    C3 --> C35["3.3.5 Reward scaling"]

    %% =========================
    %% NEURAL NETWORK
    %% =========================

    D --> D1["4.1 Design MLP architecture"]
    D --> D2["4.2 Implement feature layers"]
    D --> D3["4.3 PPO Actor"]
    D --> D4["4.4 PPO Critic"]
    D --> D5["4.5 DQN Network"]
    D --> D6["4.6 Network Testing"]

    D3 --> D31["4.3.1 Direction probability head"]
    D3 --> D32["4.3.2 Jump probability head"]
    D3 --> D33["4.3.3 Boost probability head"]

    D4 --> D41["4.4.1 State-value output"]

    D5 --> D51["4.5.1 Q-value outputs"]

    D6 --> D61["4.6.1 Verify input dimensions"]
    D6 --> D62["4.6.2 Verify output dimensions"]
    D6 --> D63["4.6.3 Verify forward/backward passes"]

    D --> M2{{"M2: Neural Network Complete<br/>30 October"}}

    %% =========================
    %% LEARNING AGENT
    %% =========================

    E --> E1["5.1 PPO Implementation"]
    E --> E2["5.2 DQN Implementation"]
    E --> E3["5.3 Agent Integration"]

    E1 --> E11["5.1.1 Action sampling"]
    E1 --> E12["5.1.2 Trajectory collection"]
    E1 --> E13["5.1.3 Return calculation"]
    E1 --> E14["5.1.4 Advantage estimation"]
    E1 --> E15["5.1.5 PPO clipped objective"]
    E1 --> E16["5.1.6 Critic / value loss"]
    E1 --> E17["5.1.7 Policy update loop"]

    E2 --> E21["5.2.1 Experience collection"]
    E2 --> E22["5.2.2 Replay buffer"]
    E2 --> E23["5.2.3 Q-learning update"]
    E2 --> E24["5.2.4 Exploration strategy"]

    E3 --> E31["5.3.1 Observations → network"]
    E3 --> E32["5.3.2 Network → actions"]
    E3 --> E33["5.3.3 Rewards → learning algorithm"]
    E3 --> E34["5.3.4 End-to-end learning test"]

    E --> M3{{"M3: Learning Agent Complete<br/>6 November"}}

    %% =========================
    %% OPPONENT
    %% =========================

    F --> F1["6.1 Implement opponent FSM"]
    F --> F2["6.2 Ball chasing"]
    F --> F3["6.3 Defensive behaviour"]
    F --> F4["6.4 Attacking behaviour"]
    F --> F5["6.5 Jump behaviour"]
    F --> F6["6.6 Boost behaviour"]
    F --> F7["6.7 Lookahead behaviour"]
    F --> F8["6.8 Opponent difficulty levels"]

    %% =========================
    %% TRAINING
    %% =========================

    G --> G1["7.1 Curriculum Level 1<br/>No opponent"]
    G --> G2["7.2 Curriculum Level 2<br/>Basic opponent"]
    G --> G3["7.3 Curriculum Level 3<br/>Stronger opponent"]
    G --> G4["7.4 Curriculum Level 4<br/>Full opponent"]
    G --> G5["7.5 Save checkpoints & metrics"]

    %% =========================
    %% EVALUATION
    %% =========================

    H --> H1["8.1 Evaluate trained policies"]
    H --> H2["8.2 Performance Metrics"]
    H --> H3["8.3 Curriculum transfer evaluation"]
    H --> H4["8.4 PPO vs DQN"]
    H --> H5["8.5 Reward ablation"]
    H --> H6["8.6 Observation-noise robustness"]
    H --> H7["8.7 Comparison matrix"]

    H2 --> H21["8.2.1 Win rate"]
    H2 --> H22["8.2.2 Goals scored"]
    H2 --> H23["8.2.3 Goals conceded"]
    H2 --> H24["8.2.4 Goal differential"]

    %% =========================
    %% REPORT
    %% =========================

    I --> I1["9.1 Introduction & motivation"]
    I --> I2["9.2 Problem formulation"]
    I --> I3["9.3 Environment methodology"]
    I --> I4["9.4 RL formulation"]
    I --> I5["9.5 PPO / DQN methodology"]
    I --> I6["9.6 Experimental methodology"]
    I --> I7["9.7 Results & visualisations"]
    I --> I8["9.8 Discussion"]
    I --> I9["9.9 Limitations & future work"]
    I --> I10["9.10 Conclusion"]
    I --> I11["9.11 Editing & formatting"]
    I --> I12["9.12 Final submission"]

    I --> M4{{"M4: Final Report<br/>22 November"}}
