flowchart LR

    START(("Start"))

    A["A<br/>Requirements &<br/>System Design"]

    START --> A

    %% =====================================
    %% PARALLEL EARLY DEVELOPMENT
    %% =====================================

    A --> B["B<br/>Arena & Physics"]
    A --> C["C<br/>RL State &<br/>Action Design"]
    A --> D["D<br/>Scripted Opponent<br/>FSM Design"]

    %% =====================================
    %% ENVIRONMENT PATH
    %% =====================================

    B --> E["E<br/>Vehicle & Ball<br/>Mechanics"]

    E --> F["F<br/>Goals, Reset &<br/>Termination"]

    F --> G["G<br/>Environment<br/>Testing"]

    G --> M1{{"M1<br/>GYM ENVIRONMENT<br/>16 OCT"}}

    %% =====================================
    %% RL INTERFACE PATH
    %% =====================================

    C --> H["H<br/>Observation<br/>Implementation"]

    C --> I["I<br/>Action Space<br/>Implementation"]

    C --> J["J<br/>Reward Function<br/>Implementation"]

    %% =====================================
    %% NETWORK DEVELOPMENT
    %% =====================================

    H --> K["K<br/>MLP Architecture"]
    I --> K

    K --> L["L<br/>PPO Actor-Critic<br/>Network"]
    K --> M["M<br/>DQN Network"]

    L --> N["N<br/>Network<br/>Unit Testing"]
    M --> N

    N --> M2{{"M2<br/>NEURAL NETWORK<br/>30 OCT"}}

    %% =====================================
    %% INTEGRATION
    %% =====================================

    M1 --> O["O<br/>Gym + RL Interface<br/>Integration"]

    H --> O
    I --> O
    J --> O

    %% =====================================
    %% LEARNING ALGORITHMS
    %% =====================================

    M2 --> P["P<br/>PPO Learning<br/>Algorithm"]

    M2 --> Q["Q<br/>DQN Learning<br/>Algorithm"]

    O --> P
    O --> Q

    %% =====================================
    %% OPPONENT DEVELOPMENT
    %% =====================================

    D --> R["R<br/>Opponent Behaviours:<br/>Attack / Defence"]

    R --> S["S<br/>Jump / Boost /<br/>Lookahead"]

    S --> T["T<br/>Opponent Difficulty<br/>Levels"]

    %% =====================================
    %% AGENT INTEGRATION
    %% =====================================

    P --> U["U<br/>Agent-Gym<br/>Integration"]
    Q --> U
    T --> U

    U --> V["V<br/>End-to-End<br/>Learning Test"]

    V --> M3{{"M3<br/>LEARNING AGENT<br/>6 NOV"}}

    %% =====================================
    %% CURRICULUM TRAINING
    %% =====================================

    M3 --> W1["W1<br/>Curriculum 1<br/>No Opponent"]

    W1 --> W2["W2<br/>Curriculum 2<br/>Basic Opponent"]

    W2 --> W3["W3<br/>Curriculum 3<br/>Stronger Opponent"]

    W3 --> W4["W4<br/>Curriculum 4<br/>Full Opponent"]

    %% =====================================
    %% PARALLEL EXPERIMENTS
    %% =====================================

    W4 --> X["X<br/>PPO vs DQN<br/>Comparison"]

    W4 --> Y["Y<br/>Shaped vs Sparse<br/>Reward Test"]

    W4 --> Z["Z<br/>Opponent Transfer<br/>Evaluation"]

    W4 --> AA["AA<br/>Observation Noise<br/>Robustness Test"]

    %% =====================================
    %% RESULTS
    %% =====================================

    X --> AB["AB<br/>Compile Evaluation<br/>Metrics"]
    Y --> AB
    Z --> AB
    AA --> AB

    AB --> AC["AC<br/>Comparison Matrix &<br/>Visualisations"]

    AC --> AD["AD<br/>Results Analysis &<br/>Discussion"]

    %% =====================================
    %% REPORTING
    %% =====================================

    A --> AE["AE<br/>Draft Introduction &<br/>Problem Formulation"]

    O --> AF["AF<br/>Draft Environment &<br/>RL Methodology"]

    M3 --> AG["AG<br/>Draft Learning<br/>Methodology"]

    AE --> AH["AH<br/>Assemble Final<br/>Report"]
    AF --> AH
    AG --> AH
    AD --> AH

    AH --> AI["AI<br/>Review, Editing &<br/>Formatting"]

    AI --> M4{{"M4<br/>FINAL REPORT<br/>22 NOV"}}

    M4 --> END(("Finish"))
