# Neural Network learning points
Benjamin Teh
## Actor-Crtic
### Understanding:
- Actor
    - A neural network that takes in the observation and outputs natural logrithmic units (ln(probability/(1-probability)) aka logits)
    - hidden layers usually use tanh activation function outputs (-1, 1), and is smooth and continuous
    - The logits can be converted to probabilites of that action in the action space by applying a softmax i.e. sum of all probabilities across all actions = 1. 
- Critic
    - Value Function approximator of Temporal difference. Approximates the value of (R_t+1 + gamma * v(S_t+1)) as V(S_t+1).
    - The Value Output is then used as the target for the policy to calculate its loss. 
- Overall intuition
    - Actor-Crtic uses bootstrapping where the policy can be updated online without an episode terminating. 
### Implementation:
- Shared vs Independent Trunk
    - what: both policy and vallue networks share identical parameters vs independent parameters
    - why: 
        - is useful when both value and policy networks need to learn the same general features in the data. 
        - computationally efficient
    - why not: 
        - sharing parameters also means that there is gradient interference when there are conflicting features.
        - scale mismatch. the value targets can by much larger than policy gradient. You'd need to normalise the return or apply a coefficient. 
- Hybrid
    - Shared hidden layers + independent layers
        - Shared hidden layers learn the general features then the private layers to specialise to objective. 
        - Good for image image data. 
    - Phasic Policy Gradient (PPG). Shared hidden layers for N episodes, then split to independent episodes for the remainder of the training. 
        - High initial signal to learn general features then specialise i.e. pretraining then specialise
