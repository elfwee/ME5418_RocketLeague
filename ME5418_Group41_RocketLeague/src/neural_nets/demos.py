"""Assignment 2 demos: build each network, run one forward pass on a real observation and one backward pass.

python -m src --demo --dqn | --double_dqn | --dueling_ddqn | --ppo
"""

import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F

import src.gym_env.rocket_league_env  # noqa: F401  (import registers the gym env id)
from src.config.gym_config import GymConfig
from src.config.nn_config import DQNConfig, PPOConfig
from src.neural_nets.dueling_ddqn_torch import DQN
from src.neural_nets.ppo_torch import ActorCritic


def run_dqn_network(enable_dueling_dqn: bool = True, enable_double_dqn: bool = True):
    """Assignment 2 Demo: Network Instantiation, Forward Pass & Backward Pass (Headless)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(GymConfig.gym_id, render_mode=None, bot_type="heuristic_bot", discrete_action=True)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    hidden_dim = DQNConfig.hidden_size

    # 1. Create Policy and Target DQN Networks
    policy_dqn = DQN(num_states, num_actions, hidden_dim, enable_dueling_dqn=enable_dueling_dqn).to(device)
    target_dqn = DQN(num_states, num_actions, hidden_dim, enable_dueling_dqn=enable_dueling_dqn).to(device)
    target_dqn.load_state_dict(policy_dqn.state_dict())
    target_dqn.eval()

    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=DQNConfig.learning_rate)
    criterion = nn.SmoothL1Loss()  # Huber loss for Temporal-Difference (TD) error
    gamma = DQNConfig.discount_factor

    arch_label = (
        "DUELING DOUBLE DQN"
        if (enable_dueling_dqn and enable_double_dqn)
        else ("DOUBLE DQN" if enable_double_dqn else "VANILLA DQN")
    )
    total_params = sum(p.numel() for p in policy_dqn.parameters() if p.requires_grad)

    print("=" * 65)
    print(f"1. NETWORK ARCHITECTURE ({arch_label})")
    print("=" * 65)
    print(
        f"Device: {device} | State Dim: {num_states} | Action Dim: {num_actions} | "
        f"Hidden Dim: {hidden_dim} | Trainable Params: {total_params:,}"
    )
    print(policy_dqn)

    # 2. Forward Pass with Environment Observation
    obs, info = env.reset()
    state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 52)

    q_values = policy_dqn(state_tensor)  # Shape: (1, 36)
    action_idx = q_values.argmax(dim=-1).item()

    # Decode discrete action index (0..35) into (direction, jump, boost)
    dir_idx, jump_bit, boost_bit = action_idx // 4, (action_idx % 4) // 2, action_idx % 2

    print("\n" + "=" * 65)
    print("2. FORWARD PASS (INPUT -> OUTPUT)")
    print("=" * 65)
    print(f"Input State Shape : {state_tensor.shape}")
    print(f"Input State (first 8 features): {state_tensor[0, :8].detach().cpu().numpy().round(4)}")
    print(f"Output Q(s, .) Shape: {q_values.shape}")
    print(f"Output Q(s, .) Sample (first 6 actions): {q_values[0, :6].detach().cpu().numpy().round(4)}")
    print(
        f"Selected Action   : {action_idx} -> dir={GymConfig.directions[dir_idx]}, jump={bool(jump_bit)}, boost={bool(boost_bit)}"
    )

    # Step environment once with the selected action to observe transition (s, a, r, s', done)
    next_state, reward, terminated, truncated, _ = env.step(action_idx)
    done = terminated or truncated
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

    # 3. Backward Pass (TD Target, Loss Computation & Gradient Backpropagation)
    with torch.no_grad():
        if enable_double_dqn:
            # Double DQN target: select best next action using policy network, evaluate using target network
            next_best_action = policy_dqn(next_state_tensor).argmax(dim=-1, keepdim=True)
            next_q_value = target_dqn(next_state_tensor).gather(1, next_best_action).squeeze(1)
        else:
            # Vanilla DQN target: target network both selects and evaluates max_a' Q_target(s', a')
            next_q_value = target_dqn(next_state_tensor).max(dim=-1)[0]
        # Bootstrap on time-limit truncation; only zero out future value on true goal termination
        q_target = (
            torch.tensor([reward], dtype=torch.float32, device=device)
            + gamma * (1.0 - float(terminated)) * next_q_value
        )

    # Predicted Q(s, a) for the executed action
    q_pred = q_values.gather(1, torch.tensor([[action_idx]], dtype=torch.long, device=device)).squeeze(1)

    # Compute Huber TD loss and backpropagate gradients
    loss = criterion(q_pred, q_target)
    optimizer.zero_grad()
    loss.backward()

    print("\n" + "=" * 65)
    print("3. BACKWARD PASS (LOSS & GRADIENTS)")
    print("=" * 65)
    print(f"Env Step Reward     : {reward:.4f} | Done: {done}")
    print(f"Predicted Q(s, a={action_idx}) : {q_pred.item():.6f} (before update)")
    print(f"Target Q-value (y)  : {q_target.item():.6f}")
    print(f"TD Huber Loss       : {loss.item():.6f}")

    print("\nLayer Gradient Norms (L2):")
    for name, param in policy_dqn.named_parameters():
        if param.grad is not None:
            print(f"  {name:<22} | shape={str(list(param.shape)):<12} | grad_norm={param.grad.norm().item():.6f}")

    # Apply optimizer step and verify parameter update changes Q(s, a)
    optimizer.step()
    with torch.no_grad():
        q_pred_after = policy_dqn(state_tensor)[0, action_idx].item()
    print(f"\nPredicted Q(s, a={action_idx}) : {q_pred_after:.6f} (after 1 optimizer step)")
    print("=" * 65)

    env.close()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = PPOConfig.discount_factor,
    gae_lambda: float = PPOConfig.gae_lambda,
):
    """Generalized Advantage Estimation over a rollout of length T.

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * not_done * next_v - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def run_ppo_network():
    """Assignment 2 Demo: Network Instantiation, Forward Pass & Backward Pass (Headless)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factored = getattr(PPOConfig, "factored_action", False)
    env = gym.make(GymConfig.gym_id, render_mode=None, bot_type=PPOConfig.bot_type, discrete_action=not factored)

    num_states = env.observation_space.shape[0]
    action_dim = list(env.action_space.nvec) if factored else env.action_space.n
    hidden_dim = PPOConfig.hidden_size

    # 1. Create Actor-Critic Network
    agent = ActorCritic(num_states, action_dim, hidden_dim).to(device)
    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(agent.parameters(), lr=PPOConfig.learning_rate, eps=1e-5)
    gamma = PPOConfig.discount_factor
    gae_lambda = PPOConfig.gae_lambda
    clip_coef = PPOConfig.clip_coef  # PPO surrogate clipping range epsilon
    vf_coef = PPOConfig.value_loss_coef  # Value loss weight
    ent_coef = PPOConfig.entropy_coef  # Entropy bonus weight
    rollout_steps = PPOConfig.demo_rollout_steps

    print("=" * 65)
    print("1. NETWORK ARCHITECTURE (PPO ACTOR-CRITIC)")
    print("=" * 65)
    print(
        f"Device: {device} | State Dim: {num_states} | Action Dim: {action_dim} | "
        f"Hidden Dim: {hidden_dim} | Trainable Params: {total_params:,}"
    )
    print(agent)

    # 2. Forward Pass with Environment Observation
    obs, info = env.reset()
    state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 52)

    logits, value = agent(state_tensor)
    if agent.is_factored:
        action_parts = [l.squeeze(0).argmax().item() for l in logits]
        dir_idx, jump_bit, boost_bit = action_parts[0], action_parts[1], action_parts[2]
        dir_probs = F.softmax(logits[0], dim=-1)
        print("\n" + "=" * 65)
        print("2. FORWARD PASS (INPUT -> OUTPUT)")
        print("=" * 65)
        print(f"Input State Shape : {state_tensor.shape}")
        print(f"Input State (first 8 features): {state_tensor[0, :8].detach().cpu().numpy().round(4)}")
        print(f"Output Logits Shapes: {[list(l.shape) for l in logits]}")
        print(f"Output Direction pi Sample (first 5 actions): {dir_probs[0, :5].detach().cpu().numpy().round(4)}")
        print(f"Output V(s)         : {value.item():.6f}")
        print(
            f"Greedy Action     : {action_parts} -> dir={GymConfig.directions[dir_idx]}, "
            f"jump={bool(jump_bit)}, boost={bool(boost_bit)}"
        )
    else:
        probs = F.softmax(logits, dim=-1)
        action_idx = probs.argmax(dim=-1).item()
        dir_idx, jump_bit, boost_bit = action_idx // 4, (action_idx % 4) // 2, action_idx % 2
        print("\n" + "=" * 65)
        print("2. FORWARD PASS (INPUT -> OUTPUT)")
        print("=" * 65)
        print(f"Input State Shape : {state_tensor.shape}")
        print(f"Input State (first 8 features): {state_tensor[0, :8].detach().cpu().numpy().round(4)}")
        print(f"Output Logits Shape : {logits.shape}")
        print(f"Output pi(.|s) Sample (first 6 actions): {probs[0, :6].detach().cpu().numpy().round(4)}")
        print(f"Output V(s)         : {value.item():.6f}")
        print(
            f"Greedy Action     : {action_idx} -> dir={GymConfig.directions[dir_idx]}, "
            f"jump={bool(jump_bit)}, boost={bool(boost_bit)}"
        )

    # Collect a short on-policy rollout of transitions (s, a, log pi_old(a|s), r, done, V(s))
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(rollout_steps):
            action, log_prob, _, v = agent.get_action_and_value(state_tensor)
            step_act = action.squeeze(0).cpu().numpy().tolist() if agent.is_factored else action.item()
            next_state, reward, terminated, truncated, _ = env.step(step_act)
            done = terminated or truncated

            states.append(state_tensor.squeeze(0))
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            values.append(v.squeeze(0))
            rewards.append(float(reward))
            dones.append(float(done))

            if done:
                next_state, _ = env.reset()
            state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        _, next_value = agent(state_tensor)

    states = torch.stack(states)  # Shape: (T, 52)
    actions = torch.stack(actions)  # Shape: (T,)
    old_log_probs = torch.stack(log_probs)  # Shape: (T,)
    values = torch.stack(values)  # Shape: (T,)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)  # Shape: (T,)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)  # Shape: (T,)

    # 3. Backward Pass (GAE, Clipped Surrogate Loss & Gradient Backpropagation)
    advantages, returns = compute_gae(rewards, values, dones, next_value.squeeze(0), gamma, gae_lambda)
    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Re-evaluate rollout actions under the current policy
    _, new_log_probs, entropy, new_values = agent.get_action_and_value(states, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped surrogate objective: L^CLIP = E[min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t)]
    pg_loss1 = -norm_advantages * ratio
    pg_loss2 = -norm_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    v_loss = F.mse_loss(new_values, returns)
    entropy_loss = entropy.mean()

    # Total loss: policy loss + weighted value loss - weighted entropy bonus
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
    optimizer.zero_grad()
    loss.backward()

    print("\n" + "=" * 65)
    print("3. BACKWARD PASS (LOSS & GRADIENTS)")
    print("=" * 65)
    print(
        f"Rollout Steps       : {rollout_steps} | Total Reward: {rewards.sum().item():.4f} | Episodes Done: {int(dones.sum().item())}"
    )
    print(f"Mean Advantage (raw): {advantages.mean().item():.6f}")
    print(f"Mean Return         : {returns.mean().item():.6f}")
    print(f"Mean Ratio          : {ratio.mean().item():.6f} (should be ~1.0 before update)")
    print(f"Policy (Clip) Loss  : {pg_loss.item():.6f}")
    print(f"Value MSE Loss      : {v_loss.item():.6f}")
    print(f"Entropy             : {entropy_loss.item():.6f}")
    print(f"Total Loss          : {loss.item():.6f}")

    print("\nLayer Gradient Norms (L2):")
    for name, param in agent.named_parameters():
        if param.grad is not None:
            print(f"  {name:<22} | shape={str(list(param.shape)):<12} | grad_norm={param.grad.norm().item():.6f}")

    # Apply optimizer step and verify parameter update changes the policy ratio and value estimates
    nn.utils.clip_grad_norm_(agent.parameters(), max_norm=PPOConfig.max_grad_norm)
    optimizer.step()
    with torch.no_grad():
        _, log_probs_after, _, values_after = agent.get_action_and_value(states, actions)
        log_ratio_after = log_probs_after - old_log_probs
        ratio_after = torch.exp(log_ratio_after)
        approx_kl = ((ratio_after - 1.0) - log_ratio_after).mean()  # non-negative KL(old || new) estimate
    print(f"\nMean Ratio          : {ratio_after.mean().item():.6f} (after 1 optimizer step)")
    print(f"Approx KL           : {approx_kl.item():.6f}")
    print(f"Value MSE Loss      : {F.mse_loss(values_after, returns).item():.6f} (after 1 optimizer step)")
    print("=" * 65)

    env.close()
