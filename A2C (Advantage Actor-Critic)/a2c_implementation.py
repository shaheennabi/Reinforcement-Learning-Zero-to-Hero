import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



class SharedNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Shared feature extractor
        # Takes state -> 128-dim feature vector
        self.fc = nn.Linear(state_dim, 128)

        # Actor head:
        # Maps features -> action logits (NOT probabilities)
        self.policy_head = nn.Linear(128, action_dim)

        # Critic head:
        # Maps features -> scalar value V(s)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        # Shared representation
        x = F.relu(self.fc(state))

        # Actor output (logits for Categorical)
        logits = self.policy_head(x)

        # Critic output (value estimate)
        value = self.value_head(x)

        return logits, value



# Create environment
env = gym.make("CartPole-v1")

# Environment dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize model and optimizer
model = SharedNet(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Discount factor
gamma = 0.99

# Loss coefficients
entropy_coef = 0.01   # encourages exploration
value_coef = 0.5      # balances critic loss

# n-step rollout length
n_steps = 5

# Number of policy updates
num_updates = 1000




# Reset environment
state, _ = env.reset()



for update in range(num_updates):

    # Storage for rollout data
    values = []      # V(s_t)
    rewards = []     # r_t
    log_probs = []   # log π(a_t | s_t)
    entropies = []   # policy entropy

    # Track whether episode ended inside rollout
    last_done = False




    for _ in range(n_steps):
        # Convert state to tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        # Forward pass through actor-critic
        logits, value = model(state_tensor)

        # Create categorical policy
        dist = Categorical(logits=logits)

        # Sample action from policy
        action = dist.sample()

        # Log-probability of chosen action
        log_prob = dist.log_prob(action)

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        last_done = done

        # Store rollout data
        values.append(value.squeeze(-1))                   # scalar V(s)
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        log_probs.append(log_prob)
        entropies.append(dist.entropy())

        # If episode ended, reset environment
        if done:
            next_state, _ = env.reset()

        # Move to next state
        state = next_state





    # If episode ended, terminal state value = 0
    if last_done:
        next_value = torch.tensor(0.0)
    else:
        # Otherwise, bootstrap from critic
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            _, next_value = model(state_tensor)
            next_value = next_value.squeeze(-1)




    returns = []
    R = next_value   # bootstrap value

    # Compute returns backwards ## the actual returns 
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.stack(returns)
    values = torch.stack(values)




    # Advantage = (n-step return) - V(s)
    # Detach values so actor does NOT update critic
    advantages = returns - values.detach()



    # Actor loss:
    # maximize E[log π(a|s) * A]  -> minimize negative
    actor_loss = -(torch.stack(log_probs) * advantages).mean()

    # Critic loss:
    # minimize squared TD error
    critic_loss = value_coef * (returns - values).pow(2).mean()

    # Entropy bonus:
    # maximize entropy -> minimize negative entropy
    entropy_loss = -entropy_coef * torch.stack(entropies).mean()

    # Total loss
    loss = actor_loss + critic_loss + entropy_loss



    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute gradients
    optimizer.step()        # update parameters


    if update % 100 == 0:
        print(f"Update {update}: loss = {loss.item():.3f}")



## code refined-by chat...