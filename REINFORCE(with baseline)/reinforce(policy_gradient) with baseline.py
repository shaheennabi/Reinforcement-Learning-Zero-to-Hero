## in this file we will implement the Reinforce with baseline...
import gymnasium as gym
import torch
import  torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.distributions import  Categorical





## PolicyNet (actor) ## this will give us the probability of the actions etc...
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, 128)
        self.output_layer = nn.Linear(128, action_dim)



    def forward(self, state): 
        x = F.relu(self.input_layer(state))
        probabilities = F.softmax(self.output_layer(x), dim=-1)
        return probabilities
    



## this will give us the scalar value or state-value
class ValueNet(nn.Module): 
    def __init__(self, state_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, 128)
        self.output_layer = nn.Linear(128, 1)

    

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        return self.output_layer(x)





def training_reinforce_with_baseline():
    env = gym.make("CartPole-v1")  ## from openai-gymnasium

    state_dim = env.observation_space.shape[0]  ## this is the shape of the state_dim ---4
    action_dim = env.action_space.n  ## this is the shape, ---2

    policy = PolicyNet(state_dim, action_dim)   
    value_net = ValueNet(state_dim)

    policy_optim = optim.Adam(policy.parameters(), lr=3e-4)
    value_optim = optim.Adam(value_net.parameters(), lr=1e-4)

    num_episodes = 100
    gamma = 0.99

    for episode in range(num_episodes):
        state, _ = env.reset()  ## this is the initial_state : start
        done = False

        log_probs = []
        values = []
        rewards = []

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32)

            probs = policy(state_t)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value_net(state_t))

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)
            state = next_state

        # returns
        actual_return = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            actual_return.insert(0, G)

        actual_return = torch.tensor(actual_return, dtype=torch.float32)
        values = torch.cat(values).squeeze()

        # advantage
        advantage = actual_return - values.detach()

        # policy update
        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()

        # value update
        value_loss = F.mse_loss(values, actual_return)
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()

        print(f"Episode {episode}, Return: {sum(rewards)}")

    env.close()


if __name__ == "__main__":
    training_reinforce_with_baseline()