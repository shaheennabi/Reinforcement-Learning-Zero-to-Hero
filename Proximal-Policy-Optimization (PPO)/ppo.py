"""
Minimal, correct PPO implementation for continuous control (Pendulum-v1)

Key design principles:
- On-policy data collection
- Generalized Advantage Estimation (GAE)
- PPO clipped surrogate objective
- Shared backbone Actor-Critic
- Clear separation: rollout -> advantage -> update
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ============================================================
# Actor-Critic Network
# ============================================================
"""
We use a shared backbone for feature extraction,
then separate heads for:
- Policy (mean + std of Gaussian)
- Value function V(s)

This mirrors most PPO / RLHF implementations.
"""

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: outputs mean of Gaussian policy
        self.mean = nn.Linear(hidden_dim, act_dim)

        # Log standard deviation (state-independent, common PPO choice)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head: outputs scalar value V(s)
        self.v_head = nn.Linear(hidden_dim, 1)

    def policy(self, obs):
        """
        Returns a torch.distributions.Normal object.
        Sampling, log_prob, entropy all come from this.
        """
        features = self.shared(obs)
        mean = self.mean(features)
        std = torch.exp(self.log_std)  # ensure positivity
        return Normal(mean, std)

    def value(self, obs):
        """
        Returns V(s) as a scalar per observation.
        """
        features = self.shared(obs)
        return self.v_head(features).squeeze(-1)


# ============================================================
# Rollout Buffer (On-policy storage)
# ============================================================
"""
Stores exactly what PPO needs from the OLD policy:
- observations
- actions
- rewards
- done flags
- log_probs under old policy
- value estimates under old policy

No gradients are stored here.
"""

class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def store(self, obs, act, rew, done, logp, val):
        """
        Store one timestep of data.
        """
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.logp[self.ptr] = logp
        self.val[self.ptr] = val
        self.ptr += 1

    def get(self):
        """
        Convert stored numpy arrays into torch tensors.
        """
        return dict(
            obs=torch.as_tensor(self.obs),
            act=torch.as_tensor(self.act),
            rew=torch.as_tensor(self.rew),
            done=torch.as_tensor(self.done),
            logp=torch.as_tensor(self.logp),
            val=torch.as_tensor(self.val),
        )


# ============================================================
# Generalized Advantage Estimation (GAE)
# ============================================================
"""
GAE trades bias vs variance using lambda.

Core idea:
- Compute TD residuals (delta)
- Exponentially accumulate them backward in time

adv_t = delta_t + gamma * lambda * adv_{t+1}
"""

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_adv = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    # Target for value function
    returns = advantages + values

    # Advantage normalization (important for PPO stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ============================================================
# PPO Update Step
# ============================================================
"""
This is where learning actually happens.

We:
- Recompute log_probs under current policy
- Form ratio = pi_new / pi_old
- Apply PPO clipping
- Add value loss and entropy bonus
"""

def ppo_update(
    model,
    optimizer,
    data,
    advantages,
    returns,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    epochs=10,
    batch_size=64,
):

    obs = data["obs"]
    act = data["act"]
    logp_old = data["logp"]

    N = len(obs)

    kl_list = []
    clip_frac_list = []

    for _ in range(epochs):
        idx = torch.randperm(N)

        for start in range(0, N, batch_size):
            batch = idx[start:start + batch_size]

            # New policy distribution
            dist = model.policy(obs[batch])

            # Log-prob under new policy
            logp = dist.log_prob(act[batch]).sum(-1)

            # Importance sampling ratio
            ratio = torch.exp(logp - logp_old[batch])

            # PPO clipped objective
            surr1 = ratio * advantages[batch]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[batch]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value = model.value(obs[batch])
            value_loss = ((value - returns[batch]) ** 2).mean()

            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().sum(-1).mean()

            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Diagnostics (no gradients)
            approx_kl = (logp_old[batch] - logp).mean().item()
            clip_frac = (torch.abs(ratio - 1.0) > clip_eps).float().mean().item()

            kl_list.append(approx_kl)
            clip_frac_list.append(clip_frac)

    return {
        "kl": np.mean(kl_list),
        "clip_frac": np.mean(clip_frac_list),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }


# ============================================================
# Training Loop
# ============================================================
"""
Outer loop:
- Collect rollout with OLD policy
- Compute advantages
- Run PPO update
"""

def train(
    total_iters=50,
    steps_per_iter=2048,
):

    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for itr in range(total_iters):
        buffer = RolloutBuffer(steps_per_iter, obs_dim, act_dim)
        obs, _ = env.reset()

        ep_return = 0
        ep_returns = []

        # -------- Rollout collection --------
        for t in range(steps_per_iter):
            obs_t = torch.tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                dist = model.policy(obs_t)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = model.value(obs_t)

            next_obs, reward, terminated, truncated, _ = env.step(act.numpy())
            done = terminated or truncated
            ep_return += reward

            buffer.store(
                obs=obs,
                act=act.numpy(),
                rew=reward,
                done=done,
                logp=logp.item(),
                val=val.item(),
            )

            obs = next_obs if not done else env.reset()[0]

            if done:
                ep_returns.append(ep_return)
                ep_return = 0

        # -------- PPO update --------
        data = buffer.get()
        adv, returns = compute_gae(data["rew"], data["val"], data["done"])
        metrics = ppo_update(model, optimizer, data, adv, returns)

        if itr % 10 == 0:
            print(
                f"Iter {itr:4d} | "
                f"Return {np.mean(ep_returns):8.1f} | "
                f"KL {metrics['kl']:.4f} | "
                f"Clip {metrics['clip_frac']:.2f} | "
                f"VLoss {metrics['value_loss']:.3f} | "
                f"Entropy {metrics['entropy']:.3f}"
            )


if __name__ == "__main__":
    train()




### this code is mostly written by chatgpt, or improved or finetuned by chat....(flow for this algorithm is big)---so break it into pieces...etc
## thanks 