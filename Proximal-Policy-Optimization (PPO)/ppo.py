import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym


# =========================
# Actor–Critic Network
# =========================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic
        self.value_head = nn.Linear(hidden_dim, 1)

    def policy(self, obs):
        h = self.shared(obs)
        mean = self.mean(h)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def value(self, obs):
        h = self.shared(obs)
        return self.value_head(h).squeeze(-1)


# =========================
# Rollout Buffer
# =========================

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
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.logp[self.ptr] = logp
        self.val[self.ptr] = val
        self.ptr += 1

    def get(self):
        return dict(
            obs=torch.as_tensor(self.obs),
            act=torch.as_tensor(self.act),
            rew=torch.as_tensor(self.rew),
            done=torch.as_tensor(self.done),
            logp=torch.as_tensor(self.logp),
            val=torch.as_tensor(self.val),
        )


# =========================
# GAE
# =========================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T)
    last_adv = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        adv[t] = last_adv

    returns = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, returns


# =========================
# PPO Update
# =========================

def ppo_update(
    model,
    optimizer,
    data,
    adv,
    returns,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    epochs=10,
    batch_size=64,
):

    obs, act = data["obs"], data["act"]
    logp_old = data["logp"]

    N = len(obs)

    kl_list = []
    clip_frac_list = []

    for _ in range(epochs):
        idx = torch.randperm(N)

        for start in range(0, N, batch_size):
            batch = idx[start:start + batch_size]

            dist = model.policy(obs[batch])
            logp = dist.log_prob(act[batch]).sum(-1)
            ratio = torch.exp(logp - logp_old[batch])

            surr1 = ratio * adv[batch]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[batch]
            policy_loss = -torch.min(surr1, surr2).mean()

            value = model.value(obs[batch])
            value_loss = ((value - returns[batch]) ** 2).mean()

            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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


# =========================
# Training Loop
# =========================

def train_ppo(
    env_name="Pendulum-v1",
    steps_per_iter=2048,
    iters=500,
    device="cpu",
):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for itr in range(iters):
        buffer = RolloutBuffer(steps_per_iter, obs_dim, act_dim)
        obs, _ = env.reset()

        ep_return = 0
        ep_returns = []

        # ---- Rollout ----
        for t in range(steps_per_iter):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                dist = model.policy(obs_t)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = model.value(obs_t)

            next_obs, rew, terminated, truncated, _ = env.step(act.cpu().numpy())
            done = terminated or truncated
            ep_return += rew

            buffer.store(
                obs,
                act.cpu().numpy(),
                rew,
                done,
                logp.cpu().numpy(),
                val.cpu().numpy(),
            )

            obs = next_obs if not done else env.reset()[0]

            if done:
                ep_returns.append(ep_return)
                ep_return = 0

        # ---- GAE ----
        data = buffer.get()
        adv, returns = compute_gae(
            data["rew"], data["val"], data["done"]
        )

        # ---- Update ----
        metrics = ppo_update(
            model,
            optimizer,
            data,
            adv,
            returns,
        )

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
    train_ppo()
