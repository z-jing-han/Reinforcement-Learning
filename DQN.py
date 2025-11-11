# dqn_cartpole.py
import random
from collections import deque, namedtuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
import os
from DrawPicture import save_training_data
from BaseAgent import BaseAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Utilities / Replay Buffer
# ----------------------
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Convert to torch tensors
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ----------------------
# Q-network (MLP)
# ----------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------
# Agent (DQN)
# ----------------------
class DQNAgent(BaseAgent):
    def __init__(self, env_name = "CartPole-v1", gamma = 0.99, lr = 1e-3, BUFFER_CAPACITY = 50000, episodes = 5000, MAX_STEPS_PER_EPISODE = 500, BATCH_SIZE = 64):
        super().__init__(env_name, episodes)
        env = gym.make(self.env_name)
        obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.policy_net = QNetwork(obs_dim, self.n_actions).to(DEVICE)  # Q Approximate Function
        self.target_net = QNetwork(obs_dim, self.n_actions).to(DEVICE)  
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.BUFFER_CAPACITY = BUFFER_CAPACITY
        self.replay = ReplayBuffer(self.BUFFER_CAPACITY)
        self.gamma = gamma
        self.MAX_STEPS_PER_EPISODE = MAX_STEPS_PER_EPISODE
        self.BATCH_SIZE = BATCH_SIZE
        self.MODEL_SAVE_PATH = None
        

    def select_action(self, state, eps_threshold):
        # state: np.array
        if random.random() < eps_threshold:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                qvals = self.policy_net(s)
                return int(torch.argmax(qvals, dim=1).item())

    def optimize_model(self):
        if len(self.replay) < self.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.BATCH_SIZE)

        # current Q
        q_values = self.policy_net(states).gather(1, actions)  # (batch,1)

        # target: r + gamma * max_a Q_target(next_state, a) * (1-done)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # optional gradient clipping:
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        self.MODEL_SAVE_PATH = "record/DQN_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pth"
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.MODEL_SAVE_PATH)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ----------------------
    # Main training loop
    # ----------------------
    def train(self, MIN_REPLAY_SIZE=1000, TARGET_UPDATE_FREQ=1000, EPS_START=1.0, EPS_END=0.02, EPS_DECAY=20000):
        env = gym.make("CartPole-v1")
        total_steps = 0
        episode_rewards = []

        # Warm-up buffer with random policy
        print("Warming up replay buffer...")
        state, _ = env.reset()
        for _ in range(MIN_REPLAY_SIZE):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.replay.push(state, action, reward, next_state, done)
            state = next_state if not done else env.reset()[0]

        print("Start training")
        for ep in range(1, self.episodes + 1):
            state, _ = env.reset()
            ep_reward = 0
            for t in range(self.MAX_STEPS_PER_EPISODE):
                # epsilon schedule (exponential decay)
                eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * total_steps / EPS_DECAY)
                action = self.select_action(state, eps_threshold)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.replay.push(state, action, reward, next_state, done)

                loss = self.optimize_model()

                state = next_state
                ep_reward += reward
                total_steps += 1

                # target network update (hard)
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    self.update_target()

                if done:
                    break

            episode_rewards.append(ep_reward)
            if ep % 10 == 0:
                avg_last_10 = np.mean(episode_rewards[-10:])
                print(f"Episode {ep}\tSteps {total_steps}\tEpisodeReward {ep_reward:.1f}\tAvgLast10 {avg_last_10:.2f}\tEps {eps_threshold:.3f}")

            # optional: early stop if solved (CartPole solved ~195 over 100 episodes)
            if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 195.0:
                print(f"Solved in {ep} episodes!")
                self.save()
                break

        env.close()
        # final save
        self.save()
        return episode_rewards

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    start_time = time.time()
    agent = DQNAgent(env_name="CartPole-v1", gamma=0.99, lr=1e-3, BUFFER_CAPACITY=50000, episodes=5000, MAX_STEPS_PER_EPISODE=500)
    rewards = agent.train(MIN_REPLAY_SIZE=1000, EPS_START=1.0, EPS_END=0.02, EPS_DECAY=20000)
    elapsed = time.time() - start_time
    print(f"Training finished, elapsed {elapsed/60:.2f} min")

    # Collect all self attributes (state of the agent)
    agent_info = {
        k: v for k, v in agent.__dict__.items()
        if isinstance(k, str) and isinstance(v, (str, int, float, bool)) or v is None
    }

    # Save both agent info and rewards
    save_training_data(
        filename="record/DQN_",
        rewards=rewards,
        **agent_info
    )
