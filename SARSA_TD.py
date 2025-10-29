import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from MCAgent import MCAgent
from DrawPicture import save_training_data

class SARSAgent(MCAgent):
    def __init__(self, env_name = "CartPole-v1", episodes = 500, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env_name, episodes, gamma, epsilon)
        self.alpha = alpha      # learning rate

    def update(self, state, action, reward, next_state, next_action, done):
        target = reward + (0 if done else self.gamma * self.Q[next_state][next_action])
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def discretize_state(self, state):
        """
        discretize state
        """
        bins = np.array([
            np.linspace(-2.4, 2.4, 10),    # cart position
            np.linspace(-3.0, 3.0, 10),    # cart velocity
            np.linspace(-0.5, 0.5, 10),    # pole angle
            np.linspace(-2.0, 2.0, 10)     # pole velocity
        ])
        return tuple(np.digitize(s, b) for s, b in zip(state, bins))

    def train(self):
        env = gym.make(self.env_name)
        rewards = []

        for ep in range(self.episodes):
            state, _ = env.reset()
            state = self.discretize_state(state)
            action = self.select_action(state)
            total_reward = 0

            while True:
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = self.discretize_state(next_state)
                next_action = self.select_action(next_state)
                self.update(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action
                total_reward += reward
                if done or truncated:
                    break

            rewards.append(total_reward)
            print(f"Episode {ep+1}, Reward: {total_reward}")

        env.close()
        return rewards

if __name__ == "__main__":
    agent = SARSAgent(episodes=1000)
    rewards = agent.train()
    
    # Collect all self attributes (state of the agent)
    agent_info = {
        k: v for k, v in agent.__dict__.items()
        if isinstance(k, str) and isinstance(v, (str, int, float, bool)) or v is None
    }

    # Save both agent info and rewards
    save_training_data(
        filename="record/SARSA_TD_",
        rewards=rewards,
        **agent_info
    )
