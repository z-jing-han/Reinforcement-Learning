import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from SARSA_TD import SARSAgent
from DrawPicture import save_training_data

class QLearningAgent(SARSAgent):
    def __init__(self, env_name = "CartPole-v1", episodes = 500, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(env_name, episodes, alpha, gamma, epsilon)

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning Temperal Different
        The most different part with SARSA: no need of next_action, just take the best action
        """
        best_next_action = np.argmax(self.Q[next_state])
        target = reward + (0 if done else self.gamma * self.Q[next_state][best_next_action])
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def train(self):
        env = gym.make(self.env_name)
        rewards = []

        for ep in range(self.episodes):
            state, _ = env.reset()
            state = self.discretize_state(state)
            total_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = self.discretize_state(next_state)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done or truncated:
                    break

            rewards.append(total_reward)
            print(f"Episode {ep+1}: Reward = {total_reward}")

        env.close()
        return rewards

if __name__ == "__main__":
    agent = QLearningAgent(episodes=1000)
    rewards = agent.train()

    # Collect all self attributes (state of the agent)
    agent_info = {
        k: v for k, v in agent.__dict__.items()
        if isinstance(k, str) and isinstance(v, (str, int, float, bool)) or v is None
    }

    # Save both agent info and rewards
    save_training_data(
        filename="record/QLearning_TD_",
        rewards=rewards,
        **agent_info
    )