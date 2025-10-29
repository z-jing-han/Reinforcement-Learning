import gymnasium as gym
import numpy as np

class BaseAgent:
    def __init__(self, env_name, episodes):
        self.env_name = env_name
        self.episodes = episodes

    def select_action(self, state):
        raise NotImplementedError

    def update(self, *args):
        pass
    
    def train(self):
        """
        General Train Process
        """
        env = gym.make(self.env_name)
        rewards = []

        for ep in range(self.episodes):
            state, _ = env.reset()
            state = tuple(np.round(state, 2))
            episode = []
            total_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = tuple(np.round(next_state, 2))
                episode.append((state, action, reward))
                state = next_state
                total_reward += reward
                if done or truncated:
                    break

            self.update(episode)
            rewards.append(total_reward)
            print(f"Episode {ep+1}: Reward = {total_reward}")

        env.close()
        return rewards
