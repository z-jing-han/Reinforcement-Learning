import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from BaseAgent import BaseAgent
from DrawPicture import save_training_data

# ----- Monte Carlo Agent -----
class MCAgent(BaseAgent):
    def __init__(self, env_name = "CartPole-v1", episodes = 500, gamma=0.99, epsilon=0.1):
        super().__init__(env_name, episodes)
        self.n_actions = gym.make(self.env_name).action_space.n
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))  # Q(s,a)
        self.returns = defaultdict(list)                        # store each (s,a) returns
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        # ε-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.Q[state])

    def update(self, episode):
        """
        episode = [(state, action, reward), ...]
        Monte Carlo update after each episode end
        """
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited:
                self.returns[(state, action)].append(G)
                # update Q as average of returns
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                visited.add((state, action))

    # train same as BaseAgent

if __name__ == "__main__":
    agent = MCAgent(episodes=1000)
    rewards = agent.train()

    agent_info = {
        k: v for k, v in agent.__dict__.items()
        if isinstance(k, str) and isinstance(v, (str, int, float, bool)) or v is None
    }

    save_training_data(
        filename = "record/MCAgent_", 
        rewards = rewards,
        **agent_info
    )
