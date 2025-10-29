import random
from BaseAgent import BaseAgent
from DrawPicture import save_training_data


class RandomAgent(BaseAgent):
    def __init__(self, env_name="CartPole-v1", episodes=500):
        super().__init__(env_name, episodes)

    def select_action(self, state):
        return random.choice([0, 1])
    
    # update and train same as BaseAgent


if __name__ == "__main__":
    agent = RandomAgent(episodes=5000)
    rewards = agent.train()

    # Collect all self attributes (state of the agent)
    agent_info = {
        k: v for k, v in agent.__dict__.items()
        if isinstance(k, str) and isinstance(v, (str, int, float, bool)) or v is None
    }

    # Save both agent info and rewards
    save_training_data(
        filename="record/RandomAgent_",
        rewards=rewards,
        **agent_info
    )
