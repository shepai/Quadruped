path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from environment import *
from CPG import NN

class NNPolicy:
    def __init__(self, input_size, hidden_size,env):
        self.nn = NN(inp=input_size, hidden=hidden_size)
        self.env=env

    def predict(self, observation):
        # Forward pass to generate action
        action = self.nn.get_positions(observation,motors=self.env.quad.motors)
        return action
    def save(self, filepath):
        """Save the current genotype to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.nn.genotype)
        print(f"Policy saved to {filepath}")

    def load(self, filepath):
        """Load the genotype from a file."""
        if os.path.exists(filepath):
            genotype = np.load(filepath)
            self.nn.set_genotype(genotype)
            print(f"Policy loaded from {filepath}")
        else:
            print(f"File {filepath} not found!")

# Main script
if __name__ == "__main__":
    # Initialize PyBullet

    # Initialize environment and custom policy
    env = GYM(1,delay=1)
    input_size = env.observation_space.shape[0]  # Assuming environment provides observation_space
    hidden_size = 32  # Arbitrary choice; adjust as needed
    policy = NNPolicy(input_size=input_size, hidden_size=hidden_size,env=env)

    # Optionally load a saved policy
    save_path = "/its/home/drs25/Documents/GitHub/Quadruped/my_quadruped_model"
    try:
        policy.load(save_path)
    except FileNotFoundError:
        print("No saved policy found. Starting training from scratch.")
    
    # Test the trained policy
    obs = env.reset()
    for _ in range(1000):
        action = policy.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()