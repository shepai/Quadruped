path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
from CPG import NN
import pybullet as p
import pybullet_data
import numpy as np
from environment import GYM  # Your custom GYM environment
import time
import os

# Define the custom NN-based policy class
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

# Define a basic training loop for the NN-based policy
def train_policy(env, policy, episodes=1000, max_steps=1000, mutation_rate=0.1):
    best_genotype = policy.nn.genotype.copy()
    best_reward = float('-inf')
    rewards_history = []

    for episode in range(episodes):
        total_reward = 0
        obs = env.reset()
        done = False

        for step in range(max_steps):
            if done:
                break

            # Predict action using NNPolicy
            action = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        # Evaluate and adjust NN genotype
        if total_reward > best_reward:
            best_reward = total_reward
            best_genotype = policy.nn.genotype.copy()
        else:
            # Apply mutation
            policy.nn.mutate()

        # Update the genotype with the best so far
        policy.nn.set_genotype(best_genotype)
        rewards_history.append(total_reward)

        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    return rewards_history

# Main script
if __name__ == "__main__":
    # Initialize PyBullet
    #p.disconnect()  # Ensure clean start
    #p.connect(p.GUI)  # Use GUI for visualization
    #p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Initialize environment and custom policy
    env = GYM(0,delay=0)
    input_size = env.observation_space.shape[0]  # Assuming environment provides observation_space
    hidden_size = 32  # Arbitrary choice; adjust as needed
    policy = NNPolicy(input_size=input_size, hidden_size=hidden_size,env=env)

    # Train the policy
    start_time = time.time()
    train_policy(env, policy, episodes=5000, max_steps=1000)
    print(f"Training complete. Time taken: {(time.time() - start_time) / 3600:.2f} hours")

    # Test the trained policy
    obs = env.reset()
    for _ in range(1000):
        action = policy.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    policy.save("/its/home/drs25/Documents/GitHub/Quadruped/my_quadruped_model")

