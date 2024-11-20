path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
from CPG import NN
import pybullet as p
import pybullet_data
import numpy as np
from environment import GYM  # Your custom GYM environment
import time
import os
import torch
def loss_function(predictions, rewards):
    # Example: Mean squared error loss
    return torch.mean((predictions - rewards) ** 2)

import torch
import torch.optim as optim
import torch.nn.functional as F

def train_policy(env, policy, episodes=1000, max_steps=1000, learning_rate=1e-2):
    # Define optimizer
    optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
    rewards_history = []

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Forward pass through the policy
            action = policy(obs_tensor)
            motors = policy.forward_positions(action, torch.tensor(env.quad.motors))

            # Take a step in the environment
            next_obs, reward, done, _ = env.step(motors)
            total_reward += reward

            # Convert reward to tensor for loss computation
            reward_tensor = torch.tensor(reward, dtype=torch.float32)

            # Compute loss
            loss = F.mse_loss(action, reward_tensor)

            # Zero gradients, backpropagate, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                break
            obs = next_obs

        rewards_history.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    return rewards_history


# Main script
if __name__ == "__main__":
    # Initialize PyBullet
    #p.disconnect()  # Ensure clean start
    #p.connect(p.GUI)  # Use GUI for visualization
    #p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Initialize environment and custom policy
    env = GYM(1,delay=0)
    input_size = env.observation_space.shape[0]  # Assuming environment provides observation_space
    hidden_size = 32  # Arbitrary choice; adjust as needed
    policy = NN(input_size, hidden_size,env=env)

    # Train the policy
    start_time = time.time()
    rewards_history=train_policy(env, policy, episodes=10000, max_steps=1000)
    print(f"Training complete. Time taken: {(time.time() - start_time) / 3600:.2f} hours")

    # Test the trained policy
    obs = env.reset()
    for _ in range(1000):
        action = policy.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    torch.save(policy.state_dict(), "/its/home/drs25/Documents/GitHub/Quadruped/my_quadruped_model")
    import matplotlib.pyplot as plt#
    import matplotlib
    matplotlib.use('TkAgg')
    # Plot the reward progression
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Progression During Training')
    plt.show()