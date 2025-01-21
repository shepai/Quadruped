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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def loss_function(predictions, rewards):
    # Example: Mean squared error loss
    return torch.mean((predictions - rewards) ** 2)

import torch
import torch.optim as optim
import torch.nn.functional as F
def train_policy(env, policy, episodes=1000, max_steps=1000, learning_rate=1e-2, gamma=0.99, noise_scale=0.1):
    import torch
    from torch import optim
    import matplotlib.pyplot as plt

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    rewards_history = []

    for episode in range(episodes):
        obs = env.reset()
        episode_rewards = []
        log_probs = []
        ar=[]
        total_reward = 0
        motor_positions_history = []

        plt.cla()
        plt.title("Motor positions of robot")

        for step in range(max_steps):
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Forward pass through the policy to get sine wave parameters
            sine_params = policy(obs_tensor)  # Outputs the sine wave parameters deterministically

            # Add exploration noise to the parameters
            noise = torch.normal(0, noise_scale, size=sine_params.shape)
            noisy_params = sine_params + noise

            # Generate motor positions using the (noisy) parameters
            motors = policy.forward_positions(noisy_params, torch.tensor(env.quad.motors))
            motor_positions_history.append(motors.cpu().detach().numpy())

            # Take a step in the environment
            next_obs, reward, done, _ = env.step(motors.numpy())
            episode_rewards.append(reward)

            total_reward += reward
            obs = next_obs
            ar.append(motors.cpu().detach().numpy())
            if done:
                break
            obs = next_obs
            if step%100:
                torch.save(policy.state_dict(), "/its/home/drs25/Documents/GitHub/Quadruped/my_quadruped_model")
            if len(ar)>250: 
                plt.cla()
                ar.pop(0)
            
            plt.plot(ar)
            plt.pause(0.01)

        # Compute discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(episode_rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Compute loss
        loss = 0
        for param, reward in zip(noisy_params, discounted_rewards):
            # Penalize divergence of noisy actions from original deterministic policy output
            loss += ((param - sine_params) ** 2).sum() * reward  # Weighted by reward

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    rewards_history=train_policy(env, policy, episodes=550, max_steps=800)
    print(f"Training complete. Time taken: {(time.time() - start_time) / 3600:.2f} hours")

    # Test the trained policy
    obs = env.reset()
    for _ in range(1000):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action = policy(obs_tensor)
        motors = policy.forward_positions(action, torch.tensor(env.quad.motors))
        obs, rewards, done, info = env.step(motors)
        env.render()
        if done:
            obs = env.reset()
    
    import matplotlib.pyplot as plt#
    import matplotlib
    matplotlib.use('TkAgg')
    # Plot the reward progression
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Progression During Training')
    plt.show()