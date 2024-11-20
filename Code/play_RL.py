path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from environment import *
from CPG import NN


# Main script
if __name__ == "__main__":
    # Initialize PyBullet

    # Initialize environment and custom policy
    env = GYM(1,delay=1)
    input_size = env.observation_space.shape[0]  # Assuming environment provides observation_space
    hidden_size = 32  # Arbitrary choice; adjust as needed
    policy =  NN(input_size,hidden_size)
    policy.load_state_dict("/its/home/drs25/Documents/GitHub/Quadruped/my_quadruped_model")

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