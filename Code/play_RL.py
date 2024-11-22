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
    policy.load_state_dict(torch.load("/its/home/drs25/Documents/GitHub/Quadruped/my_quadruped_model_2"))

    
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