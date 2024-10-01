path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3.common.env_checker import check_env
import gym
from gym import spaces
from stable_baselines3 import PPO
import time
from environment import *
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Reconnect to PyBullet in GUI mode
p.connect(p.GUI)  
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Initialize the environment
env = GYM()

# Load the previously saved model
model = PPO.load("/its/home/drs25/Documents/GitHub/Quadruped/ppo_quadruped_model.zip")

# Reset the environment
obs = env.reset()

# Run the trained model and visualize it
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    
    # Reset the environment if done
    if done:
        obs = env.reset()

# Close the PyBullet simulation
p.disconnect()