path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import Quadruped
import numpy as np
from stable_baselines3.common.env_checker import check_env
import gym
from gym import spaces
from stable_baselines3 import PPO
import time
from environment import *
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

env=GYM(p)
#check_env(env)
# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_quadruped_tensorboard/",device="cuda")

# Train the model
model.learn(total_timesteps=100000)  # Adjust the number of timesteps as needed

# Save the model
model.save("/its/home/drs25/Documents/GitHub/Quadruped/ppo_quadruped_model.zip")
p.disconnect()
p.connect(p.GUI) #DIRECT
p.setAdditionalSearchPath(pybullet_data.getDataPath())
env=GYM(p)
# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

print("*********************************\n\n\nTIME TAKEN",(time.time()-t1) /(60*60))