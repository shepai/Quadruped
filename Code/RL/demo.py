if __name__=="__main__":
    import sys
    sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')#
    sys.path.insert(1, '/its/home/drs25/Quadruped/Code')#
    sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
datapath="/its/home/drs25/Quadruped/"
#datapath="C:/Users/dexte/Documents/GitHub/Quadruped/"
from environment import *
from CPG import *
import time
from copy import deepcopy
import pickle
import os
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np

def F3(done,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        positions = np.array(history['positions'])
        Y, X, Z = positions[:, 0], positions[:, 1], positions[:, 2]
        orientations=history['orientations']
        magnitude=np.sqrt(np.sum(np.square(orientations),axis=1))
        forward_movement = np.abs(X[-1] - X[0])
        fitness=np.abs(np.sum(np.diff(Y))) - np.sum(np.abs(magnitude))/10
    if type(fitness)!=type(0): 
        try:
            if type(fitness)==type([]): fitness=float(fitness[0])
            else:fitness=float(fitness)
        except:
            print("shit",fitness,np.array(history['motors']).shape,np.array(history['positions']).shape,np.array(history['orientations']).shape)
            fitness=-1000
    if done: fitness=-1000
    return fitness

class FrictionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)

class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=FrictionFeatureExtractor,
            features_extractor_kwargs={},
        )


env = GYM()
env.setFitness(F3)
env.setAgent(sinBot())

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ppo import PPO
check_env(env)

model = PPO(CustomMLPPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs,_ = env.reset()
for step in range(200):  # Run for 200 steps
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if done:
        break


ar=np.array(env.history['positions'])
np.save(datapath+"models/RL/testpath",ar)

np.save(datapath+"models/RL/history",np.array(env.fitness_over_time))

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') 
plt.plot(ar[:,0],ar[:,1])
plt.show()