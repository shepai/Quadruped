path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"

import os
import numpy as np
from environment import environment
from agent import agent
clear = lambda: os.system('clear')
# Initialize the PyBullet physics engine
env=environment(True)
a=agent()
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


while 1:
    env.runTrial(a,500)
    clear()
    print(env.quad.getOrientation(),env.quad.start_orientation,euclidean_distance(env.quad.start_orientation,env.quad.getOrientation()))
    contact=env.quad.getContact()
    for i in range(len(contact)):
        print("\t-- robot link",contact[i][3],"with force",contact[i][9],"Has fallen",env.quad.hasFallen())

    print("Feet sensors",env.quad.getFeet())