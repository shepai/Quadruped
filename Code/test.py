path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
import Quadruped
import os
import numpy as np

clear = lambda: os.system('clear')
# Initialize the PyBullet physics engine
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF('plane.urdf')
flags = p.URDF_USE_SELF_COLLISION
robot_id = p.loadURDF(path+"Quadruped_prestip.urdf",flags=flags)
p.setGravity(0, 0, -9.81)

quad=Quadruped.Quadruped(p,robot_id)
quad.neutral=[-10,0,30,0,0,0,0,0,0,0,0,0]
quad.reset()

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


while 1:
    for i in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    clear()
    print(quad.getOrientation(),quad.start_orientation,euclidean_distance(quad.start_orientation,quad.getOrientation()))
