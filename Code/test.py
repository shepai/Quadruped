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
print(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
plane_id = p.loadURDF("plane.urdf")#/its/home/drs25/Documents/GitHub/Terrain_generator_3D/assets/test.urdf
flags = p.URDF_USE_SELF_COLLISION
robot_id = p.loadURDF(path+"Quadruped_prestip.urdf", [0, 0, 0.3], flags=flags)
p.setGravity(0, 0, -9.81)

quad=Quadruped.Quadruped(p,robot_id,plane_id)
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
    contact=quad.getContact()
    for i in range(len(contact)):
        print("\t-- robot link",contact[i][3],"with force",contact[i][9],"Has fallen",quad.hasFallen())

    print("Feet sensors",quad.getFeet())