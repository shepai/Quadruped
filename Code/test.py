path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
import Quadruped

# Initialize the PyBullet physics engine
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF('plane.urdf')
robot_id = p.loadURDF(path+"Quadruped.urdf")
p.setGravity(0, 0, -9.81)

quad=Quadruped.Quadruped(p,robot_id)
quad.neutral=[80,1,1,1,0,0,0,0,0,0,0,0]
quad.reset()

while 1:
    inp=input(">").split()
    i=inp[0]
    deg=inp[1]
    quad.neutral[int(i)]=int(deg)
    quad.reset()
    for i in range(100):
        p.stepSimulation()
        #print("")
        # Optional sleep for real-time simulation
        time.sleep(1./240.)