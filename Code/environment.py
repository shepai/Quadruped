path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/PressTip/urdf/"
path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/PressTip/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
import Quadruped
import numpy as np
import os
import random
from copy import deepcopy
def demo(variable):
    return 0
class environment:
    def __init__(self,show=False):
        self.show=show
        if show: p.connect(p.GUI)
        else: p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        self.robot_id=None
        self.plane_id=None
        self.quad=None
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF('plane.urdf')
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        initial_position = [0, 0, 0.3]  # x=1, y=2, z=0.5
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
        flags = p.URDF_USE_SELF_COLLISION
        self.robot_id = p.loadURDF(path+"Quadruped_prestip.urdf", initial_position, initial_orientation,flags=flags)
        for i in range(100):
            p.stepSimulation()
            p.setTimeStep(1./24.)
        self.quad=Quadruped.Quadruped(p,self.robot_id,self.plane_id)
        self.quad.neutral=[-10,0,30,0,0,0,0,0,0,0,0,0]
        self.quad.reset()
    def runTrial(self,agent,generations,delay=False,fitness=demo):
        self.reset()
        for i in range(generations):
            motor_positions=agent.get_positions(self.quad.motors)
            self.quad.setPositions(motor_positions)
            for k in range(10): #update simulation
                p.stepSimulation()
                if delay: time.sleep(1./240.)
                else: p.setTimeStep(1./240.)
                basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id) # Get model position
                p.resetDebugVisualizerCamera( cameraDistance=0.3, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos) # fix camera onto model
                if self.quad.hasFallen():
                    break
            if self.quad.hasFallen():
                break
        return fitness(self.quad)
