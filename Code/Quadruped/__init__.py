import math as maths
import numpy as np

class Quadruped:
    def __init__(self,p,robot_id,floor=None):
        self.robot_id =robot_id  #connect to simulation
        self.p=p
        self.motors=[0 for i in range(3*4)] #positions in degrees
        self.neutral=[-30, 0, 40, -30, 50, -10, 0, 10, 20, 30, -30, 50] #positions in degrees
        self.start=self.getPos()
        self.start_orientation=self.getOrientation()[0:3]
        self.floor=floor
    def reset(self):
        for joint_index in range(12): 
            self.p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=maths.radians(self.neutral[joint_index])
            )
    def setPositions(self,positions): #positions in degrees
        for i in range(len(positions)):
           self.motors[i]=positions[i]
           self.p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=maths.radians(self.motors[i])
            ) 
    def getPos(self):
        position, orientation = self.p.getBasePositionAndOrientation(self.robot_id)
        # Extract the x, y, z position
        x, y, z = position
        return x,y,z
    def getOrientation(self):
        position, orientation = self.p.getBasePositionAndOrientation(self.robot_id)
        # Extract the x, y, z position
        return  orientation[0:3]
    def get_self_collision_count(self):
        # Get all contact points where the robot is in contact with itself
        contact_points = self.p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        return len(contact_points)
    def getContact(self):
        if type(self.floor)==type(None):
            raise TypeError("No floor ")
        contact_points = self.p.getContactPoints(bodyA=self.robot_id, bodyB=self.floor)
        return contact_points
    def getFeet(self):
        feet={8:0,11:0,5:0,2:0}
        points=self.getContact()
        for p in points:
            if feet.get(p[3],"wrong")!="wrong": # only gather feet
                feet[p[3]]=p[9] # gather force
        return np.array([feet[2],feet[5],feet[8],feet[11]])
    def hasFallen(self):
        points=self.getContact()
        for p in points:
            if p[3] in [0]:
                return True
        return False