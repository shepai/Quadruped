import math as maths

class Quadruped:
    def __init__(self,p,robot_id):
        self.robot_id =robot_id  #connect to simulation
        self.p=p
        self.motors=[0 for i in range(3*4)] #positions in degrees
        self.neutral=[0 for i in range(3*4)] #positions in degrees
        self.start=self.getPos()
    def reset(self):
        for joint_index in range(12): 
            joint_info = self.p.getJointInfo(self.robot_id, joint_index)
            self.p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=maths.radians(self.neutral[joint_index])
            )
    def setPositions(self,positions): #positions in degrees
        for i in range(len(positions)):
           self.motors[i]=positions[i]
           joint_info = self.p.getJointInfo(self.robot_id, i)
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