import math as maths

class Quadruped:
    def __init__(self,p,robot_id):
        self.robot_id =robot_id  #connect to simulation
        self.p=p
        self.motors=[0 for i in range(3*4)] #positions in degrees
        self.neutral=[0 for i in range(3*4)] #positions in degrees
        self.start=self.getPos()
        self.start_orientation=self.getOrientation()
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
        return  orientation
    def get_self_collision_count(self):
        # Get all contact points where the robot is in contact with itself
        contact_points = self.p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        return len(contact_points)