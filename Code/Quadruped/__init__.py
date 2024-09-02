import math as maths

class Quadruped:
    def __init__(self,p,robot_id):
        self.robot_id =robot_id  #connect to simulation
        self.p=p
        self.motors=[0 for i in range(3*4)]
        self.neutral=[0 for i in range(3*4)]
        
    def reset(self):
        for joint_index in range(12):
            joint_info = self.p.getJointInfo(self.robot_id, joint_index)
            self.p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=maths.radians(self.neutral[joint_index])
            )
