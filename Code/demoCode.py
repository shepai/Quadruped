path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
#path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
# Initialize the PyBullet physics engine
p.connect(p.GUI)

# Optionally set the path to PyBullet's data directory
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF('plane.urdf')
# Load the URDF file. Replace 'robot.urdf' with your URDF file's path
flags = p.URDF_USE_SELF_COLLISION
robot_id = p.loadURDF(path+"Quadruped.urdf",flags=flags)
p.setGravity(0, 0, -9.81)
# Get the number of joints in the loaded URDF

num_joints = p.getNumJoints(robot_id)

# Iterate through each joint and print some information
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    print(f"Joint {joint_index}: {joint_info[1].decode('utf-8')}")

# Move all joints through their ranges of motion
while True:
    for joint_index in range(num_joints):
        # Get the joint's range of motion
        joint_info = p.getJointInfo(robot_id, joint_index)
        lower_limit = 0
        upper_limit = 50

        # Set the joint position to a specific value within the range
        target_position = (lower_limit + upper_limit) / 2.0  # Midpoint for simplicity
        
        # Use a sinusoidal motion for continuous movement
        target_position = lower_limit + (upper_limit - lower_limit) * (0.01 * (1 + maths.sin(time.time())))
        #print(target_position,end=",")
        #print(upper_limit,lower_limit)
        # Move the joint to the target position
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position
        )
    # Step the simulation
    p.stepSimulation()
    #print("")
    # Optional sleep for real-time simulation
    time.sleep(1./240.)

# Disconnect when done (in a real application you'd have a condition to exit)
p.disconnect()
