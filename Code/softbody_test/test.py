path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/PressTip/urdf/"
#path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
import os
import numpy as np

clear = lambda: os.system('clear')
# Initialize the PyBullet physics engine
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())

plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])#/its/home/drs25/Documents/GitHub/Terrain_generator_3D/assets/test.urdf
soft_body_id1 = p.loadSoftBody("sphere_smooth.obj", scale=0.1, basePosition=[0, 0, 2], mass=0.1,collisionMargin=0.05, 
                              useNeoHookean=1,  # Enable Neo-Hookean material model
                              NeoHookeanMu=100,  # Higher values for more stiffness
                              NeoHookeanLambda=100,  # Increase for more stiffness
                              NeoHookeanDamping=0.1,  # Damping factor for softer behavior
                              useBendingSprings=1,  # Enable bending springs
                              frictionCoeff=1.0,  # Higher friction
                              repulsionStiffness=500)  # Higher repulsion stiffness)  
# Set the collision group and mask for the soft body
#p.setCollisionFilterGroupMask(soft_body_id1, -1, collisionFilterGroup=1, collisionFilterMask=1)
flags = p.URDF_USE_SELF_COLLISION
rigid_body_id = p.loadURDF(path+"Quadruped_prestip.urdf", basePosition=[0, 0, 3],flags=flags)
p.setTimeStep(1. / 1000.)
p.setPhysicsEngineParameter(numSolverIterations=500)
p.setGravity(0, 0, -9.81)
#p.setCollisionFilterGroupMask(rigid_body_id, -1, collisionFilterGroup=1, collisionFilterMask=1)
p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)



# Get the joint information for the last joint of the rigid body
joint_index = 2  # Choose the joint to which the soft body will be attached (e.g., joint 2)
joint_state = p.getLinkState(rigid_body_id, joint_index)

# Get the position and orientation of the joint link (end of the joint)
joint_pos, joint_orn = joint_state[0], joint_state[1]

# Attach the soft body to the end of the joint by setting the soft body's base position at the joint's position
p.resetBasePositionAndOrientation(soft_body_id1, joint_pos, joint_orn)

# Use a fixed constraint to attach the soft body to the rigid body joint
# Attach the center of the soft body (anchor) to the joint
p.createSoftBodyAnchor(soft_body_id1, 0, rigid_body_id, joint_index)


for i in range(10000):
    p.stepSimulation()
    #time.sleep(1./240.)

p.disconnect()