path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import Quadruped
import numpy as np
from stable_baselines3.common.env_checker import check_env
import gym
from gym import spaces
from stable_baselines3 import PPO
import time
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
t1=time.time()
class GYM(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,sim,view=True):
        super(GYM, self).__init__()
        self.p=sim
        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.81)
        self.plane_id = self.p.loadURDF('plane.urdf')
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        initial_position = [0, 0, 0.1]  # x=1, y=2, z=0.5
        initial_orientation = self.p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
        flags = self.p.URDF_USE_SELF_COLLISION
        self.id = self.p.loadURDF(path+"Quadruped_prestip.urdf", initial_position, initial_orientation,flags=flags)
        self.view=view
        self.quad=Quadruped.Quadruped(self.p,self.id,self.plane_id)
        self.quad.neutral=[-10,0,30,0,0,0,0,0,0,0,0,0]
        self.quad.reset()
        self.robot_id=id
        self.p=p
        self.action_space = spaces.Box(low= -0.1, high = 1, shape = (12,), dtype = np.float32)
        self.robot_position=self.quad.getPos()
        self.observation_space=spaces.Box(low= -10, high = 300, shape = (20,), dtype = np.float32)#spaces.Discrete(9)
        self.start_position=self.quad.start
    def step(self,action):
        self.quad.setPositions(np.degrees(action))
        for i in range(50):
            p.stepSimulation()
            p.setTimeStep(1./240.)
            #time.sleep(1/240.)
        orientation = self.quad.getOrientation()
        foot_pressure = self.quad.getFeet()
        curr=self.quad.getPos()
        preshape=self.observation_space
        self.observation_space = np.array(np.concatenate([foot_pressure, orientation,self.quad.motors]), dtype = np.float32).reshape((20,))
        #self.observation_space[self.observation_space<0]=0

        distance_moved = curr[0]- self.start_position[0]  # Forward movement in x-direction
        reward = 10 * distance_moved - np.abs(curr[1])  # Penalize deviation from straight line
        if self.view:
            basePos, baseOrn = p.getBasePositionAndOrientation(self.id) # Get model position
            self.p.resetDebugVisualizerCamera( cameraDistance=0.3, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos) # fix camera onto model
	
        # Check if the episode should end (e.g., robot falls)
        done = self.quad.hasFallen()
        return self.observation_space, reward, done, {}
    def observation(self):
        orientation = self.quad.getOrientation()
        foot_pressure = self.quad.getFeet()
        return np.concatenate([foot_pressure, orientation,self.quad.motors])
    def reset(self):
        self.p.removeBody(self.id)
        del self.quad
        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.81)
        plane_id = self.p.loadURDF('plane.urdf')
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        initial_position = [0, 0, 0.3]  # x=1, y=2, z=0.5
        initial_orientation = self.p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
        flags = self.p.URDF_USE_SELF_COLLISION
        self.id = self.p.loadURDF(path+"Quadruped_prestip.urdf", initial_position, initial_orientation,flags=flags)
        
        self.quad=Quadruped.Quadruped(self.p,self.id,self.plane_id)
        self.quad.neutral=[-10,0,30,0,0,0,0,0,0,0,0,0]
        self.quad.reset()
        curr=self.quad.getPos()
        self.observation_space = np.zeros((20,), dtype=np.float32)
        return self.observation_space
    def close(self):
        self.p.removeBody(self.robot_id)
        del self.quad
    def render(self, mode='human'):
        pass

# Initialize the PyBullet physics engine
p.connect(p.DIRECT) #
p.setAdditionalSearchPath(pybullet_data.getDataPath())
env=GYM(p)

#check_env(env)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_quadruped_tensorboard/")

# Train the model
model.learn(total_timesteps=1)  # Adjust the number of timesteps as needed

# Save the model
model.save("ppo_quadruped_model.zip")
p.disconnect()
p.connect(p.GUI) #DIRECT
p.setAdditionalSearchPath(pybullet_data.getDataPath())
env=GYM(p)
# Test the trained model
obs = env.reset()
for _ in range(2):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

print("*********************************\n\n\nTIME TAKEN",(time.time()-t1) /(60*60))