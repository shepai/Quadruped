path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/PressTip/urdf/"
#path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import Quadruped
import numpy as np
#from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
import time
import os
clear = lambda: os.system('clear')

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
        initial_position = [0, 0, 0.15]  # x=1, y=2, z=0.5
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
        self.observation_space=spaces.Box(low= -10, high = 300, shape = (16,), dtype = np.float32)#spaces.Discrete(9)
        self.start_position=self.quad.start
        self.timer=0
        self.dt=0.01
        self.reward_history=[]
    def step(self,action):
        action[action<-1]=-1
        action[action>1]=1
        self.quad.setPositions(np.degrees(np.arcsin(action))/2)
        T=0
        self.timer+=self.dt
        for i in range(50):
            p.stepSimulation()
            p.setTimeStep(1./240.)
            #time.sleep(1/240.)
        orientation = self.quad.getOrientation()
        foot_pressure = self.quad.getFeet()
        curr=self.quad.getPos()
        self.observation_space = np.array(np.concatenate([foot_pressure/10, np.radians(self.quad.motors)]), dtype = np.float32).reshape((16,))
        alpha=0.6
        beta=0.2
        gamma=0.4
        change_in_orient=euclidean_distance(self.quad.start_orientation,self.quad.getOrientation())
        distance_moved = curr[0]- self.start_position[0]  # Forward movement in x-direction
        reward = (alpha * distance_moved) - (beta*np.abs(curr[1]-self.start_position[1])) - (gamma*change_in_orient) # Penalize deviation from straight line
        if self.view:
            basePos, baseOrn = p.getBasePositionAndOrientation(self.id) # Get model position
            self.p.resetDebugVisualizerCamera( cameraDistance=0.3, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos) # fix camera onto model
	
        # Check if the episode should end (e.g., robot falls)
        done = self.quad.hasFallen()
        if done: self.timer=0
        self.reward_history.append(reward)
        #reward*=self.timer
        clear()
        print("Step",len(self.reward_history),reward,max(self.reward_history))
        return self.observation_space, reward, done, {}, {}
    def observation(self):
        orientation = self.quad.getOrientation()
        foot_pressure = self.quad.getFeet()
        return np.concatenate([foot_pressure/10, np.radians(self.quad.motors)])
    def reset(self,seed=0):
        self.p.removeBody(self.id)
        del self.quad
        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.81)
        plane_id = self.p.loadURDF('plane.urdf')
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        initial_position = [0, 0, 0.15]  # x=1, y=2, z=0.5
        initial_orientation = self.p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
        flags = self.p.URDF_USE_SELF_COLLISION
        self.id = self.p.loadURDF(path+"Quadruped_prestip.urdf", initial_position, initial_orientation,flags=flags)
        
        self.quad=Quadruped.Quadruped(self.p,self.id,self.plane_id)
        self.quad.neutral=[-10,0,30,0,0,0,0,0,0,0,0,0]
        self.quad.reset()
        curr=self.quad.getPos()
        self.observation_space = np.zeros((16,), dtype=np.float32)
        return self.observation_space,{}
    def close(self):
        self.p.removeBody(self.robot_id)
        del self.quad
    def render(self, mode='human'):
        pass

# Initialize the PyBullet physics engine
p.connect(p.GUI) #
p.setAdditionalSearchPath(pybullet_data.getDataPath())
env=GYM(p)

#check_env(env)

# Initialize PPO model
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_quadruped_tensorboard/",device="cuda")
policy_kwargs = dict(net_arch=dict(pi=[32, 64], qf=[100, 50]))
model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# Train the model
model.learn(total_timesteps=100000)  # Adjust the number of timesteps as needed

# Save the model
model.save("C:/Users/dexte/Documents/GitHub/Quadruped/modelRF_A2C_2.zip")
p.disconnect()
p.connect(p.GUI) #DIRECT
p.setAdditionalSearchPath(pybullet_data.getDataPath())
env=GYM(p)
# Test the trained model
obs,_ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info, _ = env.step(action)
    env.render()
    if done:
        obs,_ = env.reset()
p.disconnect()
print("*********************************\n\n\nTIME TAKEN",(time.time()-t1) /(60*60))