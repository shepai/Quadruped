path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/PressTip/urdf/"
path="/its/home/drs25/Quadruped/Quadruped_sim/PressTip/urdf/"
#path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/PressTip/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
import Quadruped
import numpy as np
import os
import random
from copy import deepcopy
import uuid
try:
    import gymnasium as gym
    from gymnasium import spaces
except:
    try:
        import gym
        from gym import spaces
        from stable_baselines3 import PPO, A2C
    except:
        pass
import torch
def demo(variable,history={}):
    return 0
class environment:
    def __init__(self,show=False,record=False,filename="",friction=0.5,UI=1,floorpath='plane.urdf'):
        self.show=show
        if show: p.connect(p.GUI)
        else: p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Ensure GUI is enabled
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Hide Explorer
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # Hide RGB view
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Hide Depth view
        self.friction=friction
        self.robot_id=None
        self.plane_id=None
        self.quad=None
        self.record=record
        self.filename=filename
        self.recording=0
        self.history={}
        self.UI=UI
        if self.show and UI:
            self.x_slider = p.addUserDebugParameter("dt", -5, 5, 0.1)
        self.INCREASE=0
        self.BALANCE=0
        self.STRIDE=1
        self.plane_file=floorpath
    def take_agent_snapshot(self,p, agent_id, alpha=0.1, width=640, height=480):
        # Make all objects except the agent transparent
        num_bodies = p.getNumBodies()
        for i in range(num_bodies):
            body_id = p.getBodyUniqueId(i)
            if body_id != agent_id:
                visual_shapes = p.getVisualShapeData(body_id)
                for visual in visual_shapes:
                    p.changeVisualShape(body_id, visual[1], rgbaColor=[1, 1, 1, alpha])  # Set transparency
        # Capture snapshot from the current camera view
        _, _, img_arr, _, _ = p.getCameraImage(width, height)
        # Convert the image array to a NumPy array
        return np.array(img_arr, dtype=np.uint8)
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF(self.plane_file,[-1, -1, -0.1], useFixedBase=True)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Ensure GUI is enabled
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Hide Explorer
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # Hide RGB view
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Hide Depth view
        p.changeDynamics(self.plane_id, -1, lateralFriction=self.friction)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.changeDynamics(self.plane_id, -1, lateralFriction=self.friction)
        initial_position = [0, 0, 5.8]  # x=1, y=2, z=0.5
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
        flags = p.URDF_USE_SELF_COLLISION
        self.robot_id = p.loadURDF(path+"Quadruped_prestip.urdf", initial_position, initial_orientation,flags=flags)
        p.changeDynamics(self.robot_id, -1, lateralFriction=self.friction)
        self.quad=Quadruped.Quadruped(p,self.robot_id,self.plane_id)
        self.quad.neutral=[-30, 0, 40, -30, 50, -10, 0, 10, 20, 30, -30, 50]
        self.quad.reset()
        for i in range(500):
            p.stepSimulation()
            p.setTimeStep(1./240.)
        if self.record:
            self.video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.filename)
            self.recording=1
        #if self.show:
            #self.x_slider = p.addUserDebugParameter("dt", -2, 2, 0.1)
    """def step(self,delay=0,T=1,dt=1):
        t=0
        while t<T:
            p.stepSimulation()
            if delay: time.sleep(1./240.)
            else: p.setTimeStep(1./240.)
            t+=dt"""
    def runTrial(self,agent,generations,delay=False,fitness=demo,photos=-1):
        history={}
        history['positions']=[]
        history['orientations']=[]
        history['motors']=[]
        history['accumalitive_reward']=[]
        history['self_collisions']=[]
        history['feet']=[]
        self.reset()
        a=[]
        photos_l=[]
        for i in range(generations*10):
            pos=self.step(agent,0,delay=delay)
            if photos>-1 and i%photos==0:
                print("snap")
                photos_l.append(self.take_agent_snapshot(p,self.robot_id))
            #pos[[2,5,8,11]]=180-pos[[1,4,7,10]]
            basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id) # Get model position
            history['positions'].append(basePos)
            history['orientations'].append(baseOrn[0:3])
            history['motors'].append(pos)
            history['accumalitive_reward'].append(fitness(self.quad,history=history))
            history['self_collisions'].append(self.quad.get_self_collision_count())
            history['feet'].append(self.quad.getFeet())
            p.resetDebugVisualizerCamera( cameraDistance=0.3, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos) # fix camera onto model
            if self.quad.hasFallen():
                break
            if self.quad.hasFallen():
                break
            a.append(pos)
        history['positions']=np.array(history['positions'])
        history['orientations']=np.array(history['orientations'])
        history['motors']=np.array(history['motors'])
        history['accumalitive_reward']=np.array(history['accumalitive_reward'])
        history['self_collisions']=np.array(history['self_collisions'])
        history['feet']=np.array(history['feet'])
        filename = str(uuid.uuid4())
        #np.save("/its/home/drs25/Documents/GitHub/Quadruped/Code/data_collect_proj/trials_all/"+str(filename),history)
        return fitness(self.quad,history=history),history,photos_l
    def step(self,agent,action,delay=False,gen=0):
        if self.show and self.UI:
            agent.dt=p.readUserDebugParameter(self.x_slider)
        motor_positions=agent.get_positions(np.array(self.quad.getOrientation()))
        motor_positions[[1,10]]+=self.INCREASE
        motor_positions[[7,4]]-=self.INCREASE
        #print(motor_positions[[2,5,8,11]])
        motor_positions[[2,11]]+=self.BALANCE#*(motor_positions[[2,5,8,11]]/np.abs(motor_positions[[2,5,8,11]]))
        motor_positions[[8,5]]-=self.BALANCE
        motor_positions[[0,3,6,9]]*=self.STRIDE
        #print("=",motor_positions[[2,5,8,11]])
        #print(motor_positions)
        self.quad.setPositions(motor_positions)
        for k in range(10): #update simulation
            p.stepSimulation()
            if delay: time.sleep(1./240.)
            else: p.setTimeStep(1./240.)
            basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id) # Get model position
            p.resetDebugVisualizerCamera( cameraDistance=0.3, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos) # fix camera onto model
            if self.quad.hasFallen():
                
                break
        return motor_positions
    def stop(self):
        if self.recording and self.record:
            p.stopStateLogging(self.video_log_id)
            self.recording=0
    def close(self):
        p.disconnect()

class GYM(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,show=False,record=False,filename="",friction=0.5,UI=1):
        self.show=show
        if show: p.connect(p.GUI)
        else: p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Ensure GUI is enabled
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Hide Explorer
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # Hide RGB view
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Hide Depth view
        self.friction=friction
        self.robot_id=None
        self.plane_id=None
        self.quad=None
        self.record=record
        self.filename=filename
        self.recording=0
        self.history={}
        self.UI=UI
        if self.show and UI:
            self.x_slider = p.addUserDebugParameter("dt", -5, 5, 0.1)
        self.action_space = spaces.Box(low= -0.1, high = 1, shape = (12,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)#spaces.Discrete(9)
        self.fitness_over_time=[]
    def take_agent_snapshot(self,p, agent_id, alpha=0.1, width=640, height=480):
        # Make all objects except the agent transparent
        num_bodies = p.getNumBodies()
        for i in range(num_bodies):
            body_id = p.getBodyUniqueId(i)
            if body_id != agent_id:
                visual_shapes = p.getVisualShapeData(body_id)
                for visual in visual_shapes:
                    p.changeVisualShape(body_id, visual[1], rgbaColor=[1, 1, 1, alpha])  # Set transparency
        # Capture snapshot from the current camera view
        _, _, img_arr, _, _ = p.getCameraImage(width, height)
        # Convert the image array to a NumPy array
        return np.array(img_arr, dtype=np.uint8)
    def reset(self,seed=0):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF('plane.urdf')
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Ensure GUI is enabled
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Hide Explorer
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # Hide RGB view
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Hide Depth view
        p.changeDynamics(self.plane_id, -1, lateralFriction=self.friction)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.changeDynamics(self.plane_id, -1, lateralFriction=self.friction)
        initial_position = [0, 0, 5.8]  # x=1, y=2, z=0.5
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
        flags = p.URDF_USE_SELF_COLLISION
        self.robot_id = p.loadURDF(path+"Quadruped_prestip.urdf", initial_position, initial_orientation,flags=flags)
        p.changeDynamics(self.robot_id, -1, lateralFriction=self.friction)
        self.quad=Quadruped.Quadruped(p,self.robot_id,self.plane_id)
        self.quad.neutral=[-30, 0, 40, -30, 50, -10, 0, 10, 20, 30, -30, 50]
        self.quad.reset()
        for i in range(500):
            p.stepSimulation()
            p.setTimeStep(1./240.)
        if self.record:
            self.video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.filename)
            self.recording=1
        #if self.show:
            #self.x_slider = p.addUserDebugParameter("dt", -2, 2, 0.1)
        try:
            self.fitness_over_time.append(self.history['accumalitive_reward'][-1]) #collect over time
        except:
            pass
        self.history={}
        self.history['positions']=[]
        self.history['orientations']=[]
        self.history['motors']=[]
        self.history['accumalitive_reward']=[]
        self.history['self_collisions']=[]
        self.history['feet']=[]
        self.agent.t=0
        return self.observation(), {}
    def setFitness(self,fitness):
        self.fitness=fitness
    def setAgent(self,agent):
        self.agent=agent
    def step(self,action,delay=False,gen=0):
        self.agent.set_genotype(action)
        motor_positions=self.agent.step(np.array(self.quad.getOrientation()))*40
        self.quad.setPositions(motor_positions)
        for k in range(10): #update simulation
            p.stepSimulation()
            if delay: time.sleep(1./240.)
            else: p.setTimeStep(1./240.)
            basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id) # Get model position
            p.resetDebugVisualizerCamera( cameraDistance=0.3, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos) # fix camera onto model
            if self.quad.hasFallen():
                break
        motor_positions[[2,5,8,11]]=180-motor_positions[[1,4,7,10]]
        basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id) # Get model position
        done = self.quad.hasFallen()
        self.history['positions'].append(basePos)
        self.history['orientations'].append(baseOrn[0:3])
        self.history['motors'].append(motor_positions)
        self.history['accumalitive_reward'].append(self.fitness(done,history=self.history))
        self.history['self_collisions'].append(self.quad.get_self_collision_count())
        self.history['feet'].append(self.quad.getFeet())
        #self.observation_space=self.observation()
        reward=self.fitness(done,self.history)
        return self.observation(), reward, done, False, {}
    def observation(self):
        #basePos#, baseOrn = p.getBasePositionAndOrientation(self.robot_id) # Get model position
        return np.array([self.friction], dtype=np.float32)
    def stop(self):
        if self.recording and self.record:
            p.stopStateLogging(self.video_log_id)
            self.recording=0
    
    def render(self, mode='human'):
        pass
    def close(self):
        p.disconnect()
