import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, '/its/home/drs25/Quadruped/Code')
import environment

###################
# Transformer
###################


###################
# LSTM
###################


#select model that is used
model=
#create environment 
vector=[_,_,_]
friction=0.5
env=environment(0,friction=friction)
timesteps=100
dt=0.1
#loop through time steps
for i in np.arange(0,timesteps,dt):
    #get observation
    feet=env.quad.getFeet()
    positions=env.quad.getPositions()
    data=np.concatenate(positions,feet,[friction])
    motors=model(data)
    env.quad.setPositions(motors)#get model output
    for k in range(10): #update simulation
        p.stepSimulation()
        p.setTimeStep(1./240.)
#directly set positions of robot to model outputs


#save trajectory 
#show trajectory for sake of debugging 


