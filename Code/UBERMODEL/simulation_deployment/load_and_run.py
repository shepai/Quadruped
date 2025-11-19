import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, '/its/home/drs25/Quadruped/Code')
from environment import *

###################
# Transformer
###################
"""from Code.UBERMODEL.Transformer.trainTransformer import MotorTransformer, generate_noisy_sine_dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MotorTransformer(input_dim=V1,output_dim=V2,T=T).to(device)
model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/gait_transformer.pth"))
model.eval()
"""
###################
# LSTM
###################
from tensorflow.keras.models import load_model
import tensorflow as tf
model=load_model("/its/home/drs25/Quadruped/Code/UBERMODEL/models/gait_model.keras")


#create environment 
vector=[0.01,0,0]
friction=0.5
env=environment(0,friction=friction)
env.reset()
timesteps=100
dt=0.1
#loop through time steps
rolling=[]
traj=[]
for i in np.arange(0,timesteps,dt):
    #get observation
    feet=env.quad.getFeet()
    positions=env.quad.getPositions()
    data=np.concatenate([positions,vector,feet,[friction]])
    rolling.append(data)
    if len(rolling)>100:
        rolling.pop(0)
    data=np.array(rolling)
    if len(data.shape)<3:
        data=data.reshape((1,len(data),20))
    motors=model.predict(data)
    #motors=model(data.to(device))
    env.quad.setPositions(motors[-1][-1])#get model output
    for k in range(10): #update simulation
        p.stepSimulation()
        p.setTimeStep(1./240.)
    traj.append(env.quad.getPos())
#directly set positions of robot to model outputs
traj=np.array(traj)
plt.plot(traj[:,0],traj[:,1])
plt.show()
#save trajectory 
#show trajectory for sake of debugging 


