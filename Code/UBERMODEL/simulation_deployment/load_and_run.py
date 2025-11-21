import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
#matplotlib.use('TkAgg')
import sys
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, '/its/home/drs25/Quadruped/Code')
from environment import *

###################
# Transformer
###################
"""from UBERMODEL.Transformer.trainTransformer import MotorTransformer, generate_noisy_sine_dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MotorTransformer(input_dim=20,output_dim=12,T=200).to(device)
model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/gait_transformer.pth"))
model.eval()
"""
###################
# LSTM
###################
"""from tensorflow.keras.models import load_model
import tensorflow as tf
model=load_model("/its/home/drs25/Quadruped/Code/UBERMODEL/models/gait_model.keras")"""
from UBERMODEL.LSTM.PytorchLSTMTrain import LSTMModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_dim=20, output_dim=12).to(device)
model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/lstm_gait_autoregressiveDeltas.pth"))
model.eval()

#create environment 
vector=[0.1,0,0]
friction=0.5
env=environment(0,friction=friction)
env.reset()
timesteps=100
dt=0.1
#loop through time steps
rolling=[]
traj=[]
motors_=[]
X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/X_DATA.npy")[:100]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in np.arange(0,timesteps,dt):
    #get observation
    feet=env.quad.getFeet()
    positions=env.quad.getPositions()
    data=np.concatenate([positions,vector,feet,[friction]])
    data=(data-np.min(X))/(np.max(X)-np.min(X))
    rolling.append(data)
    if len(rolling)>10:
        rolling.pop(0)
    data=np.array(rolling)
    if len(data.shape)<3:
        data=data.reshape((1,len(data),20))
    
    motors=model(torch.tensor(data,dtype=torch.float32).to(device))
    #motors=model(torch.tensor(data,  dtype=torch.float32).to(device)).cpu().detach().numpy()
    motors=motors[-1][-1].cpu().detach().numpy()+rolling[-1][:12]
    env.quad.setPositions(motors)#get model output
    motors_.append(motors) #.cpu().detach().numpy()[-1][-1]
    for k in range(10): #update simulation
        p.stepSimulation()
        p.setTimeStep(1./240.)
    traj.append(env.quad.getPos())
#directly set positions of robot to model outputs
traj=np.array(traj)
plt.plot(traj[:,0],traj[:,1])
plt.show()
plt.plot(motors_)
plt.savefig("/its/home/drs25/Quadruped/assets/plots/traj.pdf")
#save trajectory 
#show trajectory for sake of debugging 


