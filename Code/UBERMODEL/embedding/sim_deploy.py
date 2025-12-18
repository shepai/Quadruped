import numpy as np 
import matplotlib.pyplot as plt 
from models.train_to_predict_traj import RegressiveLSTM
import torch
from models.test_to_predict_traj import reform
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')

###################
# Data tools
###################
X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_y.npy")
y=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_X.npy")
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).reshape((len(y),-1))
mean = X.mean(axis=1, keepdims=True)
std  = X.std(axis=1, keepdims=True) + 1e-8

X = (X - mean) / std
scaler = StandardScaler()
y = scaler.fit_transform(y)

###################
# LSTM
###################

device="cuda" if torch.cuda.is_available() else "cpu"
model = RegressiveLSTM(input_size=11, hidden_size=64, num_layers=2, output_size=60).to(device)

model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/regressive_lstm_norm_t1.pth", map_location=device))
model.eval()

###################
# Sim
###################

idx = 0#np.random.randint(0,len(X))
x_sample = torch.tensor(X[idx],dtype=torch.float32)          # shape: [seq_len, 11]
y_true = scaler.inverse_transform(y)
y_true = y_true[idx].reshape((12,5))

x_sample = x_sample.unsqueeze(0)

friction=0.5
env=environment(0,friction=friction)
env.reset()

#loop through time steps
def runTrial(env):
    vector=[0.01,0,0]
    timesteps=100
    dt=0.1
    rolling=[]
    traj=[]
    motors_=[]
    env.reset()
    for i in np.arange(0,timesteps,dt):
        #get observation
        feet=env.quad.getFeet()
        positions=env.quad.getPositions()
        data=np.concatenate([vector,feet,[friction]])
        #data=(data-np.min(X))/(np.max(X)-np.min(X))
        rolling.append(data)
        if len(rolling)>15:
            rolling.pop(0)
        data=np.array(rolling)
        if len(data.shape)<3:
            data=data.reshape((1,len(data),20))

        motors=model(torch.tensor(data,dtype=torch.float32).to(device)).cpu().detach().numpy()[-1][-1]
        m=[]
        for motor in motors:
            m.append(reform(*motor,repeat=timesteps))
        motors=np.array(m)[:,i] #gather the timestep of this phase
       
        env.quad.setPositions(motors)#get model output
        motors_.append(motors) #.cpu().detach().numpy()[-1][-1]
        for k in range(10): #update simulation
            p.stepSimulation()
            p.setTimeStep(1./240.)
        traj.append(env.quad.getPos())
    #directly set positions of robot to model outputs
    traj=np.array(traj)
    return traj

traj=runTrial(env)

plt.plot(traj[:,0],traj[:,1])
plt.savefig("/its/home/drs25/Quadruped/Code/UBERMODEL/embedding/models/traj.pdf")
plt.show()