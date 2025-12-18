import numpy as np 
import matplotlib.pyplot as plt 
from models.train_to_predict_traj import RegressiveLSTM
import torch
from models.test_to_predict_traj import reform
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import matplotlib
import sys
from collections import deque
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, '/its/home/drs25/Quadruped/Code')
from environment import *
matplotlib.use('TkAgg')

###################
# Data tools
###################
X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_y.npy")[0:1000]
y=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_X.npy")[0:1000]
np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_y_lite.npy",X)
np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_X_lite.npy",y)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).reshape((len(y),-1))
mean = X.mean()
std  = X.std() + 1e-8

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

         # shape: [seq_len, 11]
y_true = scaler.inverse_transform(y)
y_true = y_true[idx].reshape((12,5))

#x_sample = x_sample.unsqueeze(0).to(device)

env=environment(0,friction=friction)
env.reset()

#loop through time steps
def runTrial(env, model, start_point):

    vector = np.array([0.01, 0.0, 0.0])
    timesteps = 500
    dt = 0.1
    sim_steps = int(timesteps / dt)

    rolling = deque(maxlen=1000)
    for x in start_point:
        rolling.append(x)

    traj = []
    motors_ = []

    env.reset()
    p.setTimeStep(1.0 / 240.0)

    motor_waves = None  # cache

    with torch.no_grad():
        for step in range(sim_steps):
            print(step,"/",sim_steps)
            i = step * dt
            # ---- Observation ----
            feet = env.quad.getFeet()
            pos, orientation = env.get_pos_or()
            obs = np.concatenate([
                vector,
                feet,
                [friction],
                orientation[:3]
            ])
            obs = (obs - mean) / std
            rolling.append(obs)
            data = np.asarray(rolling)[np.newaxis, :, :]
            # ---- Model ----
            motors_raw = model(
                torch.from_numpy(data).float().to(device)
            ).cpu().numpy()
            motors_raw = scaler.inverse_transform(motors_raw)[-1].reshape(12, 5)

            # ---- Waveform cache (only once) ----
            if motor_waves is None:
                waves = []
                for motor in motors_raw:
                    _, wave = reform(*motor, repeat=timesteps * 10)
                    waves.append(wave[:sim_steps])
                motor_waves = np.asarray(waves)

            # ---- Select current timestep ----
            try:
                motors = motor_waves[:, step]
            except IndexError:
                motors = env.quad.getPositions()

            env.quad.setPositions(motors)
            motors_.append(motors)

            # ---- Physics ----
            for _ in range(10):
                p.stepSimulation()

            traj.append(env.quad.getPos())

    traj = np.asarray(traj)
    motors_ = np.asarray(motors_)

    return traj, motors_
for i in range(5):
    idx = np.random.randint(0,len(X))
    x_sample = torch.tensor(X[idx],dtype=torch.float32)[0:800] 
    friction=x_sample[0][6]
    traj,motors=runTrial(env,model,x_sample)
    np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/embedding/assets/traj"+str(i),traj)
    np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/embedding/assets/motor"+str(i),motors)
    plt.plot(traj[:,0],traj[:,1])
    plt.savefig("/its/home/drs25/Quadruped/Code/UBERMODEL/embedding/assets/traj"+str(i)+".pdf")
    plt.cla()