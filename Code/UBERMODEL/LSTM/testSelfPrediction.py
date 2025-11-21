import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
from PytorchLSTMTrain import *
#matplotlib.use('TkAgg')

def test_data(X):
    # Register the missing operation used by Masking()
    model = load_model("/its/home/drs25/Quadruped/Code/UBERMODEL/models/gait_model.keras")

    next_motors = model.predict(X)
    return next_motors
def display_data(y, Pred, dt=0.1):
    example = y[0]
    pred = Pred[0]
    print(example.shape,pred.shape)
    timesteps = np.arange(len(example)) * dt

    plt.figure(figsize=(8, 5), dpi=200)  # good for A4 export

    plt.plot(timesteps, example, c="r", label="Truth", linewidth=2)
    plt.plot(timesteps, pred, "--", c="b", label="Predicted", linewidth=2)

    plt.xlabel("Time (dt)", fontsize=14)
    plt.ylabel("Position (Degrees)", fontsize=14)
    plt.title("Truth vs Predicted Signal", fontsize=16)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig("/its/home/drs25/Quadruped/Code/UBERMODEL/LSTM/example_lstm.pdf")

if __name__=="__main__":
    #load in the data 
    #X, y = generate_noisy_sine_dataset(100, 200, 20, 12, noise_std=0.2)
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/X_DATA.npy")[:100]
    X=(X-np.min(X))/(np.max(X)-np.min(X))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32)
    INPUT_DIM=X_tensor.shape[2]
    OUTPUT_DIM=12
    model = LSTMModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/lstm_gait_autoregressiveDeltas.pth"))
    model.eval()
    recorded = []
    window=10
    x = X_tensor[0:1, 0:window].to(device)  # shape (1,1,16)
    TIME=100
    
    for i in range(window,TIME):
        pred = model(x)          # (1,12)
        #pred = pred.unsqueeze(2) # (1,1,12)
        motors=X_tensor[0:1,i-window:i,:12].to(device)+pred
        recorded.append(motors.cpu().detach().numpy())

        remainder = X_tensor[0:1, i-window:i, 12:].to(device) # should be (1,window,8)

        # fix any extra accidental dims
        remainder = remainder.reshape(1,window,8)
        x = torch.cat([motors, remainder], dim=2)
        x = x.reshape(1,window,20)    # keep clean shape

    recorded = np.concatenate(recorded, axis=1)   # -> (1, TIME, 12)
    print(recorded.shape)
    display_data(X[0:1,:TIME,:12], recorded[0:1,:TIME-1])




