from trainTransformer import MotorTransformer, generate_noisy_sine_dataset
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
#matplotlib.use('TkAgg')
import torch

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
    plt.ylim([-20,20])
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig("/its/home/drs25/Quadruped/Code/UBERMODEL/Transformer/example.pdf")

if __name__=="__main__":
    #load in the data 
    #X, y = generate_noisy_sine_dataset(100, 200, 20, 12, noise_std=0.2)
    """X = np.concatenate([X[..., :15],      # first 15
                    X[..., -1:],      # last value (keep dims)
                   ], axis=-1)"""
    N = 100     # number of sequences
    T = 200     # timesteps
    V2 = 12      # joints
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA_one.npy")[0:1]
    """X = np.concatenate([X[..., :15],      # first 15
                    X[..., -1:],      # last value (keep dims)
                   ], axis=-1)"""
    mean=np.mean(X)
    std=np.std(X)
    X = (X - mean) / std
    V1 = X[:,:,:].shape[2]     # joints
    X=torch.tensor(X,  dtype=torch.float32).to(device)
    model = MotorTransformer(input_dim=V1,output_dim=V2,T=T).to(device)
    model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/transformer_gait_one1.pth"))
    model.eval()
    
    recorded = []
    window=15
    x = X[0:1, 0:window,:].to(device)  # shape (1,1,16)
    TIME=1000

    for i in range(window,TIME-window):
        pred = model(x)          # (1,12)
        #pred = pred.unsqueeze(2) # (1,1,12)
        motors=pred#X[0:1,i-window:i,:12].to(device)+pred
        recorded.append(motors.cpu().detach().numpy())

        remainder = X[0:1, i-window:i, 12:].to(device) # should be (1,window,8)

        # fix any extra accidental dims
        remainder = remainder.reshape(1,window,X.shape[2]-12)
        x = torch.cat([motors, remainder], dim=2)
        x = x.reshape(1,window,V1)    # keep clean shape
        #x=remainder

    recorded = np.concatenate(recorded, axis=1)  # (1,100,12)
    
    display_data(X[0:1,:TIME,:12].cpu().detach().numpy(), recorded[0:1,:TIME])




