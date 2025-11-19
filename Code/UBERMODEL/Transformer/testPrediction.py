from trainTransformer import MotorTransformer, generate_noisy_sine_dataset
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import torch
def show_example(channels,expected):
    example = channels.detach().numpy()[0]
    pred = expected.detach().numpy()[0]
    timesteps = np.arange(len(example)) * dt

    plt.figure(figsize=(8, 5), dpi=200)  # good for A4 export

    plt.plot(timesteps, example, c="r", label="Truth", linewidth=2)
    plt.plot(timesteps, pred, "--", c="b", label="Predicted", linewidth=2)

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Truth vs Predicted Signal", fontsize=16)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()
if __name__=="__main__":
    N = 100     # number of sequences
    T = 200     # timesteps
    V1 = 20      # joints
    V2 = 12      # joints
    model = MotorTransformer(input_dim=V1,output_dim=V2,T=T).to(device)
    model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/motor_transformer.pth"))
    model.eval()
    

    #X, y = generate_noisy_sine_dataset(N, T, V1,V2, noise_std=0.2)
    X=torch.tensor(np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA.npy"),  dtype=torch.float32)[:100].to(device)
    y=torch.tensor(np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/y_DATA.npy"),  dtype=torch.float32)[:100].to(device)
    pred=model(X)
    show_example(y, pred)
