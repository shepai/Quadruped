from trainTransformer import MotorTransformer, generate_noisy_sine_dataset
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import torch
def show_example(channels,expected):

    plt.plot(channels.detach().numpy(),c="b")
    plt.plot(expected.detach().numpy(),c="r")
    plt.show()
if __name__=="__main__":
    N = 100     # number of sequences
    T = 200     # timesteps
    V1 = 20      # joints
    V2 = 12      # joints
    model = MotorTransformer(input_dim=V1,output_dim=V2,T=T)
    model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/motor_transformer.pth"))
    model.eval()
    

    X, y = generate_noisy_sine_dataset(N, T, V1,V2, noise_std=0.2)
    pred=model(X)
    show_example(X[0][:,0:12], pred[0])
