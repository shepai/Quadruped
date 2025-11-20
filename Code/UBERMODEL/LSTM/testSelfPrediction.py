from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
from trainModel import generate_noisy_sine_dataset
matplotlib.use('TkAgg')

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
    plt.show()

if __name__=="__main__":
    #load in the data 
    #X, y = generate_noisy_sine_dataset(100, 200, 20, 12, noise_std=0.2)
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA.npy")[0:100]
    X = np.concatenate([X[..., :15],      # first 15
                    X[..., -1:],      # last value (keep dims)
                   ], axis=-1)
    y=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/y_DATA.npy")[0:100]
    recorded=[]
    x=X[0:1,0:1]
    for i in range(100):
        pred=test_data(x)
        recorded.append(pred.copy())
        remainder = x[:, :, 12:] 
        x = np.concatenate([pred, remainder], axis=2)

    recorded=np.array(recorded).reshape((1,100,12))
    print(recorded.shape,y[0:1,].shape)
    display_data(y[0:1,:], recorded[0:1,:99])




