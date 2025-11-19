from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
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
    #load in the data 
    X=np.random.random((100,200,20)) #random for now
    y=np.random.random((100,200,12))
    pred=test_data(X)
    display_data(y, pred)




