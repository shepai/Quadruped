import numpy as np 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers, models
def generate_noisy_sine_dataset(N, T, V1,V2, noise_std=0.1):
    """
    N = number of sequences
    T = timesteps
    V = joints
    """
    X = np.zeros((N, T, V1))
    y = np.zeros((N, T, V2))

    for n in range(N):
        t = np.linspace(0, 4*np.pi, T+1)  # +1 for next-step target

        # Make each joint have its own amplitude, frequency, phase, offset
        amp   = np.random.uniform(0.5, 2.5, size=V1)
        freq  = np.random.uniform(0.5, 2.0, size=V1)
        phase = np.random.uniform(0, 2*np.pi, size=V1)
        off   = np.random.uniform(-1.0, 1.0, size=V1)

        # Build the signal
        signal = np.zeros((T+1, V1))
        for j in range(V1):
            signal[:, j] = off[j] + amp[j] * np.sin(freq[j] * t + phase[j])

        # Add noise
        signal += np.random.normal(scale=noise_std, size=signal.shape)

        # Fill X[t] = signal[t], y[t] = signal[t+1]
        X[n] = signal[:-1]
        amp   = np.random.uniform(0.5, 2.5, size=V2)
        freq  = np.random.uniform(0.5, 2.0, size=V2)
        phase = np.random.uniform(0, 2*np.pi, size=V2)
        off   = np.random.uniform(-1.0, 1.0, size=V2)

        # Build the signal
        signal = np.zeros((T+1, V2))
        for j in range(V2):
            signal[:, j] = off[j] + amp[j] * np.sin(freq[j] * t + phase[j])

        # Add noise
        signal += np.random.normal(scale=noise_std, size=signal.shape)
        y[n] = signal[1:]

    return X, y
if __name__=="__main__":
    #load in the data 
    #X, y = generate_noisy_sine_dataset(100, 200, 20, 12, noise_std=0.2)
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA.npy")[100:]
    y=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/y_DATA.npy")[100:]
    input_dim = X.shape[-1]
    output_dim = y.shape[-1]
    inp = layers.Input(shape=(None, input_dim))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(output_dim)(x)

    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    #train the model
    model.fit(
        X, y,
        epochs=1500,
        batch_size=32,
        validation_split=0.1
    )
    model.save("/its/home/drs25/Quadruped/Code/UBERMODEL/models/gait_model.keras")