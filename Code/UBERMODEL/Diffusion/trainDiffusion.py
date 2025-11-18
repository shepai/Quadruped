import numpy as np 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers, models

if __name__=="__main__":
    #load in the data 
    X=np.random.random((100,200,20)) #random for now
    y=np.random.random((100,200,12))
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
        epochs=100,
        batch_size=32,
        validation_split=0.1
    )
    model.save("/its/home/drs25/Quadruped/Code/UBERMODEL/gait_model.keras")